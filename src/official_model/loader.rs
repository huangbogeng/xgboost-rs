use std::fs;
use std::path::Path;

use crate::error::{Result, XGBError};
use crate::model::XGBModel;
use crate::tree::{RegressionTree, TreeNode};

use super::schema::{JsonModel, JsonTree};

pub(crate) fn load_json_model<P: AsRef<Path>>(path: P) -> Result<XGBModel> {
    let contents = fs::read_to_string(path)?;
    let model: JsonModel = serde_json::from_str(&contents)?;
    convert_model(model)
}

pub(super) fn convert_model(model: JsonModel) -> Result<XGBModel> {
    if model.learner.gradient_booster.name != "gbtree" {
        return Err(XGBError::UnsupportedModel {
            context: "gradient booster",
            value: model.learner.gradient_booster.name,
        });
    }

    if model.learner.objective.name != "reg:squarederror" {
        return Err(XGBError::UnsupportedModel {
            context: "objective",
            value: model.learner.objective.name,
        });
    }

    let num_target = parse_usize_field(
        &model.learner.model_param.num_target,
        "learner_model_param.num_target",
    )?;
    if num_target != 1 {
        return Err(XGBError::UnsupportedModel {
            context: "num_target",
            value: num_target.to_string(),
        });
    }

    if model.learner.gradient_booster.model.tree_info.len()
        != model.learner.gradient_booster.model.trees.len()
    {
        return Err(XGBError::InvalidModelFormat(
            "tree_info length does not match trees length",
        ));
    }

    if model
        .learner
        .gradient_booster
        .model
        .tree_info
        .iter()
        .any(|group| *group != 0)
    {
        return Err(XGBError::UnsupportedModel {
            context: "tree_info output groups",
            value: format!("{:?}", model.learner.gradient_booster.model.tree_info),
        });
    }

    let base_score = parse_base_score(&model.learner.model_param.base_score)?;
    let n_features = parse_usize_field(
        &model.learner.model_param.num_feature,
        "learner_model_param.num_feature",
    )?;
    let trees = model
        .learner
        .gradient_booster
        .model
        .trees
        .into_iter()
        .enumerate()
        .map(|(tree_idx, tree)| convert_tree(tree_idx, tree, n_features))
        .collect::<Result<Vec<_>>>()?;

    XGBModel::new(base_score, n_features, trees)
}

fn convert_tree(
    expected_tree_id: usize,
    tree: JsonTree,
    n_features: usize,
) -> Result<RegressionTree> {
    if tree.id != expected_tree_id {
        return Err(XGBError::InvalidModelFormat(
            "tree id does not match tree position",
        ));
    }

    let num_nodes = parse_usize_field(&tree.tree_param.num_nodes, "tree_param.num_nodes")?;
    if tree.tree_param.size_leaf_vector != "1" {
        return Err(XGBError::UnsupportedModel {
            context: "tree_param.size_leaf_vector",
            value: tree.tree_param.size_leaf_vector,
        });
    }

    validate_tree_len(tree.left_children.len(), num_nodes, "left_children")?;
    validate_tree_len(tree.right_children.len(), num_nodes, "right_children")?;
    validate_tree_len(tree.split_indices.len(), num_nodes, "split_indices")?;
    validate_tree_len(tree.split_conditions.len(), num_nodes, "split_conditions")?;
    validate_tree_len(tree.base_weights.len(), num_nodes, "base_weights")?;
    validate_tree_len(tree.default_left.len(), num_nodes, "default_left")?;
    validate_tree_len(tree.split_type.len(), num_nodes, "split_type")?;
    validate_tree_structure(&tree, num_nodes)?;

    let nodes = (0..num_nodes)
        .map(|node_idx| convert_node(&tree, node_idx, num_nodes, n_features))
        .collect::<Result<Vec<_>>>()?;

    Ok(RegressionTree { nodes })
}

fn convert_node(
    tree: &JsonTree,
    node_idx: usize,
    num_nodes: usize,
    n_features: usize,
) -> Result<TreeNode> {
    let left = tree.left_children[node_idx];
    let right = tree.right_children[node_idx];

    if left == -1 && right == -1 {
        return Ok(TreeNode::leaf(xgb_f32(tree.base_weights[node_idx])));
    }

    if tree.split_type[node_idx] != 0 {
        return Err(XGBError::UnsupportedModel {
            context: "split_type",
            value: tree.split_type[node_idx].to_string(),
        });
    }

    let split_feature = tree.split_indices[node_idx] as usize;
    if split_feature >= n_features {
        return Err(XGBError::InvalidModelFormat(
            "split feature index out of bounds",
        ));
    }

    let left_child = usize::try_from(left)
        .map_err(|_| XGBError::InvalidModelFormat("left child index must be non-negative"))?;
    let right_child = usize::try_from(right)
        .map_err(|_| XGBError::InvalidModelFormat("right child index must be non-negative"))?;

    if left_child >= num_nodes {
        return Err(XGBError::InvalidModelFormat(
            "left child index out of bounds",
        ));
    }
    if right_child >= num_nodes {
        return Err(XGBError::InvalidModelFormat(
            "right child index out of bounds",
        ));
    }

    Ok(TreeNode {
        split_feature: Some(split_feature),
        split_bin: None,
        split_value: Some(xgb_f32(tree.split_conditions[node_idx])),
        left_child: Some(left_child),
        right_child: Some(right_child),
        leaf_value: None,
        default_left: tree.default_left[node_idx] == 1,
    })
}

fn parse_base_score(raw: &str) -> Result<f64> {
    if let Ok(value) = raw.parse::<f64>() {
        return Ok(xgb_f32(value));
    }

    if let Ok(values) = serde_json::from_str::<Vec<f64>>(raw) {
        if values.len() == 1 {
            return Ok(xgb_f32(values[0]));
        }
    }

    Err(XGBError::InvalidModelFormat(
        "base_score must be a scalar or a single-element vector",
    ))
}

fn parse_usize_field(raw: &str, context: &'static str) -> Result<usize> {
    raw.parse::<usize>()
        .map_err(|_| XGBError::InvalidModelFormat(context))
}

fn validate_tree_len(actual: usize, expected: usize, context: &'static str) -> Result<()> {
    if actual != expected {
        return Err(XGBError::InvalidShape {
            context,
            expected,
            actual,
        });
    }

    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Unvisited,
    Visiting,
    Visited,
}

fn validate_tree_structure(tree: &JsonTree, num_nodes: usize) -> Result<()> {
    if num_nodes == 0 {
        return Err(XGBError::InvalidModelFormat(
            "trees must contain at least one node",
        ));
    }

    let mut visit_state = vec![VisitState::Unvisited; num_nodes];
    let mut indegree = vec![0usize; num_nodes];
    validate_reachable_subtree(tree, 0, num_nodes, &mut visit_state, &mut indegree)?;

    if visit_state.contains(&VisitState::Unvisited) {
        return Err(XGBError::InvalidModelFormat(
            "tree contains unreachable nodes",
        ));
    }

    Ok(())
}

fn validate_reachable_subtree(
    tree: &JsonTree,
    node_idx: usize,
    num_nodes: usize,
    visit_state: &mut [VisitState],
    indegree: &mut [usize],
) -> Result<()> {
    match visit_state[node_idx] {
        VisitState::Unvisited => {}
        VisitState::Visiting => {
            return Err(XGBError::InvalidModelFormat("tree contains a cycle"));
        }
        VisitState::Visited => {
            return Ok(());
        }
    }

    visit_state[node_idx] = VisitState::Visiting;

    let default_left = tree.default_left[node_idx];
    if default_left > 1 {
        return Err(XGBError::InvalidModelFormat(
            "default_left must be encoded as 0 or 1",
        ));
    }

    let left = tree.left_children[node_idx];
    let right = tree.right_children[node_idx];
    match (left, right) {
        (-1, -1) => {}
        (-1, _) | (_, -1) => {
            return Err(XGBError::InvalidModelFormat(
                "split nodes must contain both children",
            ));
        }
        (left, right) => {
            let left_child = usize::try_from(left).map_err(|_| {
                XGBError::InvalidModelFormat("left child index must be non-negative")
            })?;
            let right_child = usize::try_from(right).map_err(|_| {
                XGBError::InvalidModelFormat("right child index must be non-negative")
            })?;

            if left_child >= num_nodes {
                return Err(XGBError::InvalidModelFormat(
                    "left child index out of bounds",
                ));
            }
            if right_child >= num_nodes {
                return Err(XGBError::InvalidModelFormat(
                    "right child index out of bounds",
                ));
            }

            indegree[left_child] += 1;
            if indegree[left_child] > 1 {
                return Err(XGBError::InvalidModelFormat(
                    "tree nodes must have exactly one parent",
                ));
            }

            indegree[right_child] += 1;
            if indegree[right_child] > 1 {
                return Err(XGBError::InvalidModelFormat(
                    "tree nodes must have exactly one parent",
                ));
            }

            validate_reachable_subtree(tree, left_child, num_nodes, visit_state, indegree)?;
            validate_reachable_subtree(tree, right_child, num_nodes, visit_state, indegree)?;
        }
    }

    visit_state[node_idx] = VisitState::Visited;
    Ok(())
}

fn xgb_f32(value: f64) -> f64 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "official XGBoost JSON stores and evaluates tree values as f32"
    )]
    f64::from(value as f32)
}
