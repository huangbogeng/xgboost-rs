use std::fs;
use std::path::Path;

use crate::error::{Result, XgbError};
use crate::model::{Objective, XgbModel};
use crate::tree::{BoosterTree, TreeNode};

use super::schema::{JsonModel, JsonTree};

pub(crate) fn load_model_json<P: AsRef<Path>>(path: P) -> Result<XgbModel> {
    let contents = fs::read_to_string(path)?;
    let model: JsonModel = serde_json::from_str(&contents)?;
    build_model_from_json(model)
}

pub(super) fn build_model_from_json(model: JsonModel) -> Result<XgbModel> {
    if model.learner.gradient_booster.name != "gbtree" {
        return Err(XgbError::UnsupportedModel {
            context: "gradient booster",
            value: model.learner.gradient_booster.name,
        });
    }

    let objective = parse_objective(&model.learner.objective.name)?;

    let num_target = parse_usize_field(
        &model.learner.model_param.num_target,
        "learner_model_param.num_target",
    )?;
    if num_target != 1 {
        return Err(XgbError::UnsupportedModel {
            context: "num_target",
            value: num_target.to_string(),
        });
    }

    if model.learner.gradient_booster.model.tree_info.len()
        != model.learner.gradient_booster.model.trees.len()
    {
        return Err(XgbError::InvalidModelFormat(
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
        return Err(XgbError::UnsupportedModel {
            context: "tree_info output groups",
            value: format!("{:?}", model.learner.gradient_booster.model.tree_info),
        });
    }

    let base_margin = parse_base_score(&model.learner.model_param.base_score, objective)?;
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
        .map(|(tree_idx, tree)| build_tree_from_json(tree_idx, tree, n_features))
        .collect::<Result<Vec<_>>>()?;

    XgbModel::from_parts(objective, base_margin, n_features, trees)
}

fn parse_objective(raw: &str) -> Result<Objective> {
    match raw {
        "reg:squarederror" => Ok(Objective::Regression),
        "binary:logistic" => Ok(Objective::BinaryLogistic),
        _ => Err(XgbError::UnsupportedModel {
            context: "objective",
            value: raw.to_owned(),
        }),
    }
}

fn build_tree_from_json(
    expected_tree_id: usize,
    tree: JsonTree,
    n_features: usize,
) -> Result<BoosterTree> {
    if tree.id != expected_tree_id {
        return Err(XgbError::InvalidModelFormat(
            "tree id does not match tree position",
        ));
    }

    let num_nodes = parse_usize_field(&tree.tree_param.num_nodes, "tree_param.num_nodes")?;
    if tree.tree_param.size_leaf_vector != "1" {
        return Err(XgbError::UnsupportedModel {
            context: "tree_param.size_leaf_vector",
            value: tree.tree_param.size_leaf_vector,
        });
    }

    validate_array_len(tree.left_children.len(), num_nodes, "left_children")?;
    validate_array_len(tree.right_children.len(), num_nodes, "right_children")?;
    validate_array_len(tree.split_indices.len(), num_nodes, "split_indices")?;
    validate_array_len(tree.split_conditions.len(), num_nodes, "split_conditions")?;
    validate_array_len(tree.base_weights.len(), num_nodes, "base_weights")?;
    validate_array_len(tree.default_left.len(), num_nodes, "default_left")?;
    validate_array_len(tree.split_type.len(), num_nodes, "split_type")?;
    validate_tree_structure(&tree, num_nodes)?;

    let nodes = (0..num_nodes)
        .map(|node_idx| build_node_from_json(&tree, node_idx, num_nodes, n_features))
        .collect::<Result<Vec<_>>>()?;

    Ok(BoosterTree { nodes })
}

fn build_node_from_json(
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
        return Err(XgbError::UnsupportedModel {
            context: "split_type",
            value: tree.split_type[node_idx].to_string(),
        });
    }

    let split_feature = tree.split_indices[node_idx] as usize;
    if split_feature >= n_features {
        return Err(XgbError::InvalidModelFormat(
            "split feature index out of bounds",
        ));
    }

    let left_child = usize::try_from(left)
        .map_err(|_| XgbError::InvalidModelFormat("left child index must be non-negative"))?;
    let right_child = usize::try_from(right)
        .map_err(|_| XgbError::InvalidModelFormat("right child index must be non-negative"))?;

    if left_child >= num_nodes {
        return Err(XgbError::InvalidModelFormat(
            "left child index out of bounds",
        ));
    }
    if right_child >= num_nodes {
        return Err(XgbError::InvalidModelFormat(
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

fn parse_base_score(raw: &str, objective: Objective) -> Result<f64> {
    if let Ok(value) = raw.parse::<f64>() {
        return parse_base_score_value(value, objective);
    }

    if let Ok(values) = serde_json::from_str::<Vec<f64>>(raw) {
        if values.len() == 1 {
            return parse_base_score_value(values[0], objective);
        }
    }

    Err(XgbError::InvalidModelFormat(
        "base_score must be a scalar or a single-element vector",
    ))
}

fn parse_base_score_value(value: f64, objective: Objective) -> Result<f64> {
    let base_score = xgb_f32(value);
    match objective {
        Objective::Regression => Ok(base_score),
        Objective::BinaryLogistic => logit(base_score),
    }
}

fn parse_usize_field(raw: &str, context: &'static str) -> Result<usize> {
    raw.parse::<usize>()
        .map_err(|_| XgbError::InvalidModelFormat(context))
}

fn validate_array_len(actual: usize, expected: usize, context: &'static str) -> Result<()> {
    if actual != expected {
        return Err(XgbError::InvalidShape {
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

#[derive(Clone, Copy)]
enum TraversalStep {
    Enter(usize),
    Exit(usize),
}

fn validate_tree_structure(tree: &JsonTree, num_nodes: usize) -> Result<()> {
    if num_nodes == 0 {
        return Err(XgbError::InvalidModelFormat(
            "trees must contain at least one node",
        ));
    }

    let mut visit_state = vec![VisitState::Unvisited; num_nodes];
    let mut indegree = vec![0usize; num_nodes];
    let mut stack = vec![TraversalStep::Enter(0)];

    while let Some(step) = stack.pop() {
        match step {
            TraversalStep::Enter(node_idx) => {
                match visit_state[node_idx] {
                    VisitState::Unvisited => {}
                    VisitState::Visiting => {
                        return Err(XgbError::InvalidModelFormat("tree contains a cycle"));
                    }
                    VisitState::Visited => {
                        continue;
                    }
                }

                visit_state[node_idx] = VisitState::Visiting;

                let default_left = tree.default_left[node_idx];
                if default_left > 1 {
                    return Err(XgbError::InvalidModelFormat(
                        "default_left must be encoded as 0 or 1",
                    ));
                }

                let left = tree.left_children[node_idx];
                let right = tree.right_children[node_idx];
                match (left, right) {
                    (-1, -1) => {
                        visit_state[node_idx] = VisitState::Visited;
                    }
                    (-1, _) | (_, -1) => {
                        return Err(XgbError::InvalidModelFormat(
                            "split nodes must contain both children",
                        ));
                    }
                    (left, right) => {
                        let left_child = usize::try_from(left).map_err(|_| {
                            XgbError::InvalidModelFormat("left child index must be non-negative")
                        })?;
                        let right_child = usize::try_from(right).map_err(|_| {
                            XgbError::InvalidModelFormat("right child index must be non-negative")
                        })?;

                        if left_child >= num_nodes {
                            return Err(XgbError::InvalidModelFormat(
                                "left child index out of bounds",
                            ));
                        }
                        if right_child >= num_nodes {
                            return Err(XgbError::InvalidModelFormat(
                                "right child index out of bounds",
                            ));
                        }

                        indegree[left_child] += 1;
                        if indegree[left_child] > 1 {
                            return Err(XgbError::InvalidModelFormat(
                                "tree nodes must have exactly one parent",
                            ));
                        }

                        indegree[right_child] += 1;
                        if indegree[right_child] > 1 {
                            return Err(XgbError::InvalidModelFormat(
                                "tree nodes must have exactly one parent",
                            ));
                        }

                        stack.push(TraversalStep::Exit(node_idx));
                        stack.push(TraversalStep::Enter(right_child));
                        stack.push(TraversalStep::Enter(left_child));
                    }
                }
            }
            TraversalStep::Exit(node_idx) => {
                visit_state[node_idx] = VisitState::Visited;
            }
        }
    }

    if visit_state.contains(&VisitState::Unvisited) {
        return Err(XgbError::InvalidModelFormat(
            "tree contains unreachable nodes",
        ));
    }

    Ok(())
}

fn xgb_f32(value: f64) -> f64 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "official XGBoost JSON stores and evaluates tree values as f32"
    )]
    f64::from(value as f32)
}

fn logit(probability: f64) -> Result<f64> {
    if !(0.0..1.0).contains(&probability) {
        return Err(XgbError::InvalidModelFormat(
            "binary:logistic base_score must be strictly between 0 and 1",
        ));
    }

    Ok((probability / (1.0 - probability)).ln())
}
