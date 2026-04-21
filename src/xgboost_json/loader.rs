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
    let learner = model.learner;

    if learner.gradient_booster.name != "gbtree" {
        return Err(XgbError::UnsupportedModel {
            context: "gradient booster",
            value: learner.gradient_booster.name,
        });
    }

    let num_target = parse_usize_field(
        &learner.model_param.num_target,
        "learner_model_param.num_target",
    )?;
    if num_target != 1 {
        return Err(XgbError::UnsupportedModel {
            context: "num_target",
            value: num_target.to_string(),
        });
    }

    let num_class = parse_usize_field(
        &learner.model_param.num_class,
        "learner_model_param.num_class",
    )?;
    let objective = parse_objective(&learner.objective.name, num_class)?;

    let booster_model = learner.gradient_booster.model;
    let tree_info = booster_model.tree_info;
    let json_trees = booster_model.trees;

    if tree_info.len() != json_trees.len() {
        return Err(XgbError::InvalidModelFormat(
            "tree_info length does not match trees length",
        ));
    }

    validate_tree_info_groups(&tree_info, objective.output_groups())?;

    let base_margins = parse_base_score(&learner.model_param.base_score, objective)?;
    let n_features = parse_usize_field(
        &learner.model_param.num_feature,
        "learner_model_param.num_feature",
    )?;
    let trees = json_trees
        .into_iter()
        .enumerate()
        .map(|(tree_idx, tree)| build_tree_from_json(tree_idx, tree, n_features))
        .collect::<Result<Vec<_>>>()?;

    XgbModel::from_parts(objective, base_margins, n_features, trees, tree_info)
}

fn parse_objective(raw: &str, num_class: usize) -> Result<Objective> {
    match raw {
        "reg:squarederror" => {
            if num_class > 1 {
                return Err(XgbError::InvalidModelFormat(
                    "reg:squarederror expects num_class to be 0 or 1",
                ));
            }
            Ok(Objective::Regression)
        }
        "binary:logistic" => {
            if num_class > 1 {
                return Err(XgbError::InvalidModelFormat(
                    "binary:logistic expects num_class to be 0 or 1",
                ));
            }
            Ok(Objective::BinaryLogistic)
        }
        "multi:softprob" => {
            if num_class < 2 {
                return Err(XgbError::InvalidModelFormat(
                    "multi:softprob requires num_class >= 2",
                ));
            }
            Ok(Objective::MultiSoftprob { num_class })
        }
        "multi:softmax" => {
            if num_class < 2 {
                return Err(XgbError::InvalidModelFormat(
                    "multi:softmax requires num_class >= 2",
                ));
            }
            Ok(Objective::MultiSoftmax { num_class })
        }
        _ => Err(XgbError::UnsupportedModel {
            context: "objective",
            value: raw.to_owned(),
        }),
    }
}

fn validate_tree_info_groups(tree_info: &[usize], output_groups: usize) -> Result<()> {
    if tree_info.iter().any(|group| *group >= output_groups) {
        return Err(XgbError::InvalidModelFormat(
            "tree_info contains out-of-range output group index",
        ));
    }

    Ok(())
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

enum RawBaseScore {
    Scalar(f64),
    Vector(Vec<f64>),
}

fn parse_base_score(raw: &str, objective: Objective) -> Result<Vec<f64>> {
    let raw_base_score = parse_raw_base_score(raw)?;

    match objective {
        Objective::Regression => {
            let value = parse_scalar_base_score(raw_base_score)?;
            Ok(vec![xgb_f32(value)])
        }
        Objective::BinaryLogistic => {
            let value = parse_scalar_base_score(raw_base_score)?;
            Ok(vec![logit(xgb_f32(value))?])
        }
        Objective::MultiSoftprob { num_class } | Objective::MultiSoftmax { num_class } => {
            parse_multiclass_base_score(raw_base_score, num_class)
        }
    }
}

fn parse_raw_base_score(raw: &str) -> Result<RawBaseScore> {
    if let Ok(value) = raw.parse::<f64>() {
        return Ok(RawBaseScore::Scalar(value));
    }

    if let Ok(values) = serde_json::from_str::<Vec<f64>>(raw) {
        return Ok(RawBaseScore::Vector(values));
    }

    Err(XgbError::InvalidModelFormat(
        "base_score must be a scalar or vector",
    ))
}

fn parse_scalar_base_score(raw_base_score: RawBaseScore) -> Result<f64> {
    match raw_base_score {
        RawBaseScore::Scalar(value) => Ok(value),
        RawBaseScore::Vector(values) => {
            if values.len() == 1 {
                Ok(values[0])
            } else {
                Err(XgbError::InvalidModelFormat(
                    "base_score must contain exactly one value for scalar objectives",
                ))
            }
        }
    }
}

fn parse_multiclass_base_score(raw_base_score: RawBaseScore, num_class: usize) -> Result<Vec<f64>> {
    match raw_base_score {
        RawBaseScore::Scalar(value) => Ok(vec![xgb_f32(value); num_class]),
        RawBaseScore::Vector(values) => {
            if values.len() != num_class {
                return Err(XgbError::InvalidShape {
                    context: "base_score",
                    expected: num_class,
                    actual: values.len(),
                });
            }

            Ok(values.into_iter().map(xgb_f32).collect())
        }
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
    if !(0.0 < probability && probability < 1.0) {
        return Err(XgbError::InvalidModelFormat(
            "binary:logistic base_score must be strictly between 0 and 1",
        ));
    }

    Ok((probability / (1.0 - probability)).ln())
}
