//! Inference-only model type used by the public crate API.

use std::path::Path;

use crate::dataset::DenseMatrix;
use crate::error::{Result, XgbError};
use crate::inference;
use crate::tree::{BoosterTree, TreeNode, validate_tree_topology};
use crate::xgboost_json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Objective {
    Regression,
    BinaryLogistic,
    MultiSoftprob { num_class: usize },
    MultiSoftmax { num_class: usize },
}

impl Objective {
    pub(crate) fn output_groups(self) -> usize {
        match self {
            Self::Regression | Self::BinaryLogistic => 1,
            Self::MultiSoftprob { num_class } | Self::MultiSoftmax { num_class } => num_class,
        }
    }
}

/// Inference-only gradient-boosted tree model.
///
/// This type is the stable public model surface. Callers can either load a
/// supported official upstream `XGBoost` model file or construct a model from
/// already prepared tree structures.
#[derive(Debug, Clone, PartialEq)]
pub struct XgbModel {
    trees: Vec<BoosterTree>,
    tree_info: Vec<usize>,
    base_margins: Vec<f64>,
    n_features: usize,
    objective: Objective,
}

impl XgbModel {
    /// Create an inference model from already prepared trees.
    ///
    /// # Errors
    ///
    /// Returns [`XgbError::InvalidParameter`] if `base_score` is not finite or
    /// if `n_features == 0`.
    pub fn new(base_score: f64, n_features: usize, trees: Vec<BoosterTree>) -> Result<Self> {
        let tree_info = vec![0; trees.len()];
        Self::from_parts(
            Objective::Regression,
            vec![base_score],
            n_features,
            trees,
            tree_info,
        )
    }

    pub(crate) fn from_parts(
        objective: Objective,
        base_margins: Vec<f64>,
        n_features: usize,
        trees: Vec<BoosterTree>,
        tree_info: Vec<usize>,
    ) -> Result<Self> {
        let expected_groups = objective.output_groups();

        if base_margins.len() != expected_groups {
            return Err(XgbError::InvalidShape {
                context: "base_score",
                expected: expected_groups,
                actual: base_margins.len(),
            });
        }

        if !base_margins.iter().all(|value| value.is_finite()) {
            return Err(XgbError::InvalidParameter {
                name: "base_score",
                reason: "must be finite",
            });
        }

        if tree_info.len() != trees.len() {
            return Err(XgbError::InvalidShape {
                context: "tree_info",
                expected: trees.len(),
                actual: tree_info.len(),
            });
        }

        if tree_info.iter().any(|group| *group >= expected_groups) {
            return Err(XgbError::InvalidModelFormat(
                "tree_info contains out-of-range output group index",
            ));
        }

        if n_features == 0 {
            return Err(XgbError::InvalidParameter {
                name: "n_features",
                reason: "must be greater than zero",
            });
        }

        for tree in &trees {
            validate_tree_structure(tree, n_features)?;
        }

        Ok(Self {
            trees,
            tree_info,
            base_margins,
            n_features,
            objective,
        })
    }

    /// Return the scalar `base_score` for scalar objectives.
    ///
    /// For multiclass objectives this returns the class-0 base margin.
    #[must_use]
    pub fn base_score(&self) -> f64 {
        match self.objective {
            Objective::Regression => self.base_margins[0],
            Objective::BinaryLogistic => inference::sigmoid(self.base_margins[0]),
            Objective::MultiSoftprob { .. } | Objective::MultiSoftmax { .. } => {
                self.base_margins[0]
            }
        }
    }

    /// Return the base margins used to initialize each output group.
    #[must_use]
    pub fn base_margins(&self) -> &[f64] {
        &self.base_margins
    }

    /// Return the number of feature columns expected by this model.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Borrow the ensemble trees.
    #[must_use]
    pub fn trees(&self) -> &[BoosterTree] {
        &self.trees
    }

    /// Predict task outputs for a dense feature matrix.
    ///
    /// For supported official JSON models this returns:
    ///
    /// - regression values for `reg:squarederror`
    /// - positive-class probabilities for `binary:logistic`
    /// - class probabilities (row-major) for `multi:softprob`
    /// - class labels encoded as `f64` for `multi:softmax`
    ///
    /// # Errors
    ///
    /// Returns [`XgbError::FeatureCountMismatch`] if the feature count differs
    /// from the model expectation.
    pub fn predict_dense(&self, features: &DenseMatrix) -> Result<Vec<f64>> {
        inference::predict_dense(
            self.objective,
            &self.base_margins,
            &self.tree_info,
            &self.trees,
            features,
            self.n_features,
        )
    }

    /// Load an official upstream `XGBoost` `model.json` file.
    ///
    /// Currently supported model scope:
    ///
    /// - `booster=gbtree`
    /// - `objective=reg:squarederror`, `objective=binary:logistic`,
    ///   `objective=multi:softprob`, and `objective=multi:softmax`
    /// - single-target prediction
    /// - numerical splits only
    ///
    /// # Errors
    ///
    /// Returns an error if reading the file fails, if the JSON is malformed, or
    /// if the model uses unsupported upstream features.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        xgboost_json::load_model_json(path)
    }
}

fn validate_tree_structure(tree: &BoosterTree, n_features: usize) -> Result<()> {
    validate_tree_topology(tree.nodes.len(), |node_idx| {
        let node = &tree.nodes[node_idx];

        if let Some(leaf_value) = node.leaf_value {
            validate_leaf_node(node, leaf_value)?;
            return Ok(None);
        }

        let children = validate_split_node(node, n_features)?;
        Ok(Some(children))
    })
}

fn validate_leaf_node(node: &TreeNode, leaf_value: f64) -> Result<()> {
    if !leaf_value.is_finite() {
        return Err(XgbError::InvalidModelFormat("leaf value must be finite"));
    }

    if node.split_feature.is_some()
        || node.split_bin.is_some()
        || node.split_value.is_some()
        || node.left_child.is_some()
        || node.right_child.is_some()
    {
        return Err(XgbError::InvalidModelFormat(
            "leaf nodes must not contain split metadata",
        ));
    }

    Ok(())
}

fn validate_split_node(node: &TreeNode, n_features: usize) -> Result<(usize, usize)> {
    let split_feature = node.split_feature.ok_or(XgbError::InvalidModelFormat(
        "split nodes must contain split_feature",
    ))?;
    if split_feature >= n_features {
        return Err(XgbError::InvalidModelFormat(
            "split feature index out of bounds",
        ));
    }

    if node.split_bin.is_some() {
        return Err(XgbError::InvalidModelFormat(
            "split_bin is unsupported for inference",
        ));
    }

    let split_value = node.split_value.ok_or(XgbError::InvalidModelFormat(
        "split nodes must contain split_value",
    ))?;
    if !split_value.is_finite() {
        return Err(XgbError::InvalidModelFormat("split value must be finite"));
    }

    let left_child = node.left_child.ok_or(XgbError::InvalidModelFormat(
        "split nodes must contain left_child",
    ))?;
    let right_child = node.right_child.ok_or(XgbError::InvalidModelFormat(
        "split nodes must contain right_child",
    ))?;

    Ok((left_child, right_child))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected) {
            assert!((lhs - rhs).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn predicts_with_base_score_only() {
        let features = DenseMatrix::from_shape_vec(2, 1, vec![0.0, 1.0]).unwrap();
        let model = XgbModel::new(0.75, 1, Vec::new()).unwrap();

        let predictions = model.predict_dense(&features).unwrap();

        assert_vec_close(&predictions, &[0.75, 0.75]);
    }

    #[test]
    fn rejects_split_nodes_without_required_metadata() {
        let tree = BoosterTree {
            nodes: vec![
                TreeNode {
                    split_feature: None,
                    split_bin: None,
                    split_value: Some(1.0),
                    left_child: Some(1),
                    right_child: Some(2),
                    leaf_value: None,
                    default_left: true,
                },
                TreeNode::leaf(-1.0),
                TreeNode::leaf(1.0),
            ],
        };

        let error = XgbModel::new(0.0, 1, vec![tree]).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_split_feature_index_out_of_bounds() {
        let tree = BoosterTree {
            nodes: vec![
                TreeNode {
                    split_feature: Some(1),
                    split_bin: None,
                    split_value: Some(1.0),
                    left_child: Some(1),
                    right_child: Some(2),
                    leaf_value: None,
                    default_left: true,
                },
                TreeNode::leaf(-1.0),
                TreeNode::leaf(1.0),
            ],
        };

        let error = XgbModel::new(0.0, 1, vec![tree]).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_tree_cycles() {
        let tree = BoosterTree {
            nodes: vec![
                TreeNode {
                    split_feature: Some(0),
                    split_bin: None,
                    split_value: Some(1.0),
                    left_child: Some(1),
                    right_child: Some(2),
                    leaf_value: None,
                    default_left: true,
                },
                TreeNode {
                    split_feature: Some(0),
                    split_bin: None,
                    split_value: Some(2.0),
                    left_child: Some(0),
                    right_child: Some(3),
                    leaf_value: None,
                    default_left: true,
                },
                TreeNode::leaf(-1.0),
                TreeNode::leaf(1.0),
            ],
        };

        let error = XgbModel::new(0.0, 1, vec![tree]).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_unreachable_nodes() {
        let tree = BoosterTree {
            nodes: vec![
                TreeNode {
                    split_feature: Some(0),
                    split_bin: None,
                    split_value: Some(1.0),
                    left_child: Some(1),
                    right_child: Some(2),
                    leaf_value: None,
                    default_left: true,
                },
                TreeNode::leaf(-1.0),
                TreeNode::leaf(1.0),
                TreeNode::leaf(3.0),
            ],
        };

        let error = XgbModel::new(0.0, 1, vec![tree]).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_leaf_nodes_with_split_metadata() {
        let tree = BoosterTree {
            nodes: vec![TreeNode {
                split_feature: Some(0),
                split_bin: None,
                split_value: Some(1.0),
                left_child: Some(0),
                right_child: Some(0),
                leaf_value: Some(1.0),
                default_left: true,
            }],
        };

        let error = XgbModel::new(0.0, 1, vec![tree]).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_non_finite_leaf_value() {
        let tree = BoosterTree {
            nodes: vec![TreeNode {
                split_feature: None,
                split_bin: None,
                split_value: None,
                left_child: None,
                right_child: None,
                leaf_value: Some(f64::NAN),
                default_left: true,
            }],
        };

        let error = XgbModel::new(0.0, 1, vec![tree]).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn rejects_non_finite_split_value() {
        let tree = BoosterTree {
            nodes: vec![
                TreeNode {
                    split_feature: Some(0),
                    split_bin: None,
                    split_value: Some(f64::INFINITY),
                    left_child: Some(1),
                    right_child: Some(2),
                    leaf_value: None,
                    default_left: true,
                },
                TreeNode::leaf(-1.0),
                TreeNode::leaf(1.0),
            ],
        };

        let error = XgbModel::new(0.0, 1, vec![tree]).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }
}
