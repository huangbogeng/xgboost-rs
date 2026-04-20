//! Prediction helpers for trained trees and ensembles.

use crate::dataset::DenseMatrix;
use crate::error::{Result, XGBError};
use crate::model::PredictionTask;
use crate::tree::RegressionTree;

/// Predict a batch of rows using a fitted tree ensemble.
///
/// Predictions start from `base_margin`, then add the contribution of each tree,
/// and finally apply the supported task-specific output transform.
///
/// # Errors
///
/// Returns [`XGBError::FeatureCountMismatch`] if `features` does not match the
/// expected number of feature columns.
pub fn predict_ensemble(
    task: PredictionTask,
    base_margin: f64,
    trees: &[RegressionTree],
    features: &DenseMatrix,
    expected_feature_count: usize,
) -> Result<Vec<f64>> {
    if features.n_cols() != expected_feature_count {
        return Err(XGBError::FeatureCountMismatch {
            expected: expected_feature_count,
            actual: features.n_cols(),
        });
    }

    let mut predictions = vec![base_margin; features.n_rows()];
    for (row_idx, prediction) in predictions.iter_mut().enumerate() {
        for tree in trees {
            *prediction += predict_tree(tree, features, row_idx);
        }

        *prediction = match task {
            PredictionTask::Regression => *prediction,
            PredictionTask::BinaryLogistic => sigmoid(*prediction),
        };
    }

    Ok(predictions)
}

/// Predict one row using one trained regression tree.
///
/// Missing values follow the node's `default_left` branch.
///
/// # Panics
///
/// Panics if the tree contains an invalid split node that is missing required
/// split metadata or child indices.
#[must_use]
pub fn predict_tree(tree: &RegressionTree, features: &DenseMatrix, row_idx: usize) -> f64 {
    let mut node_idx = 0;

    loop {
        let node = &tree.nodes[node_idx];
        if let Some(value) = node.leaf_value {
            return value;
        }

        let split_feature = node
            .split_feature
            .expect("split nodes must contain split_feature");
        let split_value = node
            .split_value
            .expect("split nodes must contain split_value");
        let feature_value = features.value(row_idx, split_feature);
        let goes_left = if features.is_missing_value(feature_value) {
            node.default_left
        } else {
            xgb_less_than(feature_value, split_value)
        };
        let next_idx = if goes_left {
            node.left_child
                .expect("split nodes must contain left_child")
        } else {
            node.right_child
                .expect("split nodes must contain right_child")
        };
        node_idx = next_idx;
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "official XGBoost tree traversal compares feature values as f32"
)]
fn xgb_less_than(feature_value: f64, split_value: f64) -> bool {
    (feature_value as f32) < (split_value as f32)
}

#[must_use]
pub(crate) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let exp_neg = (-value).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = value.exp();
        exp_pos / (1.0 + exp_pos)
    }
}
