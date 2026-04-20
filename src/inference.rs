//! Prediction helpers for trained trees and ensembles.

use crate::dataset::DenseMatrix;
use crate::error::{Result, XgbError};
use crate::model::Objective;
use crate::tree::BoosterTree;

/// Predict a batch of rows using a fitted tree ensemble.
///
/// Predictions start from `base_margin`, then add the contribution of each tree,
/// and finally apply the supported task-specific output transform.
///
/// # Errors
///
/// Returns [`XgbError::FeatureCountMismatch`] if `features` does not match the
/// expected number of feature columns.
pub fn predict_dense(
    objective: Objective,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
    expected_feature_count: usize,
) -> Result<Vec<f64>> {
    if features.n_cols() != expected_feature_count {
        return Err(XgbError::FeatureCountMismatch {
            expected: expected_feature_count,
            actual: features.n_cols(),
        });
    }

    match objective {
        Objective::Regression | Objective::BinaryLogistic => {
            predict_dense_scalar(objective, base_margins[0], trees, features)
        }
        Objective::MultiSoftprob { num_class } => {
            let mut margins = predict_dense_multiclass_margins(
                num_class,
                base_margins,
                tree_info,
                trees,
                features,
            );
            for row_margins in margins.chunks_exact_mut(num_class) {
                softmax_in_place(row_margins);
            }
            Ok(margins)
        }
        Objective::MultiSoftmax { num_class } => {
            let margins = predict_dense_multiclass_margins(
                num_class,
                base_margins,
                tree_info,
                trees,
                features,
            );
            Ok(margins
                .chunks_exact(num_class)
                .map(best_class_index)
                .map(class_index_to_f64)
                .collect())
        }
    }
}

fn predict_dense_scalar(
    objective: Objective,
    base_margin: f64,
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    let mut predictions = vec![base_margin; features.n_rows()];
    for (row_idx, prediction) in predictions.iter_mut().enumerate() {
        for tree in trees {
            *prediction += predict_tree_row(tree, features, row_idx);
        }

        *prediction = match objective {
            Objective::Regression => *prediction,
            Objective::BinaryLogistic => sigmoid(*prediction),
            Objective::MultiSoftprob { .. } | Objective::MultiSoftmax { .. } => {
                return Err(XgbError::InvalidModelFormat(
                    "multiclass objective is not scalar",
                ));
            }
        };
    }

    Ok(predictions)
}

fn predict_dense_multiclass_margins(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Vec<f64> {
    let n_rows = features.n_rows();
    let mut margins = vec![0.0; n_rows * num_class];

    for row_idx in 0..n_rows {
        let start = row_idx * num_class;
        let end = start + num_class;
        margins[start..end].copy_from_slice(base_margins);
    }

    for (tree_idx, tree) in trees.iter().enumerate() {
        let output_group = tree_info[tree_idx];
        for row_idx in 0..n_rows {
            let margin_idx = row_idx * num_class + output_group;
            margins[margin_idx] += predict_tree_row(tree, features, row_idx);
        }
    }

    margins
}

fn softmax_in_place(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }

    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut denominator = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max_value).exp();
        denominator += *value;
    }

    for value in values.iter_mut() {
        *value /= denominator;
    }
}

fn best_class_index(row_values: &[f64]) -> usize {
    row_values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "class indices are intentionally exposed as f64 for predict_dense output compatibility"
)]
fn class_index_to_f64(index: usize) -> f64 {
    index as f64
}

/// Predict one row using one trained decision tree.
///
/// Missing values follow the node's `default_left` branch.
///
/// # Panics
///
/// Panics if the tree contains an invalid split node that is missing required
/// split metadata or child indices.
#[must_use]
pub fn predict_tree_row(tree: &BoosterTree, features: &DenseMatrix, row_idx: usize) -> f64 {
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
