//! Prediction helpers for trained trees and ensembles.

#[cfg(any(feature = "infer-row-parallel", feature = "infer-tree-parallel"))]
use rayon::prelude::*;

use crate::dataset::DenseMatrix;
use crate::error::{Result, XgbError};
use crate::model::Objective;
use crate::tree::BoosterTree;

#[cfg(feature = "infer-row-parallel")]
const ROW_PARALLEL_MIN_ROWS: usize = 4;

#[cfg(feature = "infer-tree-parallel")]
const TREE_PARALLEL_MIN_TREES: usize = 8;

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
            if base_margins.len() != 1 {
                return Err(XgbError::invalid_model_format(format!(
                    "scalar objectives require exactly one base margin, got {}",
                    base_margins.len()
                )));
            }

            predict_dense_scalar(objective, base_margins[0], trees, features)
        }
        Objective::MultiSoftprob { num_class } => {
            let mut margins = predict_dense_multiclass_margins(
                num_class,
                base_margins,
                tree_info,
                trees,
                features,
            )?;
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
            )?;
            Ok(margins
                .chunks_exact(num_class)
                .map(best_class_index)
                .map(class_index_to_f64)
                .collect())
        }
    }
}

#[cfg(feature = "infer-serial")]
fn predict_dense_scalar(
    objective: Objective,
    base_margin: f64,
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    predict_dense_scalar_serial(objective, base_margin, trees, features)
}

#[cfg(feature = "infer-row-parallel")]
fn predict_dense_scalar(
    objective: Objective,
    base_margin: f64,
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    if features.n_rows() < ROW_PARALLEL_MIN_ROWS {
        return predict_dense_scalar_serial(objective, base_margin, trees, features);
    }

    predict_dense_scalar_row_parallel(objective, base_margin, trees, features)
}

#[cfg(feature = "infer-tree-parallel")]
fn predict_dense_scalar(
    objective: Objective,
    base_margin: f64,
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    if trees.len() < TREE_PARALLEL_MIN_TREES {
        return predict_dense_scalar_serial(objective, base_margin, trees, features);
    }

    predict_dense_scalar_tree_parallel(objective, base_margin, trees, features)
}

fn predict_dense_scalar_serial(
    objective: Objective,
    base_margin: f64,
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    let mut predictions = vec![base_margin; features.n_rows()];
    for (row_idx, prediction) in predictions.iter_mut().enumerate() {
        for (tree_idx, tree) in trees.iter().enumerate() {
            *prediction += predict_tree_row(tree, features, row_idx).map_err(|error| {
                error.with_model_context(format!("tree {tree_idx}, row {row_idx}"))
            })?;
        }

        *prediction = match objective {
            Objective::Regression => *prediction,
            Objective::BinaryLogistic => sigmoid(*prediction),
            Objective::MultiSoftprob { .. } | Objective::MultiSoftmax { .. } => {
                return Err(XgbError::invalid_model_format(
                    "multiclass objective is not scalar",
                ));
            }
        };
    }

    Ok(predictions)
}

#[cfg(feature = "infer-row-parallel")]
fn predict_dense_scalar_row_parallel(
    objective: Objective,
    base_margin: f64,
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    (0..features.n_rows())
        .into_par_iter()
        .map(|row_idx| {
            let mut prediction = base_margin;
            for (tree_idx, tree) in trees.iter().enumerate() {
                prediction += predict_tree_row(tree, features, row_idx).map_err(|error| {
                    error.with_model_context(format!("tree {tree_idx}, row {row_idx}"))
                })?;
            }

            match objective {
                Objective::Regression => Ok(prediction),
                Objective::BinaryLogistic => Ok(sigmoid(prediction)),
                Objective::MultiSoftprob { .. } | Objective::MultiSoftmax { .. } => Err(
                    XgbError::invalid_model_format("multiclass objective is not scalar"),
                ),
            }
        })
        .collect()
}

#[cfg(feature = "infer-tree-parallel")]
fn predict_dense_scalar_tree_parallel(
    objective: Objective,
    base_margin: f64,
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    let chunk_count = usize::min(rayon::current_num_threads(), trees.len());
    let chunk_ranges = tree_chunk_ranges(trees.len(), chunk_count);

    let mut predictions = Vec::with_capacity(features.n_rows());
    for row_idx in 0..features.n_rows() {
        let partials = chunk_ranges
            .par_iter()
            .map(|(start, end)| {
                let mut partial = 0.0;
                for (offset, tree) in trees[*start..*end].iter().enumerate() {
                    let tree_idx = *start + offset;
                    partial += predict_tree_row(tree, features, row_idx).map_err(|error| {
                        error.with_model_context(format!("tree {tree_idx}, row {row_idx}"))
                    })?;
                }
                Ok(partial)
            })
            .collect::<Result<Vec<_>>>()?;

        let prediction = match objective {
            Objective::Regression => base_margin + kahan_sum(&partials),
            Objective::BinaryLogistic => sigmoid(base_margin + kahan_sum(&partials)),
            Objective::MultiSoftprob { .. } | Objective::MultiSoftmax { .. } => {
                return Err(XgbError::invalid_model_format(
                    "multiclass objective is not scalar",
                ));
            }
        };
        predictions.push(prediction);
    }

    Ok(predictions)
}

#[cfg(feature = "infer-serial")]
fn predict_dense_multiclass_margins(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    predict_dense_multiclass_margins_serial(num_class, base_margins, tree_info, trees, features)
}

#[cfg(feature = "infer-row-parallel")]
fn predict_dense_multiclass_margins(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    if features.n_rows() < ROW_PARALLEL_MIN_ROWS {
        return predict_dense_multiclass_margins_serial(
            num_class,
            base_margins,
            tree_info,
            trees,
            features,
        );
    }

    predict_dense_multiclass_margins_row_parallel(
        num_class,
        base_margins,
        tree_info,
        trees,
        features,
    )
}

#[cfg(feature = "infer-tree-parallel")]
fn predict_dense_multiclass_margins(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    if trees.len() < TREE_PARALLEL_MIN_TREES {
        return predict_dense_multiclass_margins_serial(
            num_class,
            base_margins,
            tree_info,
            trees,
            features,
        );
    }

    predict_dense_multiclass_margins_tree_parallel(
        num_class,
        base_margins,
        tree_info,
        trees,
        features,
    )
}

fn predict_dense_multiclass_margins_serial(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    let n_rows = features.n_rows();
    let margin_len = validate_multiclass_inputs(num_class, base_margins, tree_info, trees, n_rows)?;
    let mut margins = vec![0.0; margin_len];

    for row_idx in 0..n_rows {
        let start = row_idx * num_class;
        let end = start + num_class;
        margins[start..end].copy_from_slice(base_margins);
    }

    for (tree_idx, tree) in trees.iter().enumerate() {
        let output_group = tree_info[tree_idx];
        if output_group >= num_class {
            return Err(XgbError::invalid_model_format(format!(
                "tree_info[{tree_idx}] out of bounds for {num_class} classes: {output_group}",
            )));
        }

        for row_idx in 0..n_rows {
            let margin_idx = row_idx * num_class + output_group;
            margins[margin_idx] += predict_tree_row(tree, features, row_idx).map_err(|error| {
                error.with_model_context(format!("tree {tree_idx}, row {row_idx}"))
            })?;
        }
    }

    Ok(margins)
}

#[cfg(feature = "infer-row-parallel")]
fn predict_dense_multiclass_margins_row_parallel(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    let n_rows = features.n_rows();
    let margin_len = validate_multiclass_inputs(num_class, base_margins, tree_info, trees, n_rows)?;
    let mut margins = vec![0.0; margin_len];

    margins.par_chunks_mut(num_class).enumerate().try_for_each(
        |(row_idx, row_margins)| -> Result<()> {
            row_margins.copy_from_slice(base_margins);

            for (tree_idx, tree) in trees.iter().enumerate() {
                let output_group = tree_info[tree_idx];
                row_margins[output_group] +=
                    predict_tree_row(tree, features, row_idx).map_err(|error| {
                        error.with_model_context(format!("tree {tree_idx}, row {row_idx}"))
                    })?;
            }

            Ok(())
        },
    )?;

    Ok(margins)
}

#[cfg(feature = "infer-tree-parallel")]
fn predict_dense_multiclass_margins_tree_parallel(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    features: &DenseMatrix,
) -> Result<Vec<f64>> {
    let n_rows = features.n_rows();
    let margin_len = validate_multiclass_inputs(num_class, base_margins, tree_info, trees, n_rows)?;
    let mut margins = vec![0.0; margin_len];

    let chunk_count = usize::min(rayon::current_num_threads(), trees.len());
    let chunk_ranges = tree_chunk_ranges(trees.len(), chunk_count);

    for row_idx in 0..n_rows {
        let partials = chunk_ranges
            .par_iter()
            .map(|(start, end)| {
                let mut partial = vec![0.0; num_class];
                for (offset, tree) in trees[*start..*end].iter().enumerate() {
                    let tree_idx = *start + offset;
                    let output_group = tree_info[tree_idx];
                    partial[output_group] +=
                        predict_tree_row(tree, features, row_idx).map_err(|error| {
                            error.with_model_context(format!("tree {tree_idx}, row {row_idx}"))
                        })?;
                }
                Ok(partial)
            })
            .collect::<Result<Vec<_>>>()?;

        let start = row_idx * num_class;
        let end = start + num_class;
        let row_margins = &mut margins[start..end];
        row_margins.copy_from_slice(base_margins);

        let mut compensation = vec![0.0; num_class];
        for partial in partials {
            for (class_idx, value) in partial.into_iter().enumerate() {
                let adjusted = value - compensation[class_idx];
                let updated = row_margins[class_idx] + adjusted;
                compensation[class_idx] = (updated - row_margins[class_idx]) - adjusted;
                row_margins[class_idx] = updated;
            }
        }
    }

    Ok(margins)
}

fn validate_multiclass_inputs(
    num_class: usize,
    base_margins: &[f64],
    tree_info: &[usize],
    trees: &[BoosterTree],
    n_rows: usize,
) -> Result<usize> {
    if base_margins.len() != num_class {
        return Err(XgbError::invalid_model_format(format!(
            "multiclass base_margins length mismatch: expected {num_class}, got {}",
            base_margins.len()
        )));
    }

    if tree_info.len() != trees.len() {
        return Err(XgbError::invalid_model_format(format!(
            "tree_info length mismatch: expected {}, got {}",
            trees.len(),
            tree_info.len()
        )));
    }

    if let Some((tree_idx, output_group)) = tree_info
        .iter()
        .enumerate()
        .find(|(_, output_group)| **output_group >= num_class)
    {
        return Err(XgbError::invalid_model_format(format!(
            "tree_info[{tree_idx}] out of bounds for {num_class} classes: {output_group}",
        )));
    }

    let margin_len = n_rows
        .checked_mul(num_class)
        .ok_or(XgbError::InvalidShape {
            context: "multiclass prediction buffer",
            expected: usize::MAX,
            actual: n_rows,
        })?;

    Ok(margin_len)
}

#[cfg(feature = "infer-tree-parallel")]
fn tree_chunk_ranges(num_trees: usize, chunk_count: usize) -> Vec<(usize, usize)> {
    if num_trees == 0 || chunk_count == 0 {
        return Vec::new();
    }

    let base_size = num_trees / chunk_count;
    let remainder = num_trees % chunk_count;
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut start = 0;

    for chunk_idx in 0..chunk_count {
        let chunk_size = base_size + usize::from(chunk_idx < remainder);
        if chunk_size == 0 {
            continue;
        }

        let end = start + chunk_size;
        chunk_ranges.push((start, end));
        start = end;
    }

    chunk_ranges
}

#[cfg(feature = "infer-tree-parallel")]
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;

    for value in values {
        let adjusted = *value - compensation;
        let updated = sum + adjusted;
        compensation = (updated - sum) - adjusted;
        sum = updated;
    }

    sum
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
/// # Errors
///
/// Returns [`XgbError::InvalidShape`] when `row_idx` is out of bounds.
/// Returns [`XgbError::InvalidModelFormat`] when the tree contains invalid
/// structure or split metadata.
pub fn predict_tree_row(tree: &BoosterTree, features: &DenseMatrix, row_idx: usize) -> Result<f64> {
    if row_idx >= features.n_rows() {
        return Err(XgbError::InvalidShape {
            context: "row index",
            expected: features.n_rows(),
            actual: row_idx,
        });
    }

    if tree.nodes.is_empty() {
        return Err(XgbError::invalid_model_format(
            "trees must contain at least one node",
        ));
    }

    let mut node_idx = 0;
    let mut steps = 0;

    loop {
        let node = tree.nodes.get(node_idx).ok_or_else(|| {
            XgbError::invalid_model_format(format!(
                "tree node index out of bounds: node_idx={node_idx}, node_count={}",
                tree.nodes.len()
            ))
        })?;

        if let Some(value) = node.leaf_value {
            return Ok(value);
        }

        if steps >= tree.nodes.len() {
            return Err(XgbError::invalid_model_format(format!(
                "tree traversal exceeded node count at node index {node_idx}",
            )));
        }
        steps += 1;

        let split_feature = node.split_feature.ok_or_else(|| {
            XgbError::invalid_model_format(format!(
                "split node at index {node_idx} must contain split_feature",
            ))
        })?;
        if split_feature >= features.n_cols() {
            return Err(XgbError::invalid_model_format(format!(
                "split feature index out of bounds at node {node_idx}: split_feature={split_feature}, feature_count={}",
                features.n_cols()
            )));
        }

        let split_value = node.split_value.ok_or_else(|| {
            XgbError::invalid_model_format(format!(
                "split node at index {node_idx} must contain split_value",
            ))
        })?;
        if !split_value.is_finite() {
            return Err(XgbError::invalid_model_format(format!(
                "split value must be finite at node index {node_idx}",
            )));
        }

        let feature_value = features.try_value(row_idx, split_feature)?;
        let goes_left = if features.is_missing_value(feature_value) {
            node.default_left
        } else {
            xgb_less_than(feature_value, split_value)
        };
        let next_idx = if goes_left {
            node.left_child.ok_or_else(|| {
                XgbError::invalid_model_format(format!(
                    "split node at index {node_idx} must contain left_child",
                ))
            })?
        } else {
            node.right_child.ok_or_else(|| {
                XgbError::invalid_model_format(format!(
                    "split node at index {node_idx} must contain right_child",
                ))
            })?
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

#[cfg(test)]
mod tests {
    use super::{predict_dense, predict_tree_row};
    use crate::dataset::DenseMatrix;
    use crate::error::XgbError;
    use crate::model::Objective;
    use crate::tree::{BoosterTree, TreeNode};

    #[test]
    fn predict_tree_row_rejects_missing_split_metadata() {
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
        let features = DenseMatrix::from_shape_vec(1, 1, vec![0.0]).unwrap();

        let error = predict_tree_row(&tree, &features, 0).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn predict_tree_row_rejects_cyclic_tree() {
        let tree = BoosterTree {
            nodes: vec![TreeNode {
                split_feature: Some(0),
                split_bin: None,
                split_value: Some(1.0),
                left_child: Some(0),
                right_child: Some(0),
                leaf_value: None,
                default_left: true,
            }],
        };
        let features = DenseMatrix::from_shape_vec(1, 1, vec![0.0]).unwrap();

        let error = predict_tree_row(&tree, &features, 0).unwrap_err();

        assert!(matches!(error, XgbError::InvalidModelFormat(_)));
    }

    #[test]
    fn predict_dense_error_includes_tree_and_row_context() {
        let tree = BoosterTree {
            nodes: vec![TreeNode {
                split_feature: None,
                split_bin: None,
                split_value: Some(1.0),
                left_child: Some(1),
                right_child: Some(2),
                leaf_value: None,
                default_left: true,
            }],
        };
        let features = DenseMatrix::from_shape_vec(1, 1, vec![0.0]).unwrap();

        let error =
            predict_dense(Objective::Regression, &[0.0], &[0], &[tree], &features, 1).unwrap_err();

        let XgbError::InvalidModelFormat(message) = error else {
            panic!("expected InvalidModelFormat error")
        };
        assert!(message.contains("tree 0, row 0"));
    }

    #[cfg(feature = "infer-tree-parallel")]
    #[test]
    fn tree_parallel_predictions_are_deterministic_across_runs() {
        let trees = (0..32)
            .map(|index| BoosterTree {
                nodes: vec![TreeNode::leaf(f64::from(index) * 0.01)],
            })
            .collect::<Vec<_>>();
        let tree_info = vec![0; trees.len()];
        let features = DenseMatrix::from_shape_vec(3, 1, vec![0.0, 1.0, 2.0]).unwrap();

        let baseline = predict_dense(
            Objective::Regression,
            &[0.5],
            &tree_info,
            &trees,
            &features,
            1,
        )
        .unwrap();

        for _ in 0..20 {
            let current = predict_dense(
                Objective::Regression,
                &[0.5],
                &tree_info,
                &trees,
                &features,
                1,
            )
            .unwrap();
            assert_eq!(current, baseline);
        }
    }
}
