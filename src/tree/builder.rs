//! Single-tree training using histogram-based split search.

use crate::dataset::DMatrix;
use crate::error::{Result, XGBError};
use crate::grad::GradPair;
use crate::hist::bin_matrix::BinMatrix;
use crate::hist::cuts::FeatureCuts;
use crate::hist::histogram::Histogram;
use crate::params::XGBRegressorParams;
use crate::tree::split::{NodeStats, SplitCandidate, leaf_weight, split_gain};
use crate::tree::{RegressionTree, TreeNode};

#[derive(Debug, Clone)]
struct NodeEntry {
    node_idx: usize,
    depth: usize,
    row_indices: Vec<usize>,
    stats: NodeStats,
}

#[derive(Debug, Clone)]
pub struct TreeBuilder {
    params: XGBRegressorParams,
}

impl TreeBuilder {
    /// Create a tree builder configured from regression parameters.
    #[must_use]
    pub fn new(params: XGBRegressorParams) -> Self {
        Self { params }
    }

    /// Build one regression tree from precomputed quantized training data.
    ///
    /// The current implementation is intentionally simple:
    ///
    /// - explicit row indices per node
    /// - full histogram rebuild per expanded node
    /// - exhaustive scan over all feature bins
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes of the training matrix, feature cuts,
    /// bin matrix, and gradient vector do not agree.
    pub fn build(
        &self,
        train: &DMatrix,
        cuts: &FeatureCuts,
        bin_matrix: &BinMatrix,
        gradients: &[GradPair],
    ) -> Result<RegressionTree> {
        if gradients.len() != train.n_rows() {
            return Err(XGBError::InvalidShape {
                context: "gradients",
                expected: train.n_rows(),
                actual: gradients.len(),
            });
        }

        if cuts.n_features() != train.n_cols() {
            return Err(XGBError::InvalidShape {
                context: "feature cuts",
                expected: train.n_cols(),
                actual: cuts.n_features(),
            });
        }

        if bin_matrix.n_rows != train.n_rows() || bin_matrix.n_cols != train.n_cols() {
            return Err(XGBError::InvalidShape {
                context: "bin matrix",
                expected: train.n_rows() * train.n_cols(),
                actual: bin_matrix.bins.len(),
            });
        }

        let root_rows: Vec<usize> = (0..train.n_rows()).collect();
        let root_stats = Self::sum_gradients(gradients, &root_rows);
        let root_value = self.leaf_value(root_stats);
        let mut tree = RegressionTree::new(TreeNode::leaf(root_value));
        let mut stack = vec![NodeEntry {
            node_idx: 0,
            depth: 0,
            row_indices: root_rows,
            stats: root_stats,
        }];

        while let Some(node) = stack.pop() {
            if node.depth >= self.params.max_depth || node.row_indices.len() <= 1 {
                continue;
            }

            let histogram = Histogram::build(cuts, bin_matrix, gradients, &node.row_indices);
            let Some(candidate) = self.find_best_split(cuts, &histogram, node.stats) else {
                continue;
            };

            // Rows are partitioned explicitly for now. This keeps the logic easy
            // to validate before introducing shared row buffers or histogram
            // subtraction.
            let (left_rows, right_rows) =
                Self::partition_rows(node.row_indices, bin_matrix, candidate);
            if left_rows.is_empty() || right_rows.is_empty() {
                continue;
            }

            let left_idx = tree.nodes.len();
            tree.nodes
                .push(TreeNode::leaf(self.leaf_value(candidate.left_stats)));
            let right_idx = tree.nodes.len();
            tree.nodes
                .push(TreeNode::leaf(self.leaf_value(candidate.right_stats)));

            tree.nodes[node.node_idx] = TreeNode {
                split_feature: Some(candidate.feature_idx),
                split_bin: Some(candidate.split_bin),
                split_value: Some(candidate.split_value),
                left_child: Some(left_idx),
                right_child: Some(right_idx),
                leaf_value: None,
                default_left: candidate.default_left,
            };

            stack.push(NodeEntry {
                node_idx: right_idx,
                depth: node.depth + 1,
                row_indices: right_rows,
                stats: candidate.right_stats,
            });
            stack.push(NodeEntry {
                node_idx: left_idx,
                depth: node.depth + 1,
                row_indices: left_rows,
                stats: candidate.left_stats,
            });
        }

        Ok(tree)
    }

    #[must_use]
    pub fn leaf_only_tree(value: f64) -> RegressionTree {
        RegressionTree::new(TreeNode::leaf(value))
    }

    fn leaf_value(&self, stats: NodeStats) -> f64 {
        leaf_weight(stats, self.params.lambda) * self.params.learning_rate
    }

    fn sum_gradients(gradients: &[GradPair], row_indices: &[usize]) -> NodeStats {
        let mut stats = NodeStats::default();
        for &row_idx in row_indices {
            let gradient = gradients[row_idx];
            stats += NodeStats::new(gradient.grad, gradient.hess);
        }
        stats
    }

    fn find_best_split(
        &self,
        cuts: &FeatureCuts,
        histogram: &Histogram,
        parent_stats: NodeStats,
    ) -> Option<SplitCandidate> {
        let mut best = None;

        for feature_idx in 0..cuts.n_features() {
            let bins = histogram.feature_bins(feature_idx);
            if bins.len() <= 1 {
                continue;
            }

            let missing_stats = histogram.missing_stats(feature_idx);
            let total_non_missing = bins
                .iter()
                .fold(NodeStats::default(), |stats, bin| stats + bin.stats());
            let mut left_non_missing = NodeStats::default();

            for (split_bin, bin) in bins.iter().enumerate().take(bins.len() - 1) {
                left_non_missing += bin.stats();
                let right_non_missing = total_non_missing - left_non_missing;

                for default_left in [true, false] {
                    let (left_stats, right_stats) = if default_left {
                        (left_non_missing + missing_stats, right_non_missing)
                    } else {
                        (left_non_missing, right_non_missing + missing_stats)
                    };

                    if left_stats.hess_sum < self.params.min_child_weight
                        || right_stats.hess_sum < self.params.min_child_weight
                    {
                        continue;
                    }

                    let gain = split_gain(
                        parent_stats,
                        left_stats,
                        right_stats,
                        self.params.lambda,
                        self.params.gamma,
                    );
                    if !gain.is_finite() || gain <= 0.0 {
                        continue;
                    }

                    let candidate = SplitCandidate {
                        feature_idx,
                        split_bin,
                        split_value: cuts.feature(feature_idx)[split_bin],
                        gain,
                        left_stats,
                        right_stats,
                        default_left,
                    };

                    if best
                        .as_ref()
                        .is_none_or(|current: &SplitCandidate| candidate.gain > current.gain)
                    {
                        best = Some(candidate);
                    }
                }
            }
        }

        best
    }

    fn partition_rows(
        row_indices: Vec<usize>,
        bin_matrix: &BinMatrix,
        candidate: SplitCandidate,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();

        for row_idx in row_indices {
            let bin = bin_matrix.bin(row_idx, candidate.feature_idx);
            let goes_left = if bin == BinMatrix::MISSING_BIN {
                candidate.default_left
            } else {
                usize::from(bin) <= candidate.split_bin
            };

            if goes_left {
                left_rows.push(row_idx);
            } else {
                right_rows.push(row_idx);
            }
        }

        (left_rows, right_rows)
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::{DMatrix, DenseMatrix};
    use crate::grad::GradPair;
    use crate::hist::bin_matrix::BinMatrix;
    use crate::hist::cuts::FeatureCuts;

    use super::TreeBuilder;

    #[test]
    fn builds_a_split_for_step_like_gradients() {
        let features = DenseMatrix::from_shape_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let train = DMatrix::from_dense(features.clone(), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let cuts = FeatureCuts::build(&features, 4).unwrap();
        let bin_matrix = BinMatrix::from_dense(&features, &cuts).unwrap();
        let gradients = vec![
            GradPair::new(0.5, 1.0),
            GradPair::new(0.5, 1.0),
            GradPair::new(-0.5, 1.0),
            GradPair::new(-0.5, 1.0),
        ];
        let builder = TreeBuilder::new(crate::params::XGBRegressorParams {
            n_estimators: 1,
            max_depth: 1,
            learning_rate: 1.0,
            max_bin: 4,
            lambda: 0.0,
            gamma: 0.0,
            min_child_weight: 0.0,
            base_score: Some(0.5),
        });

        let tree = builder
            .build(&train, &cuts, &bin_matrix, &gradients)
            .unwrap();

        assert!(tree.nodes[0].leaf_value.is_none());
        assert_eq!(tree.nodes.len(), 3);
    }
}
