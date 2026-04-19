//! Histogram accumulation used during split search.

use crate::grad::GradPair;
use crate::hist::bin_matrix::BinMatrix;
use crate::hist::cuts::FeatureCuts;
use crate::tree::split::NodeStats;

/// One histogram bucket storing aggregated gradient statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HistogramBin {
    pub grad_sum: f64,
    pub hess_sum: f64,
}

impl HistogramBin {
    /// Add one row's gradient pair to this bucket.
    pub fn add(&mut self, gradient: GradPair) {
        self.grad_sum += gradient.grad;
        self.hess_sum += gradient.hess;
    }

    /// Convert this bucket into reusable node statistics.
    #[must_use]
    pub fn stats(&self) -> NodeStats {
        NodeStats::new(self.grad_sum, self.hess_sum)
    }
}

/// Per-node histogram over all feature bins.
///
/// The current implementation rebuilds a fresh histogram for every expanded
/// node. This is simple and correct, but not yet optimized.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    feature_offsets: Vec<usize>,
    bins: Vec<HistogramBin>,
    missing: Vec<NodeStats>,
}

impl Histogram {
    /// Allocate an empty histogram sized for a given cut table.
    #[must_use]
    pub fn with_cuts(cuts: &FeatureCuts) -> Self {
        let mut feature_offsets = Vec::with_capacity(cuts.n_features() + 1);
        feature_offsets.push(0);

        let mut total_bins = 0;
        for feature_idx in 0..cuts.n_features() {
            total_bins += cuts.bin_count(feature_idx);
            feature_offsets.push(total_bins);
        }

        Self {
            feature_offsets,
            bins: vec![HistogramBin::default(); total_bins],
            missing: vec![NodeStats::default(); cuts.n_features()],
        }
    }

    /// Build a histogram for one node's rows.
    #[must_use]
    pub fn build(
        cuts: &FeatureCuts,
        bin_matrix: &BinMatrix,
        gradients: &[GradPair],
        row_indices: &[usize],
    ) -> Self {
        let mut histogram = Self::with_cuts(cuts);

        for &row_idx in row_indices {
            let gradient = gradients[row_idx];
            for feature_idx in 0..bin_matrix.n_cols {
                let bin = bin_matrix.bin(row_idx, feature_idx);
                if bin == BinMatrix::MISSING_BIN {
                    histogram.missing[feature_idx] += NodeStats::new(gradient.grad, gradient.hess);
                    continue;
                }

                let offset = histogram.feature_offsets[feature_idx];
                histogram.bins[offset + usize::from(bin)].add(gradient);
            }
        }

        histogram
    }

    /// Borrow all non-missing bins for one feature.
    #[must_use]
    pub fn feature_bins(&self, feature_idx: usize) -> &[HistogramBin] {
        let start = self.feature_offsets[feature_idx];
        let end = self.feature_offsets[feature_idx + 1];
        &self.bins[start..end]
    }

    /// Return the aggregated missing-value statistics for one feature.
    #[must_use]
    pub fn missing_stats(&self, feature_idx: usize) -> NodeStats {
        self.missing[feature_idx]
    }
}
