//! Feature cut generation for histogram-based training.

use serde::{Deserialize, Serialize};

use crate::dataset::DenseMatrix;
use crate::error::{Result, XGBError};

/// Per-feature cut points used to quantize dense values into bins.
///
/// For a feature with `n` cuts there are `n + 1` non-missing bins.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureCuts {
    pub values: Vec<Vec<f64>>,
}

impl FeatureCuts {
    /// Build cut points for every feature in a dense matrix.
    ///
    /// The current implementation uses a simple in-memory sorted-value strategy.
    /// This is intentionally correct-first and does not attempt to replicate the
    /// full upstream quantile sketch.
    ///
    /// # Errors
    ///
    /// Returns [`XGBError::InvalidParameter`] if `max_bin < 2`.
    pub fn build(features: &DenseMatrix, max_bin: usize) -> Result<Self> {
        if max_bin < 2 {
            return Err(XGBError::InvalidParameter {
                name: "max_bin",
                reason: "must be at least 2",
            });
        }

        let mut values = Vec::with_capacity(features.n_cols());
        for col_idx in 0..features.n_cols() {
            values.push(Self::build_feature_cuts(features, col_idx, max_bin));
        }

        Ok(Self { values })
    }

    /// Return the number of features represented in this cut table.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.values.len()
    }

    /// Return the cut points for one feature.
    #[must_use]
    pub fn feature(&self, feature_idx: usize) -> &[f64] {
        &self.values[feature_idx]
    }

    /// Return the number of non-missing bins for one feature.
    #[must_use]
    pub fn bin_count(&self, feature_idx: usize) -> usize {
        self.feature(feature_idx).len() + 1
    }

    /// Map a non-missing value to its bin index.
    #[must_use]
    pub fn find_bin(&self, feature_idx: usize, value: f64) -> usize {
        self.feature(feature_idx)
            .partition_point(|cut| value > *cut)
    }

    fn build_feature_cuts(features: &DenseMatrix, col_idx: usize, max_bin: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(features.n_rows());
        for row_idx in 0..features.n_rows() {
            let value = features.value(row_idx, col_idx);
            if !features.is_missing_value(value) {
                values.push(value);
            }
        }

        if values.len() <= 1 {
            return Vec::new();
        }

        values.sort_by(f64::total_cmp);

        // Cut points are stored as right edges. Values `<= cut[i]` fall into bin `i`.
        let mut cuts = Vec::with_capacity(max_bin.saturating_sub(1));
        for bin_idx in 1..max_bin {
            let right_edge = (values.len() * bin_idx).div_ceil(max_bin);
            if right_edge >= values.len() {
                continue;
            }

            let cut = values[right_edge - 1];
            if cuts.last().copied() != Some(cut) {
                cuts.push(cut);
            }
        }

        cuts
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::DenseMatrix;

    use super::FeatureCuts;

    fn assert_slice_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected) {
            assert!((lhs - rhs).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn cuts_ignore_missing_values() {
        let features = DenseMatrix::with_missing(3, 1, vec![1.0, -1.0, 3.0], Some(-1.0)).unwrap();
        let cuts = FeatureCuts::build(&features, 4).unwrap();

        assert_slice_close(cuts.feature(0), &[1.0]);
    }

    #[test]
    fn cuts_respect_max_bin_budget() {
        let features =
            DenseMatrix::from_shape_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let cuts = FeatureCuts::build(&features, 3).unwrap();

        assert_slice_close(cuts.feature(0), &[2.0, 4.0]);
        assert_eq!(cuts.bin_count(0), 3);
    }
}
