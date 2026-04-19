//! Quantized dense feature storage.

use serde::{Deserialize, Serialize};

use crate::dataset::DenseMatrix;
use crate::error::{Result, XGBError};

use super::cuts::FeatureCuts;

/// Dense matrix of quantized bin ids.
///
/// Bins are stored in row-major order and align with the source [`DenseMatrix`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub bins: Vec<u16>,
}

impl BinMatrix {
    /// Sentinel value used to represent missing entries.
    pub const MISSING_BIN: u16 = u16::MAX;

    /// Quantize a dense matrix using precomputed feature cuts.
    ///
    /// # Errors
    ///
    /// Returns an error if the cut table shape does not match the feature matrix
    /// or if a computed bin index cannot be represented as `u16`.
    pub fn from_dense(features: &DenseMatrix, cuts: &FeatureCuts) -> Result<Self> {
        if features.n_cols() != cuts.n_features() {
            return Err(XGBError::InvalidShape {
                context: "feature cuts",
                expected: features.n_cols(),
                actual: cuts.n_features(),
            });
        }

        let mut bins = Vec::with_capacity(features.n_rows() * features.n_cols());
        for row_idx in 0..features.n_rows() {
            for col_idx in 0..features.n_cols() {
                let value = features.value(row_idx, col_idx);
                if features.is_missing_value(value) {
                    bins.push(Self::MISSING_BIN);
                    continue;
                }

                let bin_idx = cuts.find_bin(col_idx, value);
                let stored_bin =
                    u16::try_from(bin_idx).map_err(|_| XGBError::InvalidParameter {
                        name: "max_bin",
                        reason: "bin index exceeds u16 storage capacity",
                    })?;
                bins.push(stored_bin);
            }
        }

        Ok(Self {
            n_rows: features.n_rows(),
            n_cols: features.n_cols(),
            bins,
        })
    }

    /// Return the stored bin id for one row and feature.
    #[must_use]
    pub fn bin(&self, row_idx: usize, col_idx: usize) -> u16 {
        self.bins[row_idx * self.n_cols + col_idx]
    }

    /// Check whether one entry is the missing sentinel.
    #[must_use]
    pub fn is_missing(&self, row_idx: usize, col_idx: usize) -> bool {
        self.bin(row_idx, col_idx) == Self::MISSING_BIN
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::DenseMatrix;
    use crate::hist::cuts::FeatureCuts;

    use super::BinMatrix;

    #[test]
    fn quantizes_dense_values_into_bins() {
        let features = DenseMatrix::from_shape_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cuts = FeatureCuts::build(&features, 3).unwrap();
        let bins = BinMatrix::from_dense(&features, &cuts).unwrap();

        assert_eq!(bins.bin(0, 0), 0);
        assert_eq!(bins.bin(1, 0), 0);
        assert_eq!(bins.bin(2, 0), 1);
        assert_eq!(bins.bin(3, 0), 2);
    }

    #[test]
    fn uses_missing_sentinel_for_missing_values() {
        let features = DenseMatrix::with_missing(3, 1, vec![1.0, -1.0, 3.0], Some(-1.0)).unwrap();
        let cuts = FeatureCuts::build(&features, 4).unwrap();
        let bins = BinMatrix::from_dense(&features, &cuts).unwrap();

        assert!(bins.is_missing(1, 0));
        assert_eq!(bins.bin(0, 0), 0);
        assert_eq!(bins.bin(2, 0), 1);
    }
}
