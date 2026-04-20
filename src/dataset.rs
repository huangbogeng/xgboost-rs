//! Dense dataset types used for inference.

use crate::error::{Result, XgbError};

/// Dense row-major feature matrix.
///
/// Values are stored in a contiguous `Vec<f64>` using row-major layout.
/// Missing values are detected by either `NaN` or the optional explicit
/// sentinel configured in [`Self::with_missing`].
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    n_rows: usize,
    n_cols: usize,
    data: Vec<f64>,
    missing: Option<f64>,
}

impl DenseMatrix {
    /// Create a dense matrix without an explicit missing-value sentinel.
    ///
    /// # Errors
    ///
    /// Returns [`XgbError::InvalidShape`] if `data.len() != n_rows * n_cols`.
    #[must_use = "constructors return a new matrix that must be used"]
    pub fn from_shape_vec(n_rows: usize, n_cols: usize, data: Vec<f64>) -> Result<Self> {
        Self::with_missing(n_rows, n_cols, data, None)
    }

    /// Create a dense matrix from row-major values.
    ///
    /// # Errors
    ///
    /// Returns [`XgbError::InvalidShape`] if `data.len() != n_rows * n_cols`.
    pub fn with_missing(
        n_rows: usize,
        n_cols: usize,
        data: Vec<f64>,
        missing: Option<f64>,
    ) -> Result<Self> {
        let expected = n_rows.checked_mul(n_cols).ok_or(XgbError::InvalidShape {
            context: "dense matrix",
            expected: usize::MAX,
            actual: data.len(),
        })?;

        if expected != data.len() {
            return Err(XgbError::InvalidShape {
                context: "dense matrix",
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            n_rows,
            n_cols,
            data,
            missing,
        })
    }

    /// Return the number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Return the number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Return the raw row-major backing storage.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Return the explicit missing-value sentinel, if one was configured.
    #[must_use]
    pub fn missing(&self) -> Option<f64> {
        self.missing
    }

    /// Check whether a value should be treated as missing.
    ///
    /// `NaN` is always considered missing.
    #[must_use]
    #[allow(
        clippy::float_cmp,
        reason = "the explicit missing sentinel is matched exactly"
    )]
    pub fn is_missing_value(&self, value: f64) -> bool {
        value.is_nan() || self.missing.is_some_and(|missing| value == missing)
    }

    /// Borrow one row by index.
    #[must_use]
    pub fn row(&self, row_idx: usize) -> &[f64] {
        let start = row_idx * self.n_cols;
        let end = start + self.n_cols;
        &self.data[start..end]
    }

    /// Return a single value by row and column index.
    #[must_use]
    pub fn value(&self, row_idx: usize, col_idx: usize) -> f64 {
        self.data[row_idx * self.n_cols + col_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::DenseMatrix;

    #[test]
    fn matrix_shape_must_match_input_length() {
        let result = DenseMatrix::from_shape_vec(2, 2, vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }
}
