//! Objective functions for training.

use crate::error::{Result, XGBError};
use crate::grad::GradPair;

const MAX_EXACT_F64_INT: usize = 1 << 53;

#[allow(
    clippy::cast_precision_loss,
    reason = "values up to 2^53 convert exactly to f64"
)]
fn exact_len_as_f64(len: usize) -> f64 {
    len as f64
}

/// Objective behavior required by the boosting loop.
///
/// The crate currently ships only [`SquaredError`], but this trait defines the
/// boundary between the booster and future regression objectives.
pub trait Objective {
    /// Compute per-row gradient pairs from predictions and labels.
    ///
    /// # Errors
    ///
    /// Returns an error if the input slices do not have matching lengths.
    fn gradients(&self, predictions: &[f64], labels: &[f64]) -> Result<Vec<GradPair>>;

    /// Compute the default initial prediction before any tree is trained.
    ///
    /// # Errors
    ///
    /// Returns an error if `labels` is empty or if its length cannot be
    /// represented exactly for the averaging step.
    fn default_base_score(&self, labels: &[f64]) -> Result<f64>;
}

/// Squared-error regression objective.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredError;

impl Objective for SquaredError {
    fn gradients(&self, predictions: &[f64], labels: &[f64]) -> Result<Vec<GradPair>> {
        if predictions.len() != labels.len() {
            return Err(XGBError::InvalidShape {
                context: "objective inputs",
                expected: labels.len(),
                actual: predictions.len(),
            });
        }

        Ok(predictions
            .iter()
            .zip(labels)
            .map(|(prediction, label)| GradPair::new(prediction - label, 1.0))
            .collect())
    }

    fn default_base_score(&self, labels: &[f64]) -> Result<f64> {
        if labels.is_empty() {
            return Err(XGBError::EmptyInput("objective labels"));
        }

        if labels.len() > MAX_EXACT_F64_INT {
            return Err(XGBError::InvalidShape {
                context: "objective labels",
                expected: MAX_EXACT_F64_INT,
                actual: labels.len(),
            });
        }

        let sum: f64 = labels.iter().sum();
        Ok(sum / exact_len_as_f64(labels.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::{Objective, SquaredError};

    #[test]
    fn squared_error_uses_label_mean_as_base_score() {
        let objective = SquaredError;
        let base_score = objective.default_base_score(&[1.0, 2.0, 3.0]).unwrap();
        assert!((base_score - 2.0).abs() < f64::EPSILON);
    }
}
