//! Simple regression metrics used in tests and diagnostics.

const MAX_EXACT_F64_INT: usize = 1 << 53;

#[allow(
    clippy::cast_precision_loss,
    reason = "values up to 2^53 convert exactly to f64"
)]
fn exact_len_as_f64(len: usize) -> f64 {
    len as f64
}

/// Compute root mean squared error.
///
/// Returns `None` if the inputs are empty or have different lengths.
#[must_use]
pub fn rmse(predictions: &[f64], labels: &[f64]) -> Option<f64> {
    if predictions.is_empty() || predictions.len() != labels.len() {
        return None;
    }

    if predictions.len() > MAX_EXACT_F64_INT {
        return None;
    }

    let mse = predictions
        .iter()
        .zip(labels)
        .map(|(prediction, label)| {
            let diff = prediction - label;
            diff * diff
        })
        .sum::<f64>()
        / exact_len_as_f64(predictions.len());

    Some(mse.sqrt())
}

#[cfg(test)]
mod tests {
    use super::rmse;

    #[test]
    fn rmse_is_zero_for_identical_vectors() {
        let value = rmse(&[1.0, 2.0], &[1.0, 2.0]).unwrap();
        assert!(value.abs() < f64::EPSILON);
    }
}
