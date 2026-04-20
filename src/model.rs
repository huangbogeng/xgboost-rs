//! Inference-only model type used by the public crate API.

use std::path::Path;

use crate::dataset::DenseMatrix;
use crate::error::{Result, XGBError};
use crate::official_model;
use crate::predict;
use crate::tree::RegressionTree;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredictionTask {
    Regression,
    BinaryLogistic,
}

/// Inference-only gradient-boosted tree model.
///
/// This type is the stable public model surface. Callers can either load a
/// supported official upstream `XGBoost` model file or construct a model from
/// already prepared tree structures.
#[derive(Debug, Clone, PartialEq)]
pub struct XGBModel {
    trees: Vec<RegressionTree>,
    base_margin: f64,
    n_features: usize,
    task: PredictionTask,
}

impl XGBModel {
    /// Create an inference model from already prepared trees.
    ///
    /// # Errors
    ///
    /// Returns [`XGBError::InvalidParameter`] if `base_score` is not finite or
    /// if `n_features == 0`.
    pub fn new(base_score: f64, n_features: usize, trees: Vec<RegressionTree>) -> Result<Self> {
        Self::from_parts(PredictionTask::Regression, base_score, n_features, trees)
    }

    pub(crate) fn from_parts(
        task: PredictionTask,
        base_margin: f64,
        n_features: usize,
        trees: Vec<RegressionTree>,
    ) -> Result<Self> {
        if !base_margin.is_finite() {
            return Err(XGBError::InvalidParameter {
                name: "base_score",
                reason: "must be finite",
            });
        }

        if n_features == 0 {
            return Err(XGBError::InvalidParameter {
                name: "n_features",
                reason: "must be greater than zero",
            });
        }

        Ok(Self {
            trees,
            base_margin,
            n_features,
            task,
        })
    }

    /// Return the serialized `XGBoost` `base_score` for supported objectives.
    #[must_use]
    pub fn base_score(&self) -> f64 {
        match self.task {
            PredictionTask::Regression => self.base_margin,
            PredictionTask::BinaryLogistic => predict::sigmoid(self.base_margin),
        }
    }

    /// Return the number of feature columns expected by this model.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Borrow the ensemble trees.
    #[must_use]
    pub fn trees(&self) -> &[RegressionTree] {
        &self.trees
    }

    /// Predict task outputs for a dense feature matrix.
    ///
    /// For supported official JSON models this returns:
    ///
    /// - regression values for `reg:squarederror`
    /// - positive-class probabilities for `binary:logistic`
    ///
    /// # Errors
    ///
    /// Returns [`XGBError::FeatureCountMismatch`] if the feature count differs
    /// from the model expectation.
    pub fn predict_dense(&self, features: &DenseMatrix) -> Result<Vec<f64>> {
        predict::predict_ensemble(
            self.task,
            self.base_margin,
            &self.trees,
            features,
            self.n_features,
        )
    }

    /// Load an official upstream `XGBoost` `model.json` file.
    ///
    /// Currently supported model scope:
    ///
    /// - `booster=gbtree`
    /// - `objective=reg:squarederror` or `objective=binary:logistic`
    /// - single-target prediction
    /// - numerical splits only
    ///
    /// # Errors
    ///
    /// Returns an error if reading the file fails, if the JSON is malformed, or
    /// if the model uses unsupported upstream features.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        official_model::load_json_model(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected) {
            assert!((lhs - rhs).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn predicts_with_base_score_only() {
        let features = DenseMatrix::from_shape_vec(2, 1, vec![0.0, 1.0]).unwrap();
        let model = XGBModel::new(0.75, 1, Vec::new()).unwrap();

        let predictions = model.predict_dense(&features).unwrap();

        assert_vec_close(&predictions, &[0.75, 0.75]);
    }
}
