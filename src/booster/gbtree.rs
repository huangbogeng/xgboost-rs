//! Gradient-boosted tree model and training loop.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dataset::{DMatrix, DenseMatrix};
use crate::error::{Result, XGBError};
use crate::hist::bin_matrix::BinMatrix;
use crate::hist::cuts::FeatureCuts;
use crate::model_io;
use crate::objective::{Objective, SquaredError};
use crate::params::{XGBRegressorBuilder, XGBRegressorParams};
use crate::predict;
use crate::tree::RegressionTree;
use crate::tree::builder::TreeBuilder;

#[derive(Debug, Clone, PartialEq)]
struct QuantizedTrainingCache {
    feature_cuts: FeatureCuts,
    bin_matrix: BinMatrix,
}

/// Gradient-boosted regression tree model.
///
/// The same type currently acts as both the configuration-backed training entry
/// point and the fitted model. Before `fit`, `predict_dense` returns
/// [`XGBError::ModelNotFitted`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XGBRegressor {
    params: XGBRegressorParams,
    trees: Vec<RegressionTree>,
    base_score: f64,
    n_features: usize,
    #[serde(skip)]
    training_cache: Option<QuantizedTrainingCache>,
}

impl XGBRegressor {
    /// Create a new unfitted regression model from validated parameters.
    #[must_use]
    pub fn new(params: XGBRegressorParams) -> Self {
        Self {
            base_score: params.base_score.unwrap_or(0.0),
            params,
            trees: Vec::new(),
            n_features: 0,
            training_cache: None,
        }
    }

    /// Create a builder with default parameters.
    pub fn builder() -> XGBRegressorBuilder {
        XGBRegressorBuilder::new()
    }

    /// Borrow the parameter set used by this model.
    #[must_use]
    pub fn params(&self) -> &XGBRegressorParams {
        &self.params
    }

    /// Return the initial prediction used before any tree contribution.
    #[must_use]
    pub fn base_score(&self) -> f64 {
        self.base_score
    }

    /// Return the number of features expected by this model.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Return the number of trees currently stored in the ensemble.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Return whether the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.n_features != 0
    }

    /// Train a regression model on a dense dataset.
    ///
    /// The training loop computes a quantized representation once, then builds
    /// one tree per boosting round using histogram-based split search.
    ///
    /// # Errors
    ///
    /// Returns an error if dataset shapes are invalid or if quantization or tree
    /// construction encounters an unsupported condition.
    pub fn fit(&self, train: &DMatrix) -> Result<Self> {
        let objective = SquaredError;
        let mut fitted = self.clone();
        fitted.n_features = train.n_cols();
        fitted.base_score = self
            .params
            .base_score
            .unwrap_or(objective.default_base_score(train.labels())?);
        let feature_cuts = FeatureCuts::build(train.features(), self.params.max_bin)?;
        let bin_matrix = BinMatrix::from_dense(train.features(), &feature_cuts)?;
        let tree_builder = TreeBuilder::new(self.params.clone());
        let mut predictions = vec![fitted.base_score; train.n_rows()];
        fitted.trees.clear();

        for _ in 0..self.params.n_estimators {
            let gradients = objective.gradients(&predictions, train.labels())?;
            let tree = tree_builder.build(train, &feature_cuts, &bin_matrix, &gradients)?;

            // Keep cached predictions in sync with the current ensemble so the
            // next round can compute fresh gradients without re-scoring all trees.
            for (row_idx, prediction) in predictions.iter_mut().enumerate() {
                *prediction += predict::predict_tree(&tree, train.features(), row_idx);
            }

            fitted.trees.push(tree);
        }

        fitted.training_cache = Some(QuantizedTrainingCache {
            feature_cuts,
            bin_matrix,
        });

        Ok(fitted)
    }

    /// Predict regression values for a dense feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`XGBError::ModelNotFitted`] if called before `fit`, or
    /// [`XGBError::FeatureCountMismatch`] if the feature count differs from the
    /// fitted model.
    pub fn predict_dense(&self, features: &DenseMatrix) -> Result<Vec<f64>> {
        if !self.is_fitted() {
            return Err(XGBError::ModelNotFitted);
        }

        predict::predict_ensemble(self.base_score, &self.trees, features, self.n_features)
    }

    /// Save the model as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serializing the model fails or if writing the file
    /// to disk fails.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        model_io::save_json(self, path)
    }

    /// Load a model from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if reading the file fails or if the JSON document does
    /// not match this crate's internal model format.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        model_io::load_json(path)
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::{DMatrix, DenseMatrix};

    use super::XGBRegressor;

    #[test]
    fn fit_builds_quantized_training_cache() {
        let features =
            DenseMatrix::from_shape_vec(3, 2, vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0]).unwrap();
        let train = DMatrix::from_dense(features, vec![1.0, 2.0, 3.0]).unwrap();
        let model = XGBRegressor::builder().max_bin(3).build().unwrap();

        let fitted = model.fit(&train).unwrap();
        let cache = fitted.training_cache.as_ref().unwrap();

        assert_eq!(cache.feature_cuts.n_features(), 2);
        assert_eq!(cache.bin_matrix.n_rows, 3);
        assert_eq!(cache.bin_matrix.n_cols, 2);
    }
}
