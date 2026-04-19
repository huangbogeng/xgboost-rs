//! Parameter types for regression training.

use serde::{Deserialize, Serialize};

use crate::error::{Result, XGBError};

/// Training parameters for [`crate::XGBRegressor`].
///
/// This crate currently implements a compact subset of the upstream `XGBoost`
/// parameter surface. The semantics are aligned with histogram-based
/// regression training in this crate, not with full upstream compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XGBRegressorParams {
    /// Number of boosting rounds.
    pub n_estimators: usize,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// Shrinkage applied to each tree's leaf values.
    pub learning_rate: f64,
    /// Maximum number of bins per feature during quantization.
    pub max_bin: usize,
    /// L2 regularization term applied to leaf weights.
    pub lambda: f64,
    /// Minimum split gain required to expand a node.
    pub gamma: f64,
    /// Minimum Hessian mass required in each child node.
    pub min_child_weight: f64,
    /// Optional initial prediction value.
    ///
    /// When omitted, the training mean of the labels is used.
    pub base_score: Option<f64>,
}

impl Default for XGBRegressorParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: 6,
            learning_rate: 0.3,
            max_bin: 256,
            lambda: 1.0,
            gamma: 0.0,
            min_child_weight: 1.0,
            base_score: None,
        }
    }
}

impl XGBRegressorParams {
    /// Validate the parameter set against the current implementation constraints.
    ///
    /// # Errors
    ///
    /// Returns [`XGBError::InvalidParameter`] if any value is outside the
    /// supported range.
    pub fn validate(&self) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(XGBError::InvalidParameter {
                name: "n_estimators",
                reason: "must be greater than zero",
            });
        }

        if self.learning_rate <= 0.0 {
            return Err(XGBError::InvalidParameter {
                name: "learning_rate",
                reason: "must be greater than zero",
            });
        }

        if self.max_bin < 2 {
            return Err(XGBError::InvalidParameter {
                name: "max_bin",
                reason: "must be at least 2",
            });
        }

        if self.lambda < 0.0 {
            return Err(XGBError::InvalidParameter {
                name: "lambda",
                reason: "must be non-negative",
            });
        }

        if self.gamma < 0.0 {
            return Err(XGBError::InvalidParameter {
                name: "gamma",
                reason: "must be non-negative",
            });
        }

        if self.min_child_weight < 0.0 {
            return Err(XGBError::InvalidParameter {
                name: "min_child_weight",
                reason: "must be non-negative",
            });
        }

        Ok(())
    }
}

/// Builder for [`XGBRegressorParams`] and [`crate::XGBRegressor`].
#[derive(Debug, Clone, Default)]
#[must_use = "builder methods return a modified builder that must be used"]
pub struct XGBRegressorBuilder {
    params: XGBRegressorParams,
}

impl XGBRegressorBuilder {
    /// Create a builder with default regression parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of boosting rounds.
    pub fn n_estimators(mut self, value: usize) -> Self {
        self.params.n_estimators = value;
        self
    }

    /// Set the maximum depth of each tree.
    pub fn max_depth(mut self, value: usize) -> Self {
        self.params.max_depth = value;
        self
    }

    /// Set the per-tree learning rate.
    pub fn learning_rate(mut self, value: f64) -> Self {
        self.params.learning_rate = value;
        self
    }

    /// Set the maximum number of bins per feature.
    pub fn max_bin(mut self, value: usize) -> Self {
        self.params.max_bin = value;
        self
    }

    /// Set the L2 regularization term for leaf weights.
    pub fn lambda(mut self, value: f64) -> Self {
        self.params.lambda = value;
        self
    }

    /// Set the minimum gain required for a split.
    pub fn gamma(mut self, value: f64) -> Self {
        self.params.gamma = value;
        self
    }

    /// Set the minimum Hessian mass required in each child.
    pub fn min_child_weight(mut self, value: f64) -> Self {
        self.params.min_child_weight = value;
        self
    }

    /// Set the initial prediction value used before any tree is trained.
    pub fn base_score(mut self, value: f64) -> Self {
        self.params.base_score = Some(value);
        self
    }

    /// Build a regression model with the configured parameters.
    ///
    /// The returned value is still unfitted. Call `fit` to train trees.
    ///
    /// # Errors
    ///
    /// Returns [`XGBError::InvalidParameter`] if the final parameter set is invalid.
    pub fn build(self) -> Result<crate::booster::gbtree::XGBRegressor> {
        self.params.validate()?;
        Ok(crate::booster::gbtree::XGBRegressor::new(self.params))
    }
}

#[cfg(test)]
mod tests {
    use super::XGBRegressorParams;

    #[test]
    fn default_params_are_valid() {
        let params = XGBRegressorParams::default();
        assert!(params.validate().is_ok());
    }
}
