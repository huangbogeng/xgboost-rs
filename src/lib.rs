//! A small Rust implementation of the `XGBoost` core for regression.
//!
//! The current crate focuses on one constrained use case:
//!
//! - CPU-only training
//! - dense in-memory `f64` input
//! - histogram-based tree construction
//! - squared-error regression
//! - prediction and JSON model I/O
//!
//! It is intentionally not a feature-complete port of upstream `XGBoost`.
//!
//! # Example
//!
//! ```rust,no_run
//! use xgboost_rs::{DMatrix, DenseMatrix, XGBRegressor};
//!
//! # fn main() -> Result<(), xgboost_rs::XGBError> {
//! let features = DenseMatrix::from_shape_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0])?;
//! let train = DMatrix::from_dense(features.clone(), vec![0.0, 0.0, 1.0, 1.0])?;
//!
//! let model = XGBRegressor::builder()
//!     .n_estimators(1)
//!     .max_depth(1)
//!     .learning_rate(1.0)
//!     .max_bin(4)
//!     .lambda(0.0)
//!     .gamma(0.0)
//!     .min_child_weight(0.0)
//!     .build()?;
//!
//! let fitted = model.fit(&train)?;
//! let predictions = fitted.predict_dense(&features)?;
//! assert_eq!(predictions, vec![0.0, 0.0, 1.0, 1.0]);
//! # Ok(())
//! # }
//! ```

pub mod booster;
pub mod dataset;
pub mod error;
pub mod grad;
pub mod hist;
pub mod metrics;
pub mod model_io;
pub mod objective;
pub mod params;
pub mod predict;
pub mod tree;

/// Regression model and training entry point.
pub use booster::gbtree::XGBRegressor;
/// Dense training and prediction datasets.
pub use dataset::{DMatrix, DenseMatrix};
/// Common result and error types used across the crate.
pub use error::{Result, XGBError};
/// Parameter types and builder for [`XGBRegressor`].
pub use params::{XGBRegressorBuilder, XGBRegressorParams};
