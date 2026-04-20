//! Rust inference runtime for supported official `XGBoost` `model.json` files.
//!
//! The crate is intentionally narrow in scope. It focuses on loading supported
//! official models and running CPU prediction with explicit unsupported-model
//! errors for everything outside that boundary.
//!
//! Current support:
//!
//! - official `save_model("model.json")` input
//! - `booster=gbtree`
//! - `objective=reg:squarederror` and `objective=binary:logistic`
//! - single-target regression and binary classification
//! - numerical splits only
//! - dense in-memory `f64` features
//!
//! `XGBModel::predict_dense(...)` returns task outputs:
//!
//! - regression predictions for `reg:squarederror`
//! - positive-class probabilities for `binary:logistic`
//!
//! # Example
//!
//! ```rust,no_run
//! use xgboost_rs::{DenseMatrix, RegressionTree, TreeNode, XGBModel};
//!
//! # fn main() -> Result<(), xgboost_rs::XGBError> {
//! let features = DenseMatrix::from_shape_vec(2, 1, vec![0.0, 2.0])?;
//! let tree = RegressionTree {
//!     nodes: vec![
//!         TreeNode {
//!             split_feature: Some(0),
//!             split_bin: None,
//!             split_value: Some(1.0),
//!             left_child: Some(1),
//!             right_child: Some(2),
//!             leaf_value: None,
//!             default_left: true,
//!         },
//!         TreeNode::leaf(-0.5),
//!         TreeNode::leaf(0.5),
//!     ],
//! };
//!
//! let model = XGBModel::new(0.5, 1, vec![tree])?;
//! let predictions = model.predict_dense(&features)?;
//! assert_eq!(predictions, vec![0.0, 1.0]);
//! # Ok(())
//! # }
//! ```

mod dataset;
mod error;
mod model;
mod official_model;
mod predict;
mod tree;

/// Dense prediction input.
pub use dataset::DenseMatrix;
/// Common result and error types used across the crate.
pub use error::{Result, XGBError};
/// Inference-only model surface.
pub use model::XGBModel;
/// Tree types reusable by adapters and tests.
pub use tree::{RegressionTree, TreeNode};
