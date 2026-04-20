//! Inference-focused Rust runtime for tree models inspired by `XGBoost`.
//!
//! The public API is being migrated away from training and toward model loading
//! and prediction. During this transition, the crate exposes an inference-only
//! [`XGBModel`] together with reusable dense feature and tree types.
//!
//! The current public surface focuses on:
//!
//! - dense in-memory `f64` input
//! - tree-ensemble prediction
//! - official upstream `model.json` loading for a narrow regression subset
//! - explicit tree construction for tests and adapters
//!
//! Current official model support is intentionally narrow:
//!
//! - `booster=gbtree`
//! - `objective=reg:squarederror`
//! - single-target regression
//! - numerical splits only
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
