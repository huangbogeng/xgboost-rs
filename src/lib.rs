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
//! - `objective=reg:squarederror`, `objective=binary:logistic`,
//!   `objective=multi:softprob`, and `objective=multi:softmax`
//! - single-target regression, binary classification, and multiclass classification
//! - numerical splits only
//! - dense in-memory `f64` features
//!
//! `XgbModel::predict_dense(...)` returns task outputs:
//!
//! - regression predictions for `reg:squarederror`
//! - positive-class probabilities for `binary:logistic`
//! - row-major class probabilities for `multi:softprob`
//! - class labels encoded as `f64` for `multi:softmax`
//!
//! # Example
//!
//! ```rust,no_run
//! use xgboost_rs::{BoosterTree, DenseMatrix, TreeNode, XgbModel};
//!
//! # fn main() -> Result<(), xgboost_rs::XgbError> {
//! let features = DenseMatrix::from_shape_vec(2, 1, vec![0.0, 2.0])?;
//! let tree = BoosterTree {
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
//! let model = XgbModel::new(0.5, 1, vec![tree])?;
//! let predictions = model.predict_dense(&features)?;
//! assert_eq!(predictions, vec![0.0, 1.0]);
//! # Ok(())
//! # }
//! ```

#[cfg(not(any(
    feature = "infer-serial",
    feature = "infer-row-parallel",
    feature = "infer-tree-parallel"
)))]
compile_error!(
    "xgboost-rs requires exactly one inference backend feature: infer-serial, infer-row-parallel, or infer-tree-parallel"
);

#[cfg(any(
    all(feature = "infer-serial", feature = "infer-row-parallel"),
    all(feature = "infer-serial", feature = "infer-tree-parallel"),
    all(feature = "infer-row-parallel", feature = "infer-tree-parallel")
))]
compile_error!(
    "xgboost-rs inference backend features are mutually exclusive; enable exactly one of infer-serial, infer-row-parallel, or infer-tree-parallel"
);

mod dataset;
mod error;
mod inference;
mod model;
mod tree;
mod xgboost_json;

/// Dense prediction input.
pub use dataset::DenseMatrix;
/// Common result and error types used across the crate.
pub use error::{Result, XgbError};
/// Inference-only model surface.
pub use model::XgbModel;
/// Tree types reusable by adapters and tests.
pub use tree::{BoosterTree, TreeNode};
