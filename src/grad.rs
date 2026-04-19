//! Gradient pair types used during tree construction.

use serde::{Deserialize, Serialize};

/// First- and second-order statistics for one training row.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct GradPair {
    pub grad: f64,
    pub hess: f64,
}

impl GradPair {
    /// Create a new gradient pair.
    #[must_use]
    pub fn new(grad: f64, hess: f64) -> Self {
        Self { grad, hess }
    }
}
