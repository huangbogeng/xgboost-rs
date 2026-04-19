//! Split scoring and leaf-value formulas.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Sub};

/// Aggregated gradient statistics for a node or histogram bucket.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct NodeStats {
    pub grad_sum: f64,
    pub hess_sum: f64,
}

impl NodeStats {
    /// Create a new gradient-statistics accumulator.
    #[must_use]
    pub fn new(grad_sum: f64, hess_sum: f64) -> Self {
        Self { grad_sum, hess_sum }
    }
}

impl Add for NodeStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.grad_sum + rhs.grad_sum, self.hess_sum + rhs.hess_sum)
    }
}

impl AddAssign for NodeStats {
    fn add_assign(&mut self, rhs: Self) {
        self.grad_sum += rhs.grad_sum;
        self.hess_sum += rhs.hess_sum;
    }
}

impl Sub for NodeStats {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.grad_sum - rhs.grad_sum, self.hess_sum - rhs.hess_sum)
    }
}

/// Compute the optimal leaf value under L2 regularization.
#[must_use]
pub fn leaf_weight(stats: NodeStats, lambda: f64) -> f64 {
    -stats.grad_sum / (stats.hess_sum + lambda)
}

/// Compute the gain produced by splitting a parent node into two children.
#[must_use]
pub fn split_gain(
    parent: NodeStats,
    left: NodeStats,
    right: NodeStats,
    lambda: f64,
    gamma: f64,
) -> f64 {
    let score = |stats: NodeStats| {
        let denominator = stats.hess_sum + lambda;
        if denominator <= 0.0 {
            return 0.0;
        }

        (stats.grad_sum * stats.grad_sum) / denominator
    };
    0.5 * (score(left) + score(right) - score(parent)) - gamma
}

/// Best split found while scanning one node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SplitCandidate {
    pub feature_idx: usize,
    pub split_bin: usize,
    pub split_value: f64,
    pub gain: f64,
    pub left_stats: NodeStats,
    pub right_stats: NodeStats,
    pub default_left: bool,
}

#[cfg(test)]
mod tests {
    use super::{NodeStats, leaf_weight, split_gain};

    #[test]
    fn leaf_weight_matches_closed_form() {
        let stats = NodeStats {
            grad_sum: 4.0,
            hess_sum: 2.0,
        };
        assert!((leaf_weight(stats, 1.0) - (-4.0 / 3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn split_gain_is_positive_for_better_children() {
        let parent = NodeStats {
            grad_sum: 4.0,
            hess_sum: 4.0,
        };
        let left = NodeStats {
            grad_sum: 3.0,
            hess_sum: 2.0,
        };
        let right = NodeStats {
            grad_sum: 1.0,
            hess_sum: 2.0,
        };

        assert!(split_gain(parent, left, right, 1.0, 0.0) > 0.0);
    }
}
