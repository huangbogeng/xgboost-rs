//! Tree node types used by regression tree ensembles.

/// One node in a regression tree.
///
/// A node is either a leaf with `leaf_value`, or a split node with child
/// indices and split metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    pub split_feature: Option<usize>,
    pub split_bin: Option<usize>,
    pub split_value: Option<f64>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub leaf_value: Option<f64>,
    pub default_left: bool,
}

impl TreeNode {
    /// Create a leaf node with a fixed prediction value.
    #[must_use]
    pub fn leaf(value: f64) -> Self {
        Self {
            split_feature: None,
            split_bin: None,
            split_value: None,
            left_child: None,
            right_child: None,
            leaf_value: Some(value),
            default_left: true,
        }
    }
}

/// One trained regression tree.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionTree {
    pub nodes: Vec<TreeNode>,
}

impl RegressionTree {
    /// Create a tree with a single root node.
    #[must_use]
    pub fn new(root: TreeNode) -> Self {
        Self { nodes: vec![root] }
    }
}
