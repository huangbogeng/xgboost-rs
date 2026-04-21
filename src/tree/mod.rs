//! Tree structures used by inference.

mod node;
mod validation;

pub use node::{BoosterTree, TreeNode};
pub(crate) use validation::validate_tree_topology;
