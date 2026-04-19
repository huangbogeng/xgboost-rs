use std::fmt::{Display, Formatter};
use std::io;

/// Crate-local result type.
pub type Result<T> = std::result::Result<T, XGBError>;

/// Error type for training, prediction, and model I/O.
#[derive(Debug)]
pub enum XGBError {
    /// A shape-related invariant was violated.
    InvalidShape {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A parameter value was invalid for the current implementation.
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },
    /// A required non-empty input was empty.
    EmptyInput(&'static str),
    /// The feature count used for prediction does not match the fitted model.
    FeatureCountMismatch { expected: usize, actual: usize },
    /// Prediction was requested from a model that has not been fitted yet.
    ModelNotFitted,
    /// An underlying I/O operation failed.
    Io(io::Error),
    /// Model serialization or deserialization failed.
    Serde(serde_json::Error),
    /// Placeholder error for functionality that is intentionally not implemented yet.
    Unimplemented(&'static str),
}

impl Display for XGBError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape {
                context,
                expected,
                actual,
            } => write!(
                f,
                "invalid shape for {context}: expected {expected}, got {actual}"
            ),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::EmptyInput(context) => write!(f, "empty input for {context}"),
            Self::FeatureCountMismatch { expected, actual } => write!(
                f,
                "feature count mismatch: expected {expected}, got {actual}"
            ),
            Self::ModelNotFitted => write!(f, "model has not been fitted"),
            Self::Io(err) => write!(f, "i/o error: {err}"),
            Self::Serde(err) => write!(f, "serialization error: {err}"),
            Self::Unimplemented(message) => write!(f, "not implemented: {message}"),
        }
    }
}

impl std::error::Error for XGBError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Serde(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for XGBError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for XGBError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value)
    }
}
