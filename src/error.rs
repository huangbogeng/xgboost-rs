use std::fmt::{Display, Formatter};
use std::io;

/// Crate-local result type.
pub type Result<T> = std::result::Result<T, XgbError>;

/// Error type for model loading and prediction.
#[derive(Debug)]
pub enum XgbError {
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
    /// The feature count used for prediction does not match the fitted model.
    FeatureCountMismatch { expected: usize, actual: usize },
    /// The model file is structurally invalid for the expected schema.
    InvalidModelFormat(&'static str),
    /// The model uses a valid upstream feature that this crate does not support yet.
    UnsupportedModel {
        context: &'static str,
        value: String,
    },
    /// An underlying I/O operation failed.
    Io(io::Error),
    /// Model serialization or deserialization failed.
    Serde(serde_json::Error),
}

impl Display for XgbError {
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
            Self::FeatureCountMismatch { expected, actual } => write!(
                f,
                "feature count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidModelFormat(context) => write!(f, "invalid model format: {context}"),
            Self::UnsupportedModel { context, value } => {
                write!(f, "unsupported model {context}: {value}")
            }
            Self::Io(err) => write!(f, "i/o error: {err}"),
            Self::Serde(err) => write!(f, "serialization error: {err}"),
        }
    }
}

impl std::error::Error for XgbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Serde(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for XgbError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for XgbError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value)
    }
}
