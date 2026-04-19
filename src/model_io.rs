//! JSON model serialization helpers.

use std::fs;
use std::path::Path;

use crate::booster::gbtree::XGBRegressor;
use crate::error::Result;

/// Save a fitted model to a JSON file.
///
/// Training-only caches are intentionally not persisted.
///
/// # Errors
///
/// Returns an error if serializing the model fails or if writing the file to
/// disk fails.
pub fn save_json<P: AsRef<Path>>(model: &XGBRegressor, path: P) -> Result<()> {
    let contents = serde_json::to_string_pretty(model)?;
    fs::write(path, contents)?;
    Ok(())
}

/// Load a model from a JSON file previously produced by [`save_json`].
///
/// # Errors
///
/// Returns an error if reading the file fails or if deserialization into the
/// internal model type fails.
pub fn load_json<P: AsRef<Path>>(path: P) -> Result<XGBRegressor> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}
