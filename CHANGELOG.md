# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-04-19

### Added

- Initial crate release.
- Dense in-memory matrix types for training and prediction.
- Histogram-based tree training for squared-error regression.
- Gradient-boosted regression model training and dense prediction.
- Internal JSON model save/load helpers for round-tripping fitted models.
- Regression tests covering training, prediction, and serialization.

### Notes

- This release is an experimental foundation crate, not a feature-complete port of upstream `XGBoost`.
- The current JSON model I/O is internal to this crate and does not yet load official upstream `save_model("model.json")` outputs.
