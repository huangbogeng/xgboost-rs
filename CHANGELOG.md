# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.2.0] - 2026-04-21

### Added

- Official `XGBoost` `save_model("model.json")` loading for supported `gbtree` models.
- Inference support for `reg:squarederror`, `binary:logistic`, `multi:softprob`, and `multi:softmax`.
- Objective-aware outputs in `predict_dense`, including multiclass probability and label inference.
- Defensive model-structure validation for malformed trees (for example cycles, unreachable nodes, and invalid child/default encoding).
- Fixture-based integration coverage for regression, binary classification, and multiclass classification.
- Robustness tests for malformed JSON edge cases, `base_score` boundary handling, missing-value routing, and deep unbalanced trees.

### Changed

- Public naming was aligned to Rust conventions (`XgbModel`, `XgbError`, and `BoosterTree`).
- Binary and multiclass `base_score` semantics now follow XGBoost margin/probability conventions during inference.

### Removed

- Obsolete internal training-oriented JSON model loader modules that were replaced by official-model parsing.

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
