# xgboost-rs

`xgboost-rs` is a focused Rust project for loading official XGBoost model files and running inference.

The repository is being refocused away from a full training implementation. The target is a small, dependable runtime for models produced by upstream XGBoost.

## Goal

The first release target is intentionally narrow:

- Load official XGBoost `save_model("model.json")` outputs
- Support `booster=gbtree`
- Support `objective=reg:squarederror`
- Run CPU inference on dense in-memory `f64` features
- Respect XGBoost missing-value routing through each node's default branch

## Non-Goals For The First Release

- Reimplement full XGBoost training
- Support `.ubj` model files
- Support binary or multiclass classification
- Support categorical splits
- Support `dart` or `gblinear`
- Match the full upstream XGBoost feature surface

## Current Status

The codebase already contains reusable tree and prediction components, but official XGBoost model loading is not finished yet.

In particular:

- `src/predict.rs` and `src/tree/node.rs` are good foundations for inference
- `DenseMatrix` in `src/dataset.rs` is reusable for prediction input
- the current JSON I/O only round-trips this crate's internal Rust structs
- loading official XGBoost `model.json` files is the next major implementation step

## Planned Runtime Shape

The intended flow is:

1. Parse official XGBoost `model.json`
2. Convert the array-oriented tree data into compact Rust tree structs
3. Reuse the internal tree traversal code for prediction

The expected public API will look roughly like this:

```rust,ignore
use xgboost_rs::{DenseMatrix, XGBModel};

fn main() -> Result<(), xgboost_rs::XGBError> {
    let model = XGBModel::load_json("model.json")?;
    let features = DenseMatrix::from_shape_vec(2, 3, vec![0.1, 0.2, 0.3, 1.0, 2.0, 3.0])?;
    let predictions = model.predict_dense(&features)?;
    println!("{predictions:?}");
    Ok(())
}
```

This example describes the target API for the refocused project, not the current implementation.

## Roadmap

1. Add an inference-only model type for loaded XGBoost models
2. Implement official XGBoost `model.json` parsing for `gbtree` regression
3. Add fixture-based tests with real exported upstream models
4. Extend support only when the inference core is stable

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## Contributing

Issues and PRs are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Please report vulnerabilities according to [SECURITY.md](SECURITY.md).

## License

Licensed under [MIT](LICENSE).
