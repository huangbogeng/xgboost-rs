# xgboost-rs

`xgboost-rs` is a focused Rust project for loading official `XGBoost` model files and running inference.

The repository is being refocused away from a full training implementation. The public crate API is now inference-first.

## Goal

The current supported scope is intentionally narrow:

- Load official XGBoost `save_model("model.json")` outputs
- Support `booster=gbtree`
- Support `objective=reg:squarederror`
- Support single-target regression
- Support numerical splits only
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

The public crate API now exposes an inference-only model type:

- `XGBModel::load_json(...)` loads supported official upstream `model.json` files
- `XGBModel::predict_dense(...)` runs dense CPU inference
- `XGBModel::new(...)` is available for tests and custom adapters that already have tree data
- the old training API is no longer exported from the crate root

## Quick Start

```rust,no_run
use xgboost_rs::{DenseMatrix, XGBModel};

fn main() -> Result<(), xgboost_rs::XGBError> {
    let model = XGBModel::load_json("model.json")?;
    let features = DenseMatrix::from_shape_vec(2, 3, vec![0.1, 0.2, 0.3, 1.0, 2.0, 3.0])?;
    let predictions = model.predict_dense(&features)?;
    println!("{predictions:?}");
    Ok(())
}
```

## Roadmap

1. Extend official model support beyond `reg:squarederror`
2. Add `.ubj` loading support
3. Add more fixture coverage for official upstream models

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
