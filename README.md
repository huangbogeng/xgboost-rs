# xgboost-rs

`xgboost-rs` is a Rust inference runtime for loading official `XGBoost` `model.json` files and running CPU prediction.

The project is intentionally narrow in scope. It is not a reimplementation of the full `XGBoost` training stack. The goal is to provide a small, predictable, well-tested path for supported official models.

## Project Scope

This crate currently supports:

| Area | Support |
| --- | --- |
| Model input | Official `save_model("model.json")` output |
| Booster | `gbtree` |
| Objectives | `reg:squarederror`, `binary:logistic` |
| Outputs | Single-target regression, binary classification |
| Splits | Numerical splits only |
| Inference | CPU, dense in-memory `f64` features |
| Missing values | Respects each node's `default_left` routing |

This crate does not currently support:

| Area | Not supported |
| --- | --- |
| Input formats | Anything other than official `model.json` |
| Objectives | Multiclass objectives |
| Boosters | `dart`, `gblinear` |
| Tree features | Categorical splits, multi-output trees |
| Training | Any training API or training parity with upstream |

## Prediction Semantics

`XGBModel::predict_dense(...)` returns the task output for the loaded model:

| Objective | Returned value |
| --- | --- |
| `reg:squarederror` | Regression prediction |
| `binary:logistic` | Positive-class probability |

For official binary classification models, serialized `base_score` is interpreted with XGBoost's logistic semantics before inference.

## Example

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

## API Surface

- `XGBModel::load_json(...)`
  Loads a supported official `XGBoost` `model.json` file.
- `XGBModel::predict_dense(...)`
  Runs CPU inference for dense features and returns task outputs.
- `XGBModel::new(...)`
  Builds a regression model from already prepared tree structures. This is mainly useful for tests and adapters.

## Design Notes

- The crate favors explicit unsupported-model errors over partial compatibility.
- Tree traversal follows official `XGBoost` numeric split behavior, including `f32` comparison semantics.
- Model loading performs structural validation before prediction.

## Roadmap

1. Add multiclass `gbtree` support for official `model.json` models.
2. Expand fixture coverage for supported and unsupported official models.
3. Continue hardening loader validation and inference correctness.

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Please report vulnerabilities according to [SECURITY.md](SECURITY.md).

## License

Licensed under [MIT](LICENSE).
