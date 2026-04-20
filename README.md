<div align="center">
  <h1>xgboost-rs</h1>
  <p><strong>A focused Rust inference runtime for supported official XGBoost <code>model.json</code> files.</strong></p>
  
  [![Crates.io](https://img.shields.io/crates/v/xgboost-rs.svg)](https://crates.io/crates/xgboost-rs)
  [![Documentation](https://docs.rs/xgboost-rs/badge.svg)](https://docs.rs/xgboost-rs)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

It is built for one job: load a supported upstream model, run CPU prediction, and fail explicitly when the model is outside the supported boundary.

## 🎯 Why This Project

`XGBoost` model compatibility is valuable, but full upstream parity is a large surface area. This crate takes a narrower approach:

- **Support official exported models**, not ad-hoc formats.
- **Prioritize correctness** over partial compatibility.
- **Keep the runtime small**, predictable, and easy to validate.
- **Provide clear unsupported-model errors** instead of silent divergence.

That makes `xgboost-rs` a good fit when you want a Rust-native inference path with explicit scope and tight control over behavior.

## 📦 Supported Scope

| Area | Support |
| :--- | :--- |
| **Model input** | Official `save_model("model.json")` output |
| **Booster** | `gbtree` |
| **Objectives** | `reg:squarederror`, `binary:logistic`, `multi:softprob`, `multi:softmax` |
| **Outputs** | Single-target regression, binary classification, multiclass classification |
| **Splits** | Numerical splits only |
| **Inference** | CPU, dense in-memory `f64` features |
| **Missing values** | Honors each node's `default_left` routing |

## 📊 Prediction Output

`XgbModel::predict_dense(...)` returns task outputs for the loaded model:

| Objective | Returned value |
| :--- | :--- |
| `reg:squarederror` | Regression prediction |
| `binary:logistic` | Positive-class probability |
| `multi:softprob` | Row-major class probabilities (shape: `n_rows * num_class`) |
| `multi:softmax` | Predicted class labels encoded as `f64` |

For supported binary models, serialized `base_score` is interpreted using XGBoost's logistic semantics before inference. Multiclass fixtures with vector `base_score` are interpreted as per-class base margins.

## 🚫 Out of Scope

The crate does **not** currently support:

- Anything other than official `model.json`
- `dart` or `gblinear`
- Categorical splits
- Multi-output trees
- Training APIs or training parity with upstream `XGBoost`

## 🚀 Example

```rust,no_run
use xgboost_rs::{DenseMatrix, XgbModel};

fn main() -> Result<(), xgboost_rs::XgbError> {
    // 1. Load the model
    let model = XgbModel::load_json("model.json")?;

    // 2. Prepare features
    let features = DenseMatrix::from_shape_vec(
        2,
        3,
        vec![0.1, 0.2, 0.3,
             1.0, 2.0, 3.0],
    )?;

    // 3. Predict
    let predictions = model.predict_dense(&features)?;
    println!("{predictions:?}");

    Ok(())
}
```

## 🛠️ API Overview

- **`XgbModel::load_json(...)`**: Loads a supported official `XGBoost` `model.json` file.
- **`XgbModel::predict_dense(...)`**: Runs dense CPU inference and returns task outputs.
- **`XgbModel::new(...)`**: Builds a regression model from already prepared tree structures. This is mainly useful for tests and adapters.

## 💡 Design Principles

- **Official-model first**: Support exported upstream models, not custom interpretations.
- **Explicit boundaries**: Reject unsupported boosters, objectives, and split types.
- **Runtime correctness**: Match XGBoost numeric tree traversal semantics, including `f32` split comparison behavior.
- **Defensive loading**: Validate model structure before prediction.

## 🛤️ Current Direction

The project is focused on strengthening the supported path through:
- Broader fixture coverage
- Better loader validation
- Continued alignment with official model semantics

## ⚙️ Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## 🤝 Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## 🔒 Security

Please report vulnerabilities according to [SECURITY.md](SECURITY.md).

## 📄 License

Licensed under [MIT](LICENSE).
