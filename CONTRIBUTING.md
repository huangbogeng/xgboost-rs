# Contributing

Thanks for your interest in contributing to `xgboost-rs`.

## Ways to Contribute

- Report bugs
- Propose features
- Improve docs
- Submit code via pull requests

## Development Setup

1. Fork and clone this repository.
2. Install Rust toolchain (`rustup`).
3. Run checks before opening a PR:

```bash
cargo fmt --all
cargo clippy --all-targets --no-default-features --features infer-serial -- -D warnings
cargo clippy --all-targets --no-default-features --features infer-row-parallel -- -D warnings
cargo clippy --all-targets --no-default-features --features infer-tree-parallel -- -D warnings
cargo test --all-targets --no-default-features --features infer-serial
cargo test --all-targets --no-default-features --features infer-row-parallel
cargo test --all-targets --no-default-features --features infer-tree-parallel
```

## Pull Request Guidelines

- Keep PRs focused and small when possible.
- Add or update tests when behavior changes.
- Update docs for user-facing changes.
- Write clear commit messages and PR descriptions.

## Code Style

- Follow `rustfmt` formatting.
- Treat clippy warnings as errors in CI.
- Use Rust naming conventions consistently:
  - modules/files/functions: `snake_case`
  - types/traits/enums: `UpperCamelCase`
  - acronymed type names follow Rust style (for example, `XgbModel`, not `XGBModel`)
- Prefer domain-accurate names for modules and types; avoid implementation-specific or ambiguous names.

## Questions

If anything is unclear, open an issue and ask.
