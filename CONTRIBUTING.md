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
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## Pull Request Guidelines

- Keep PRs focused and small when possible.
- Add or update tests when behavior changes.
- Update docs for user-facing changes.
- Write clear commit messages and PR descriptions.

## Code Style

- Follow `rustfmt` formatting.
- Treat clippy warnings as errors in CI.

## Questions

If anything is unclear, open an issue and ask.
