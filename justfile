init:
    uv tool install maturin


dev: init
    uv run maturin develop -r


# Build release wheel
build: init
    uv run maturin build --features python --release

# Run Python tests
pytest: dev
    #!/usr/bin/env bash
    set -euo pipefail
    set -x
    uv run python -m pytest python/tests/ -v

    for file in python/examples/*.py; do
        uv run python "$file";
    done

# Run Rust tests
rust-test:
    USE_TEST_DATA=true cargo test -- --test-threads=1
    cargo test --examples --release

test: pytest rust-test

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/wheels/