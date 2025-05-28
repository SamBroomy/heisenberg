init:
    uv tool install maturin


dev: init
    uv run maturin develop -r


# Build release wheel
build: init
    uv run maturin build --features python --release

# Run Python tests
test-py:
    python -m pytest python/tests/

# Run Rust tests
test-rust:
    cargo test --features python

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/wheels/