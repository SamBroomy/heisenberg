init:
    uv tool install maturin


dev: init
    uv run maturin develop -r


# Build release wheel
build: init
    uv run maturin build --features python --release

# Run Python tests
pytest: dev
    python -m pytest python/tests/ -v

# Run Rust tests
rust-test:
    USE_TEST_DATA=true cargo test --features test_data
    cargo test --examples --release

test: pytest rust-test

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/wheels/