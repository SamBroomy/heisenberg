set shell := ["bash", "-uc"]

# List available recipes
help:
    @just --list

# =============================================================================
# Development Environment
# =============================================================================

# Initialize development environment
[group('dev')]
init:
    uv tool install maturin

# Setup development build
[group('dev')]
dev: init
    uv run maturin develop -r

# =============================================================================
# Git Hooks
# =============================================================================

# Install Git hooks using prefligit
[group('git-hooks')]
install-pre-commit:
    #!/usr/bin/env sh
    if ! command -v prefligit &> /dev/null; then
        echo "Installing prefligit..."
        cargo install --locked --git https://github.com/j178/prefligit
    else
        echo "prefligit is already installed"
    fi
    prefligit install
    prefligit run --all-files

# Run the pre-commit hooks
[group('git-hooks')]
run-pre-commit:
    prefligit run --all-files

# Run the pre-push hooks
[group('git-hooks')]
run-pre-push:
    prefligit run --hook-stage pre-push

# Run all hooks
[group('git-hooks')]
run-hooks: install-pre-commit run-pre-commit run-pre-push

# =============================================================================
# Testing
# =============================================================================

# Run Python tests
[group('test')]
pytest: dev
    #!/usr/bin/env bash
    set -euo pipefail
    set -x
    uv run python -m pytest python/tests/ -v

    for file in python/examples/*.py; do
        uv run python "$file";
    done

# Run Rust tests
[group('test')]
rust-test:
    cargo test -- --test-threads=1
    cargo test --examples --release

# Run all tests
[group('test')]
test: rust-test pytest

# =============================================================================
# Linting
# =============================================================================

# Run Rust linting
[group('lint')]
rust-lint:
    cargo clippy --all-targets --all-features -- -D warnings
    cargo fmt --check

# Run Python linting
[group('lint')]
python-lint:
    uv run ruff check python/
    uv run ruff format --check python/

# Run all linting
[group('lint')]
lint: rust-lint python-lint

# Fix linting issues
[group('lint')]
fix:
    cargo fmt
    cargo clippy --fix --allow-dirty --allow-staged
    uv run ruff check --fix .
    uv run ruff format .

# =============================================================================
# Building
# =============================================================================

# Build release wheel
[group('build')]
build: init
    uv run maturin build --features python --release

# Build all wheel platforms
[group('build')]
build-all: init
    uv run maturin build --features python --release --target x86_64-apple-darwin
    uv run maturin build --features python --release --target aarch64-apple-darwin
    uv run maturin build --features python --release --target x86_64-unknown-linux-gnu

# Build Rust crates
[group('build')]
rust-build:
    cargo build --release --all-features

# Clean the project
[group('env')]
clean-py-project:
    # Remove Python cache files
    find . -type f -name '*.py[co]' -delete 2>/dev/null || true
    find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true

    # Remove cpython files (excluding .venv and other common dirs)
    find . -type f -name '*.cpython*.so' \
        -not -path './.venv/*' \
        -not -path './target/*' \
        -not -path './.git/*' \
        -delete 2>/dev/null || true

# Remove the virtual environment
[group('env')]
clean-venv:
    rm -rf .venv

# Remove all rust build artifacts
[group('env')]
clean-rust:
    cargo clean

# Remove any generated data files
[group('env')]
clean-data:
    rm -rf heisenberg_data
    find crates/heisenberg/src/data/embedded -type f -name '*.parquet' -delete 2>/dev/null || true
    find crates/heisenberg/src/data/embedded -type f -name '*.json' -delete 2>/dev/null || true

# Clean the project
[group('env')]
clean: clean-data clean-rust clean-py-project clean-venv
