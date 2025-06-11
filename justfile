set shell := ["bash", "-uc"]

# List available recipes
help:
    @just --list

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

# Run the pre-commit hooks
[group('git-hooks')]
run-hooks: install-pre-commit run-pre-commit run-pre-push

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
    cargo test -- --test-threads=1
    cargo test --examples --release

test: rust-test pytest

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

# Clean the project
[group('env')]
clean: clean-data clean-rust clean-py-project clean-venv
