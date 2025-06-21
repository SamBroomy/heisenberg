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
    cargo install cargo-audit
    cargo install cargo-machete

# Setup development build
[group('dev')]
dev:
    uv run maturin develop -r

# =============================================================================
# Testing
# =============================================================================

# Run Python tests and examples
[group('ci')]
[group('test')]
pytest: dev
    #!/usr/bin/env bash
    set -euo pipefail
    set -x
    uv run python -m pytest python/tests/ -v

    for file in python/examples/*.py; do
        uv run python "$file";
    done

# Run Rust tests with multiple feature combinations
[group('ci')]
[group('test')]
rust-test:
    cargo test -- --test-threads=1
    cargo test --examples --release
    cargo test --no-default-features -- --test-threads=1
    cargo test --no-default-features --features serde -- --test-threads=1

# Run all tests
[group('ci')]
[group('test')]
test: rust-test pytest

# =============================================================================
# Linting & Formatting
# =============================================================================

# Check Rust code with clippy and fmt
[group('ci')]
[group('lint')]
rust-lint:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --no-default-features -- -D warnings
    cargo clippy --all-targets --no-default-features --features serde -- -D warnings
    cargo fmt --check

# Fix Rust code with clippy and fmt (for precommit)
[group('lint')]
[group('precommit')]
rust-lint-fix:
    cargo clippy --workspace --all-targets --fix --allow-staged --allow-dirty --quiet -- -D warnings
    cargo clippy --workspace --all-targets --no-default-features --fix --allow-staged --allow-dirty --quiet -- -D warnings

# Check Python code with ruff
[group('ci')]
[group('lint')]
python-lint:
    uv run ruff check python/
    uv run ruff format --check python/

# Check maturin can build Python bindings
[group('lint')]
[group('precommit')]
maturin-check:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔍 Checking maturin build..."
    uv sync --dev --quiet
    echo "✅ Maturin build check passed"

# Test crate publishing without actually publishing
[group('lint')]
[group('precommit')]
publish-dry-run:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🧪 Testing crate publishing (dry run)..."
    cd crates/heisenberg-data-processing && cargo publish --dry-run --quiet --allow-dirty
    cd ../heisenberg && cargo publish --dry-run --quiet --allow-dirty
    echo "✅ Publish dry run completed successfully"

# Run all linting
[group('ci')]
[group('lint')]
lint: rust-lint python-lint

# Fix linting issues
[group('lint')]
fix:
    cargo fmt
    cargo clippy --fix --allow-dirty --allow-staged
    uv run ruff check --fix .
    uv run ruff format .

# Remove unused dependencies
[group('ci')]
[group('lint')]
[group('precommit')]
cargo-machete:
    cargo machete --with-metadata --fix

# Check the docs
[group('ci')]
[group('lint')]
cargo-docs:
    cargo doc --all-features --no-deps

# Cargo audit
[group('ci')]
[group('lint')]
cargo-audit:
    cargo audit --deny unsound --deny yanked

# Lint the docs
[group('ci')]
[group('lint')]
ci-lint: lint cargo-machete cargo-docs

# =============================================================================
# Building
# =============================================================================

# Build Python release wheel
[group('build')]
[group('ci')]
build-python:
    rm -rf dist/ target/wheels/
    uv run maturin build --features python --release

# Build wheels for all platforms
[group('build')]
build-all:
    uv run maturin build --features python --release --target x86_64-apple-darwin
    uv run maturin build --features python --release --target aarch64-apple-darwin
    uv run maturin build --features python --release --target x86_64-unknown-linux-gnu

# Build Rust crates
[group('build')]
rust-build:
    cargo build --release

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
# Environment Management
# =============================================================================

# Clean Python artifacts
[group('env')]
clean-python:
    find . -type f -name '*.py[co]' -delete 2>/dev/null || true
    find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.cpython*.so' -not -path './.venv/*' -not -path './target/*' -not -path './.git/*' -delete 2>/dev/null || true

# Clean Rust artifacts
[group('env')]
clean-rust:
    cargo clean

# Clean data files
[group('env')]
clean-data:
    rm -rf heisenberg_data
    find crates/heisenberg/src/data/embedded -type f \( -name '*.parquet' -o -name '*.json' \) -delete 2>/dev/null || true

# Clean virtual environment
[group('env')]
clean-venv:
    rm -rf .venv

# Clean everything
[group('env')]
clean: clean-data clean-rust clean-python clean-venv

# =============================================================================
# Release Management
# =============================================================================

# Check if project is ready for release
[group('ci')]
[group('publish')]
check-release:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔍 Checking release readiness..."

    # Check if working directory is clean
    if [ -n "$(git status --porcelain)" ]; then
        echo "❌ Working directory is not clean. Please commit all changes."
        exit 1
    fi

    # Check if on main/master branch
    BRANCH=$(git branch --show-current)
    if [[ "$BRANCH" != "main" && "$BRANCH" != "master" ]]; then
        echo "❌ Not on main/master branch. Currently on: $BRANCH"
        exit 1
    fi

    # Run linting
    echo "🔍 Running linting..."
    just lint

    # Run tests
    echo "🧪 Running tests..."
    just test


    echo "✅ Ready for release!"

# Publish Rust crates to crates.io
[group('publish')]
publish-rust: check-release
    #!/usr/bin/env bash
    set -euo pipefail
    echo "📦 Publishing Rust crates to crates.io..."

    # Publish heisenberg-data-processing first (dependency)
    echo "Publishing heisenberg-data-processing..."
    cd crates/heisenberg-data-processing
    cargo publish --dry-run
    cargo publish
    cd ../..

    # Wait a bit for crates.io to register the new crate
    echo "⏳ Waiting for crates.io to update..."
    sleep 60

    # Publish heisenberg
    echo "Publishing heisenberg..."
    cd crates/heisenberg
    cargo publish --dry-run
    cargo publish
    cd ../..

    echo "✅ Rust crates published successfully!"

# Build Python package for PyPI
[group('publish')]
build-python-package:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🐍 Building Python package..."

    # Clean previous builds
    rm -rf dist/ target/wheels/

    # Build wheels
    uv run maturin build --features python --release

    echo "✅ Python package built successfully!"

# Publish Python package to PyPI
[group('publish')]
publish-python: build-python-package
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🐍 Publishing Python package to PyPI..."

    # Install twine if not available
    if ! command -v twine &> /dev/null; then
        uv tool install twine
    fi

    # Upload to PyPI
    uvx twine upload target/wheels/*

    echo "✅ Python package published successfully!"

# Test publish to Test PyPI
[group('publish')]
publish-python-test: build-python-package
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🧪 Publishing Python package to Test PyPI..."

    # Install twine if not available
    if ! command -v twine &> /dev/null; then
        uv tool install twine
    fi

    # Upload to Test PyPI
    uvx twine upload --repository testpypi target/wheels/*

    echo "✅ Python package published to Test PyPI!"

# Publish everything (for CI)
[group('publish')]
publish-package: build-python-package
    #!/usr/bin/env bash
    set -euo pipefail
    echo "📦 Publishing Python package via trusted publishing..."

    # This assumes we're in GitHub Actions with OIDC token
    # The actual upload will be handled by pypa/gh-action-pypi-publish
    echo "Python wheels built and ready for trusted publishing"

    # Just verify the wheels exist
    ls -la target/wheels/

    echo "✅ Package ready for publication!"

# Get next version suggestions
[group('publish')]
next-version:
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | sed 's/.*"\(.*\)".*/\1/')
    echo "Current version: $CURRENT_VERSION"

    IFS='.' read -ra PARTS <<< "$CURRENT_VERSION"
    MAJOR=${PARTS[0]}
    MINOR=${PARTS[1]}
    PATCH=${PARTS[2]}

    NEXT_PATCH=$((PATCH + 1))
    NEXT_MINOR=$((MINOR + 1))
    NEXT_MAJOR=$((MAJOR + 1))

    echo "Suggested versions:"
    echo "  Patch: $MAJOR.$MINOR.$NEXT_PATCH (bug fixes)"
    echo "  Minor: $MAJOR.$NEXT_MINOR.0 (new features)"
    echo "  Major: $NEXT_MAJOR.0.0 (breaking changes)"
    echo ""
    echo "Usage:"
    echo "  just release $MAJOR.$MINOR.$NEXT_PATCH"

# Retry current version release
[group('publish')]
retry-release: check-release
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | sed 's/.*"\(.*\)".*/\1/')
    echo "🔄 Retrying release for version $CURRENT_VERSION..."
    just release "$CURRENT_VERSION"

# Create and publish a new release
[group('publish')]
release VERSION: check-release
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION="{{ VERSION }}"

    echo "🚀 Creating release $VERSION..."

    # Check if version is already set
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | sed 's/.*"\(.*\)".*/\1/')

    if [[ "$CURRENT_VERSION" == "$VERSION" ]]; then
        echo "ℹ️  Version is already set to $VERSION"

        # Check if tag already exists
        if git tag -l | grep -q "^v$VERSION$"; then
            echo "🏷️  Tag v$VERSION already exists"

            # Check if it's also on remote
            if git ls-remote --tags origin | grep -q "refs/tags/v$VERSION$"; then
                echo "🌐 Tag also exists on remote"

                # Check if crates are already published
                echo "🔍 Checking if version is already published..."
                if cargo search heisenberg --limit 1 | grep -q "heisenberg = \"$VERSION\""; then
                    echo "❌ Version $VERSION is already published to crates.io"
                    echo "💡 Tip: Use a higher version number, e.g.:"
                    IFS='.' read -ra PARTS <<< "$VERSION"
                    PATCH=$((PARTS[2] + 1))
                    echo "   just release ${PARTS[0]}.${PARTS[1]}.$PATCH"
                    exit 1
                else
                    echo "🗑️  Cleaning up failed release attempt..."
                    # Delete local and remote tags
                    git tag -d "v$VERSION" 2>/dev/null || true
                    git push origin --delete "v$VERSION" 2>/dev/null || true

                    # Delete GitHub release if it exists
                    gh release delete "v$VERSION" --yes 2>/dev/null || true

                    echo "✅ Cleaned up. Proceeding with release..."
                fi
            else
                echo "🗑️  Deleting local tag..."
                git tag -d "v$VERSION"
            fi
        fi

        echo "📝 Creating tag for existing version..."
    else
        echo "📝 Updating version from $CURRENT_VERSION to $VERSION..."

        # Update version in Cargo.toml files
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" crates/heisenberg/Cargo.toml
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" crates/heisenberg-data-processing/Cargo.toml
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

        # Update workspace dependency versions
        sed -i.bak "s/heisenberg = { version = \".*\", path = \"crates\/heisenberg\" }/heisenberg = { version = \"$VERSION\", path = \"crates\/heisenberg\" }/" Cargo.toml
        sed -i.bak "s/heisenberg-data-processing = { version = \".*\", path = \"crates\/heisenberg-data-processing\" }/heisenberg-data-processing = { version = \"$VERSION\", path = \"crates\/heisenberg-data-processing\" }/" Cargo.toml

        # Remove backup files
        find . -name "*.bak" -delete

        # Update Cargo.lock
        cargo update

        # Commit changes
        git add .
        git commit -m "chore: bump version to $VERSION"
    fi

    # Create and push tag
    git tag "v$VERSION"
    CURRENT_BRANCH=$(git branch --show-current)
    git push origin "$CURRENT_BRANCH"
    git push origin "v$VERSION"

    echo "✅ Release $VERSION created and pushed!"
