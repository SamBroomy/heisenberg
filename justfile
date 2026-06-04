set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set shell := ["bash", "-uc"]

# Default recipe shows available commands
default:
    @just --list

[group('setup')]
bootstrap: && install-pre-commit
    uv sync --dev --all-extras --all-groups

# Install pre-commit hooks
[group('setup')]
install-pre-commit:
    prek install

# Setup development build
[group('dev')]
build-bindings:
    uv run --directory python/heisenberg-geo maturin develop --release --uv

# =============================================================================
# Testing
# =============================================================================

# Run Python tests and examples
[group('ci')]
[group('test')]
pytest: build-bindings
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
    cargo test --lib
    cargo test --doc -- --test-threads=1
    cargo test --examples
    cargo test --test integration_tests
    cargo test --no-default-features --lib
    cargo test --no-default-features --doc -- --test-threads=1
    cargo test --no-default-features --features serde --lib
    cargo test --no-default-features --features serde --doc -- --test-threads=1

# Run all tests (full)
[group('test')]
test: rust-test pytest

# Fast CI Rust tests only
[group('ci')]
[group('test')]
rust-test-ci:
    cargo test --lib
    cargo test --doc -- --test-threads=1
    cargo test --test integration_tests

# Fast CI Python tests only (requires pre-built bindings)
[group('ci')]
[group('test')]
python-test-ci:
    uv run python -m pytest python/tests/ -v --maxfail=3

# Build only Rust dependencies (no Python linking)
[group('build')]
[group('ci')]
rust-build-deps:
    cargo build --lib
    cargo build --lib --no-default-features
    cargo test --no-run
    cargo test --no-run --no-default-features

# =============================================================================
# Linting & Formatting
# =============================================================================

# Check Rust code with clippy and fmt (full check)
[group('lint')]
rust-lint:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --no-default-features -- -D warnings
    cargo clippy --all-targets --no-default-features --features serde -- -D warnings
    cargo fmt --check

# Fast Rust lint for CI (only default features)
[group('ci')]
[group('lint')]
rust-lint-ci:
    cargo clippy --all-targets -- -D warnings
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
    uv run --no-project ruff check python/
    uv run --no-project ruff format --check python/

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
    cd ../heisenberg-geo && cargo publish --dry-run --quiet --allow-dirty
    echo "✅ Publish dry run completed successfully"

# Run all linting (full)
[group('lint')]
lint: rust-lint python-lint

# Run CI linting (fast)
[group('ci')]
[group('lint')]
lint-ci: rust-lint-ci python-lint

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

# Lint for CI (fast version)
[group('ci')]
[group('lint')]
ci-lint: lint-ci cargo-machete cargo-docs

# =============================================================================
# Building
# =============================================================================

# Build Rust crates
[group('build')]
rust-build:
    cargo build --release

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
    find crates/heisenberg-geo/src/data/embedded -type f \( -name '*.parquet' -o -name '*.json' \) -delete 2>/dev/null || true

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
publish-rust:
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

    # Publish heisenberg-geo
    echo "Publishing heisenberg-geo..."
    cd crates/heisenberg-geo
    cargo publish --dry-run
    cargo publish
    cd ../..

    echo "✅ Rust crates published successfully!"

# Build Python package for PyPI
[group('build')]
[group('ci')]
[group('publish')]
build-python:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🐍 Building Python package..."

    # Clean previous builds
    rm -rf dist/ target/wheels/

    # Build wheels
    uv run --directory python/heisenberg-geo maturin build --features python --release

    echo "✅ Python package built successfully!"

# Publish Python package to PyPI
[group('publish')]
publish-python: build-python
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
publish-python-test: build-python
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

# Bump patch version and release (0.1.0 -> 0.1.1)
[group('publish')]
bump-patch: check-release
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | sed 's/.*"\(.*\)".*/\1/')
    IFS='.' read -ra PARTS <<< "$CURRENT_VERSION"
    MAJOR=${PARTS[0]}
    MINOR=${PARTS[1]}
    PATCH=${PARTS[2]}
    NEXT_PATCH=$((PATCH + 1))
    NEW_VERSION="$MAJOR.$MINOR.$NEXT_PATCH"
    echo "📦 Bumping patch version: $CURRENT_VERSION → $NEW_VERSION"
    just release "$NEW_VERSION"

# Bump minor version and release (0.1.0 -> 0.2.0)
[group('publish')]
bump-minor: check-release
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | sed 's/.*"\(.*\)".*/\1/')
    IFS='.' read -ra PARTS <<< "$CURRENT_VERSION"
    MAJOR=${PARTS[0]}
    MINOR=${PARTS[1]}
    NEXT_MINOR=$((MINOR + 1))
    NEW_VERSION="$MAJOR.$NEXT_MINOR.0"
    echo "📦 Bumping minor version: $CURRENT_VERSION → $NEW_VERSION"
    just release "$NEW_VERSION"

# Bump major version and release (0.1.0 -> 1.0.0)
[group('publish')]
bump-major: check-release
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | sed 's/.*"\(.*\)".*/\1/')
    IFS='.' read -ra PARTS <<< "$CURRENT_VERSION"
    MAJOR=${PARTS[0]}
    NEXT_MAJOR=$((MAJOR + 1))
    NEW_VERSION="$NEXT_MAJOR.0.0"
    echo "📦 Bumping major version: $CURRENT_VERSION → $NEW_VERSION"
    just release "$NEW_VERSION"

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
                if cargo search heisenberg_geo --limit 1 | grep -q "heisenberg_geo = \"$VERSION\""; then
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
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" crates/heisenberg-geo/Cargo.toml
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" crates/heisenberg-data-processing/Cargo.toml
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

        # Update workspace dependency versions
        sed -i.bak "s/heisenberg-geo = { version = \".*\", path = \"crates\/heisenberg-geo\" }/heisenberg-geo = { version = \"$VERSION\", path = \"crates\/heisenberg-geo\" }/" Cargo.toml
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

# Clean all caches (embedded data + indexes)
[group('env')]
clean-cache:
    #!/usr/bin/env bash
    set -euo pipefail

    # Determine cache directory based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CACHE_DIR="${HOME}/Library/Caches/heisenberg-geo"
    else
        CACHE_DIR="${HOME}/.cache/heisenberg-geo"
    fi

    if [ -d "$CACHE_DIR" ]; then
        echo "🗑️  Cleaning cache at $CACHE_DIR"

        # Show what we're removing
        echo ""
        echo "📊 Current cache contents:"
        du -sh "$CACHE_DIR"/* 2>/dev/null || echo "  (empty)"

        # Remove it
        rm -rf "$CACHE_DIR"

        echo ""
        echo "✅ Cache cleaned"
    else
        echo "ℹ️  No cache directory found at $CACHE_DIR"
    fi

# Clean old version caches (keep current version only)
[group('env')]
clean-old-caches:
    #!/usr/bin/env bash
    set -euo pipefail

    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | sed 's/.*"\(.*\)".*/\1/')

    # Determine cache directory based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CACHE_DIR="${HOME}/Library/Caches/heisenberg-geo/embedded_data"
    else
        CACHE_DIR="${HOME}/.cache/heisenberg-geo/embedded_data"
    fi

    if [ -d "$CACHE_DIR" ]; then
        echo "🔍 Current version: $CURRENT_VERSION"
        echo "📂 Checking cache at: $CACHE_DIR"
        echo ""

        # Show what will be kept/removed
        echo "📊 Cache versions:"
        for dir in "$CACHE_DIR"/*; do
            if [ -d "$dir" ]; then
                VERSION=$(basename "$dir")
                SIZE=$(du -sh "$dir" | cut -f1)
                if [ "$VERSION" = "$CURRENT_VERSION" ]; then
                    echo "  ✅ Keep: $VERSION ($SIZE)"
                else
                    echo "  ❌ Remove: $VERSION ($SIZE)"
                fi
            fi
        done

        # Remove old versions
        echo ""
        find "$CACHE_DIR" -maxdepth 1 -type d ! -name "$CURRENT_VERSION" ! -path "$CACHE_DIR" -exec rm -rf {} \;

        echo "✅ Old caches cleaned (kept $CURRENT_VERSION)"
    else
        echo "ℹ️  No embedded data cache found"
    fi

# Show cache statistics
[group('env')]
cache-stats:
    #!/usr/bin/env bash
    set -euo pipefail

    # Determine cache directory based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        BASE_CACHE="${HOME}/Library/Caches/heisenberg-geo"
    else
        BASE_CACHE="${HOME}/.cache/heisenberg-geo"
    fi

    echo "📊 Heisenberg Cache Statistics"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [ ! -d "$BASE_CACHE" ]; then
        echo "ℹ️  No cache directory found at $BASE_CACHE"
        exit 0
    fi

    # Embedded data cache
    EMBEDDED_CACHE="$BASE_CACHE/embedded_data"
    if [ -d "$EMBEDDED_CACHE" ]; then
        echo "📦 Embedded Data Cache:"
        for version_dir in "$EMBEDDED_CACHE"/*; do
            if [ -d "$version_dir" ]; then
                VERSION=$(basename "$version_dir")
                for source_dir in "$version_dir"/*; do
                    if [ -d "$source_dir" ]; then
                        SOURCE=$(basename "$source_dir")
                        SIZE=$(du -sh "$source_dir" | cut -f1)
                        echo "  • $VERSION / $SOURCE: $SIZE"
                    fi
                done
            fi
        done
    else
        echo "📦 Embedded Data Cache: (empty)"
    fi

    echo ""

    # Index cache
    INDEX_CACHE="$BASE_CACHE/indexes"
    if [ -d "$INDEX_CACHE" ]; then
        echo "🔍 Index Cache:"
        for hash_dir in "$INDEX_CACHE"/*; do
            if [ -d "$hash_dir" ]; then
                HASH=$(basename "$hash_dir")
                SIZE=$(du -sh "$hash_dir" | cut -f1)
                echo "  • Hash $HASH: $SIZE"
            fi
        done
    else
        echo "🔍 Index Cache: (empty)"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    TOTAL_SIZE=$(du -sh "$BASE_CACHE" | cut -f1)
    echo "💾 Total cache size: $TOTAL_SIZE"
