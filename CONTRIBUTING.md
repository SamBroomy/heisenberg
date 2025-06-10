# Contributing to Heisenberg

Thank you for your interest in contributing to Heisenberg! This document provides guidelines and information for contributors.

## 🚀 Quick Start

### Prerequisites

- **Rust** (latest stable): Install via [rustup](https://rustup.rs/)
- **Python** 3.10+ (for Python bindings)
- **UV** (recommended): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Just** (optional but recommended): `cargo install just`

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SamBroomy/heisenberg.git
   cd heisenberg
   ```

2. **Install dependencies**:
   ```bash
   # Using just (recommended)
   just init

   # Or manually
   uv tool install maturin
   ```

3. **Build and test**:
   ```bash
   # Quick test with test data (recommended for development)
   just test

   # Or manually
   USE_TEST_DATA=true cargo test --features test_data
   python -m pytest python/tests/ -v
   ```

4. **Development build**:
   ```bash
   # Build Python bindings for local development
   just dev

   # Or manually
   uv run maturin develop -r
   ```

## 🧪 Testing Strategy

### Test Data vs Production Data

Heisenberg uses a **test data system** to enable fast development and testing:

```bash
# ✅ Fast tests with small dataset (~100 locations)
USE_TEST_DATA=true cargo test --features test_data

# ❌ Slow tests with full GeoNames dataset (11M+ locations, 2-5 min setup)
cargo test --release
```

**Always use test data during development** unless specifically testing full dataset behavior.

### Test Categories

1. **Unit Tests** (`src/lib.rs`, modules): Individual component testing
2. **Integration Tests** (`tests/`): End-to-end workflow testing  
3. **Python Tests** (`python/tests/`): Python binding validation
4. **Example Tests** (`examples/`): Documentation and usage validation

### Running Tests

```bash
# All tests with test data (recommended)
just test

# Rust tests only
just rust-test

# Python tests only
just pytest

# Specific test
USE_TEST_DATA=true cargo test test_searcher_creation --features test_data -- --nocapture

# With debug logging
RUST_LOG=heisenberg=debug USE_TEST_DATA=true cargo test --features test_data
```

## 📁 Project Structure

```
heisenberg/
├── src/                    # Rust source code
│   ├── lib.rs             # Main library entry point
│   ├── core.rs            # Core LocationSearcher implementation
│   ├── search/            # Search algorithms and orchestration
│   ├── index/             # Full-text search indexing (Tantivy)
│   ├── data/              # Data loading and processing
│   ├── backfill/          # Location resolution and enrichment
│   ├── config/            # Configuration builders
│   └── python/            # Python bindings (PyO3)
├── python/                # Python package
│   ├── heisenberg/        # Python module
│   ├── examples/          # Python usage examples
│   └── tests/             # Python tests
├── examples/              # Rust usage examples
├── tests/                 # Integration tests
├── justfile              # Development commands
├── Cargo.toml            # Rust dependencies and configuration
├── pyproject.toml        # Python package configuration
└── docs/                 # Additional documentation
```

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines below

3. **Test thoroughly**:
   ```bash
   # Test your changes
   just test
   
   # Test examples still work
   cargo test --examples --release
   
   # Run Python examples
   python python/examples/01_basic_usage.py
   ```

4. **Run code quality checks**:
   ```bash
   # Format code
   cargo fmt
   
   # Check for issues
   cargo clippy --all-targets --all-features -- -D warnings
   
   # Check Python formatting (if you modified Python code)
   ruff check python/ --fix
   ruff format python/
   ```

5. **Update documentation** if needed:
   ```bash
   # Generate and check Rust docs
   cargo doc --no-deps --open
   
   # Update Python examples if API changed
   ```

### Code Style Guidelines

#### Rust Code

- **Follow standard Rust conventions**: Use `rustfmt` and `clippy`
- **Document public APIs**: All public functions, structs, and modules need doc comments
- **Use descriptive names**: `LocationSearcher` not `Searcher`, `resolve_location` not `resolve`
- **Error handling**: Use `Result<T, HeisenbergError>` consistently
- **Testing**: Add tests for new functionality using test data

```rust
/// Search for locations matching the given terms.
///
/// This performs an intelligent search across administrative entities and places,
/// automatically determining the best search strategy based on the input.
///
/// # Arguments
///
/// * `terms` - Location terms to search for (e.g., ["Paris", "France"])
///
/// # Examples
///
/// ```rust
/// use heisenberg::LocationSearcher;
///
/// let searcher = LocationSearcher::new(false)?;
/// let results = searcher.search(&["Tokyo"])?;
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
pub fn search<Term>(&self, terms: &[Term]) -> Result<SearchResults, HeisenbergError>
where
    Term: AsRef<str>,
{
    // Implementation...
}
```

#### Python Code

- **Follow PEP 8**: Use `ruff` for formatting and linting
- **Type hints**: Use type annotations for function signatures
- **Docstrings**: Follow Google docstring style
- **Test coverage**: Add tests for new bindings in `python/tests/`

```python
def find_location(query: Union[str, List[str]], config: Optional[SearchOptions] = None) -> List[SearchResult]:
    """Find locations matching the given query.
    
    Args:
        query: Location query as string or list of terms
        config: Optional search configuration
        
    Returns:
        List of matching search results
        
    Examples:
        >>> results = find_location("Tokyo")
        >>> results = find_location(["Paris", "France"])
    """
```

### Performance Considerations

- **Use test data for development**: Full dataset tests should be in CI only
- **Benchmark significant changes**: Use `criterion` for micro-benchmarks
- **Memory efficiency**: Be conscious of memory usage with large datasets
- **Batch operations**: Prefer batch APIs for multiple operations

### Dependencies

- **Minimize new dependencies**: Justify new crates in PR description
- **Use established crates**: Prefer popular, well-maintained dependencies  
- **Optional features**: Use feature flags for heavy dependencies
- **Python compatibility**: Ensure Python bindings work across Python 3.10+

## 🐛 Reporting Issues

### Bug Reports

Include the following information:

1. **Heisenberg version**: Check with `pip show heisenberg` or `cargo tree`
2. **Operating system**: macOS, Linux, Windows version
3. **Python version**: If using Python bindings
4. **Minimal reproduction**: Smallest possible code that demonstrates the issue
5. **Expected vs actual behavior**: What you expected vs what happened
6. **Data environment**: Using test data or full dataset?

### Feature Requests

- **Use case description**: Why is this feature needed?
- **Proposed API**: What should the interface look like?
- **Alternative solutions**: What workarounds exist currently?
- **Breaking changes**: Would this require API changes?

## 📝 Pull Request Process

### Before Submitting

1. **Check existing issues/PRs**: Avoid duplicate work
2. **Run full test suite**: `just test` should pass
3. **Update documentation**: Add docstrings, update examples if needed
4. **Add tests**: New functionality needs test coverage
5. **Check examples**: Ensure examples still work

### PR Requirements

- **Clear description**: What does this PR do and why?
- **Test coverage**: New code should have tests
- **Documentation**: Public APIs need documentation
- **No breaking changes**: Unless discussed in issues first
- **Clean commit history**: Squash fixup commits

### PR Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass with `just test`
- [ ] Added tests for new functionality
- [ ] Examples still work
- [ ] Tested with both test data and full dataset (if applicable)

## Documentation
- [ ] Added/updated docstrings for new public APIs
- [ ] Updated examples if API changed
- [ ] Updated README if needed
```

## 🌟 Areas for Contribution

### Good First Issues

- **Documentation improvements**: Fix typos, improve examples
- **Error message improvements**: Make error messages more helpful
- **Test coverage**: Add tests for edge cases
- **Performance optimizations**: Profile and optimize hot paths
- **Configuration validation**: Better validation and error messages

### Advanced Contributions

- **Search algorithm improvements**: Better relevance scoring
- **Data processing optimization**: Faster index building
- **Python API enhancements**: More Pythonic interfaces
- **Integration examples**: Jupyter notebooks, web frameworks
- **Alternative data sources**: Support for other geographic datasets

### Architecture Improvements

- **Streaming data processing**: Handle larger-than-memory datasets
- **Distributed search**: Scale across multiple machines
- **Real-time updates**: Live data updates from GeoNames
- **Machine learning integration**: Embedding-based search

## 📚 Learning Resources

### Understanding the Codebase

1. **Start with examples**: Run and read `examples/*.rs` and `python/examples/*.py`
2. **Read the README**: Understand the high-level goals
3. **Study the tests**: `tests/integration_tests.rs` shows full workflows
4. **Check documentation**: `cargo doc --open` for API reference

### Key Concepts

- **Administrative Hierarchy**: Country → State → County → Place structure
- **Full-Text Search**: Tantivy-powered search with custom scoring
- **Resolution**: Converting search results to complete hierarchies
- **Batch Processing**: Efficient processing of multiple queries

### Technologies Used

- **Rust**: Core language and performance-critical code
- **PyO3**: Python-Rust bindings
- **Tantivy**: Full-text search engine (like Lucene)
- **Polars**: Fast DataFrame library for data processing
- **GeoNames**: Geographic database and data source

## 🤝 Community Guidelines

- **Be respectful**: Treat all contributors with respect
- **Be patient**: Reviews take time, especially for complex changes
- **Ask questions**: Don't hesitate to ask for clarification
- **Help others**: Review other PRs, answer questions in issues
- **Share knowledge**: Write blog posts, give talks about your contributions

## 📞 Getting Help

- **GitHub Issues**: For bugs, feature requests, and questions
- **GitHub Discussions**: For general questions and community discussion
- **Email**: maintainer email for sensitive issues

---

Thank you for contributing to Heisenberg! Every contribution, no matter how small, helps make location data more accessible and reliable for everyone. 🗺️✨