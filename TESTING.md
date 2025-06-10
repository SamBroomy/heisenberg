# Testing Strategy for Heisenberg

This document outlines the testing approach for the Heisenberg location search library, including how to handle the challenge of testing with large external datasets.

## Test Data Strategy

Heisenberg uses the **test data feature** to run tests with smaller, faster datasets instead of the full GeoNames data (~500MB download + processing time).

### Environment Setup

All tests should use the test data environment:

```bash
# Rust tests
USE_TEST_DATA=true cargo test --features test_data

# Python tests  
USE_TEST_DATA=true python -m pytest python/tests/ -v

# Run all tests (via justfile)
just test
```

### Test Data vs Production Data

| Aspect | Test Data | Production Data |
|--------|-----------|-----------------|
| **Size** | ~100 locations | 11+ million locations |
| **Download** | None (embedded) | ~500MB from GeoNames |
| **Index Time** | <1 second | 2-5 minutes |
| **Memory** | <10MB | ~200MB |
| **Use Case** | Testing, CI/CD | Production usage |

## Test Structure

### 1. Unit Tests (`src/lib.rs` and modules)

Located in `#[cfg(test)]` modules within source files.

**Focus**: Individual function and component testing
- Configuration validation
- Search parameter parsing
- Error handling
- Data transformation logic

**Example**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn setup_test_env() {
        env::set_var("USE_TEST_DATA", "true");
        let _ = init_logging(tracing::Level::WARN);
    }

    #[test]
    fn test_searcher_creation() {
        setup_test_env();
        let searcher = LocationSearcher::new(false);
        assert!(searcher.is_ok());
    }
}
```

### 2. Integration Tests (`tests/`)

Located in `tests/integration_tests.rs` and similar files.

**Focus**: End-to-end workflow testing
- Complete search workflows
- Multi-component interactions
- API contract validation
- Performance characteristics

**Coverage**:
- Search → Resolution workflows
- Batch processing
- Configuration presets
- Error handling and edge cases
- Concurrent access

### 3. Python Tests (`python/tests/`)

Located in `python/tests/test_bindings.py`.

**Focus**: Python binding correctness
- All Python API methods
- Configuration validation
- Type safety and conversions
- Error propagation from Rust
- Example code validation

**Coverage**:
- Basic functionality (`LocationSearcher`)
- Configuration (`SearchOptions`, `SearchConfigBuilder`)
- Advanced features (`Heisenberg` direct API)
- Entry types (`BasicEntry`, `GenericEntry`)
- Batch operations and edge cases

### 4. Example Tests

Examples double as integration tests and documentation.

**Rust Examples** (`examples/*.rs`):
- Built and tested automatically
- Demonstrate real-world usage patterns
- Validate API ergonomics

**Python Examples** (`python/examples/*.py`):
- Executable documentation
- End-to-end workflow validation
- Performance demonstrations

## Test Categories

### Functional Tests

Test core business logic:
- ✅ Location search accuracy
- ✅ Administrative hierarchy resolution
- ✅ Multi-term query handling
- ✅ Alternative name matching
- ✅ Batch processing correctness

### Configuration Tests

Test configuration system:
- ✅ Builder pattern validation
- ✅ Preset configurations
- ✅ Custom weight validation
- ✅ Parameter range checking
- ✅ Configuration inheritance

### Error Handling Tests  

Test robustness:
- ✅ Empty/invalid inputs
- ✅ Network failures (data download)
- ✅ Malformed data handling
- ✅ Resource exhaustion
- ✅ Concurrent access safety

### Performance Tests

Test efficiency (with test data):
- ✅ Search latency
- ✅ Batch vs individual performance
- ✅ Memory usage patterns
- ✅ Index build time
- ✅ Concurrent throughput

### Compatibility Tests

Test cross-platform behavior:
- ✅ Different operating systems
- ✅ Python version compatibility  
- ✅ Rust version compatibility
- ✅ Architecture differences (x86/ARM)

## Running Tests

### Local Development

```bash
# Quick test with test data
just test

# Run only Rust tests
just rust-test

# Run only Python tests  
just pytest

# Run specific test
USE_TEST_DATA=true cargo test test_searcher_creation --features test_data

# Run with verbose output
USE_TEST_DATA=true cargo test --features test_data -- --nocapture
```

### Continuous Integration

The CI pipeline should:

1. **Fast Test Suite**: Run with `USE_TEST_DATA=true` for quick feedback
2. **Full Integration**: Periodically test with full data (nightly builds)
3. **Multi-Platform**: Test on Linux, macOS, Windows
4. **Multi-Language**: Test both Rust and Python bindings

Example CI configuration:
```yaml
- name: Run fast tests
  run: |
    export USE_TEST_DATA=true
    cargo test --features test_data
    python -m pytest python/tests/ -v

- name: Run full integration (nightly)
  if: github.event.schedule == '0 2 * * *'  # Nightly at 2 AM
  run: |
    # Test with full data - slower but more comprehensive
    cargo test --release
    python -m pytest python/tests/ -v --slow
```

## Data Dependencies

### Test Data Generation

Test data is generated from a curated subset of real GeoNames data:

```rust
// Controlled via features and environment
#[cfg(feature = "test_data")]
const USE_SMALL_DATASET: bool = true;

// Runtime detection
fn should_use_test_data() -> bool {
    std::env::var("USE_TEST_DATA").unwrap_or_default() == "true"
}
```

### Managing External Dependencies

**Challenge**: Tests depend on external GeoNames data
**Solutions**:

1. **Test Data Feature**: Use smaller embedded datasets
2. **Caching**: Cache downloaded data between test runs
3. **Mocking**: Mock network calls for pure unit tests
4. **CI Optimization**: Pre-download data in CI images

## Test Guidelines

### Writing Good Tests

1. **Use Test Data**: Always set `USE_TEST_DATA=true` in test setup
2. **Isolation**: Each test should be independent
3. **Descriptive Names**: Test names should describe the scenario
4. **Appropriate Scope**: Unit tests for logic, integration tests for workflows
5. **Error Cases**: Test both success and failure paths

### Test Data Limitations

Test data has limited coverage:
- ✅ Major cities (London, Paris, Tokyo, New York)
- ✅ Major countries (US, UK, France, Germany, Japan)
- ✅ Basic administrative hierarchies
- ❌ Comprehensive alternative names
- ❌ Minor places and obscure locations
- ❌ Full geographic coverage

### Performance Considerations

When writing performance tests:
- Use `criterion` for micro-benchmarks
- Test with realistic data sizes
- Consider batch vs individual operations
- Measure memory usage, not just time
- Account for index warmup time

## Debugging Test Failures

### Common Issues

1. **"No results found"**: Test data may not include the queried location
2. **"Index not found"**: Ensure `USE_TEST_DATA=true` is set
3. **"Download failed"**: Network issues or missing test data feature
4. **"Permission denied"**: File system permissions for cache directory

### Debugging Steps

1. **Check Environment**: Verify `USE_TEST_DATA=true` is set
2. **Check Features**: Ensure `--features test_data` is used
3. **Check Logs**: Enable debug logging with `RUST_LOG=debug`
4. **Isolate Test**: Run single test to isolate issue
5. **Clean State**: Clear cache and rebuild indexes

### Debug Commands

```bash
# Enable verbose logging
RUST_LOG=heisenberg=debug USE_TEST_DATA=true cargo test test_name

# Clear cache and restart
rm -rf hberg_data/
USE_TEST_DATA=true cargo test --features test_data

# Check test data availability
USE_TEST_DATA=true cargo test --features test_data test_searcher_creation -- --nocapture
```

## Future Improvements

1. **Property-Based Testing**: Use `quickcheck` for input fuzzing
2. **Snapshot Testing**: Capture and verify complex output structures  
3. **Regression Testing**: Maintain test cases for reported bugs
4. **Load Testing**: Test with realistic production loads
5. **Cross-Language Testing**: Ensure Rust/Python API parity

---

This testing strategy ensures reliable, fast development feedback while maintaining confidence in production behavior.