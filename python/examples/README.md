# Heisenberg Python Examples

This directory contains comprehensive examples demonstrating the various features and capabilities of the Heisenberg location search library.

## Examples Overview

### 1. Basic Usage (`01_basic_usage.py`)
- Creating a LocationSearcher
- Simple location searches
- Working with search results
- Using convenience functions
- Accessing result attributes and methods

**Run:** `python 01_basic_usage.py`

### 2. Configuration (`02_configuration.py`)
- Using preset configurations (fast, comprehensive, quality places)
- Building custom configurations
- Location bias (geographical preferences)
- Advanced configuration with custom weights
- SearchOptions validation

**Run:** `python 02_configuration.py`

### 3. Batch Processing (`03_batch_processing.py`)
- Processing multiple queries efficiently
- Multi-term batch queries
- Performance comparison between individual and batch processing
- Handling mixed quality queries
- Batch processing with configurations

**Run:** `python 03_batch_processing.py`

### 4. Resolution and Backfill (`04_resolution_backfill.py`)
- Location resolution to complete administrative hierarchies
- Multi-term resolution for better accuracy
- Batch resolution processing
- Working with different entry types
- Using resolved results and administrative levels

**Run:** `python 04_resolution_backfill.py`

### 5. Advanced Features (`05_advanced_features.py`)
- Direct Rust API access
- Low-level search methods (admin_search, place_search)
- Advanced configuration patterns
- SearchResult methods and attributes
- Error handling and edge cases
- Multiple entry types

**Run:** `python 05_advanced_features.py`

## Prerequisites

1. **Install the library:**
   ```bash
   pip install heisenberg
   ```

   Or for development:
   ```bash
   maturin develop --features python
   ```

2. **Data setup:**
   The examples use test data for faster execution. In production, remove the `USE_TEST_DATA` environment variable to use the full GeoNames dataset.

## Key Concepts

### Search vs Resolution
- **Search**: Find locations matching your query terms
- **Resolution**: Enrich results with complete administrative hierarchy (country → state → county → place)

### Configuration Options
- **Fast**: Optimized for speed, fewer results
- **Comprehensive**: Optimized for accuracy, more results
- **Quality Places**: Focus on important places and landmarks
- **Custom**: Build your own configuration with specific parameters

### Entry Types
- **LocationEntry**: Complete location data (coordinates, population, admin codes, etc.) - use only the fields you need

### Administrative Levels
- **Admin0**: Country level
- **Admin1**: State/Province level
- **Admin2**: County/Department level
- **Admin3**: Local administrative division
- **Admin4**: Sub-local administrative division

## Performance Tips

1. **Use batch processing** for multiple queries
2. **Choose appropriate configurations** based on your use case
3. **Cache the searcher instance** - initialization can take time on first run
4. **Use test data during development** for faster iteration

## Common Use Cases

### Finding Cities
```python
results = searcher.find("Tokyo")
```

### Location with Context
```python
results = searcher.find(["Paris", "France"])
```

### Batch Processing
```python
queries = [["London"], ["Berlin"], ["Madrid"]]
results = searcher.find_batch(queries)
```

### Complete Administrative Hierarchy
```python
resolved = rust_searcher.resolve_location(["San Francisco", "California"])
# Access: country, state, county, place
```

### Custom Search Behavior
```python
config = SearchConfigBuilder().limit(5).place_importance(2).build()
results = searcher.find("Cambridge", config)
```

## Error Handling

The library is designed to be robust:
- Empty queries return empty results (no exceptions)
- Invalid locations return empty results
- Configuration validation provides helpful error messages
- Batch processing handles mixed valid/invalid queries gracefully

## Getting Help

- Check the test files in `../tests/` for additional examples
- Read the API documentation: `cargo doc --open`
- See the main README.md for installation and setup instructions
