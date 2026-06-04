# Heisenberg Python Package

**Fast location enrichment for Python using GeoNames dataset**

[![PyPI](https://img.shields.io/pypi/v/heisenberg)](https://pypi.org/project/heisenberg/)
[![Python Versions](https://img.shields.io/pypi/pyversions/heisenberg)](https://pypi.org/project/heisenberg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Heisenberg transforms incomplete location data into complete administrative hierarchies. It resolves ambiguous place names, fills missing administrative context, and handles alternative names across 11+ million global locations.

## Key Features

- **Fast**: Powered by Rust with embedded dataset - no downloads required
- **Comprehensive**: 11+ million locations from GeoNames dataset
- **Smart Search**: Handles typos, alternative names, and ambiguous queries
- **Complete Hierarchies**: Returns full administrative context (country → state → county → place)
- **Batch Processing**: Optimized for high-throughput applications
- **Pythonic**: Clean, well-typed API with comprehensive documentation

## Installation

```bash
pip install heisenberg
```

## Quick Start

### Basic Usage

```python
import heisenberg

# Create searcher (uses embedded data - no setup required!)
searcher = heisenberg.LocationSearcher()

# Simple search
results = searcher.find("Tokyo")
print(f"Found: {results[0].name}")  # Tokyo

# Multi-term search with context (largest to smallest location)
results = searcher.find(["France", "Paris"])
print(f"Found: {results[0].full_name()}")  # Paris, Île-de-France, France
```

### Location Enrichment

The core purpose is enriching incomplete/messy location data:

**Important**: Provide location terms in descending 'size' order (Country → State → County → Place) for optimal results.

```python
# Input: incomplete data (largest to smallest order)
incomplete_locations = [
    ["Florida", "Lakeland"],           # Missing country, county
    ["CA", "San Francisco"],           # Abbreviated state
    ["Deutschland"],                   # Alternative name
    ["London"],                        # Ambiguous (which London?)
]

# Output: complete administrative hierarchies
for location in incomplete_locations:
    results = searcher.find(location)
    if results:
        result = results[0]
        print(f"Input: {location}")
        print(f"Output: {result.full_name()}")
        print(f"Hierarchy: {' → '.join(result.admin_hierarchy())}")
        print()

# Results:
# Input: ['Florida', 'Lakeland']
# Output: Lakeland, Florida, United States
# Hierarchy: United States → Florida → Polk County

# Input: ['CA', 'San Francisco']
# Output: San Francisco, California, United States
# Hierarchy: United States → California → San Francisco County

# Input: ['Deutschland']
# Output: Germany
# Hierarchy: Germany

# Input: ['London']
# Output: London, England, United Kingdom
# Hierarchy: United Kingdom → England → Greater London
```

### Batch Processing

For high-throughput applications:

```python
# Process multiple locations efficiently (largest to smallest order)
queries = [
    ["Japan", "Tokyo"],
    ["United Kingdom", "London"],
    ["United States", "New York"],
    ["Germany", "Berlin"]
]

batch_results = searcher.find_batch(queries)

for i, results in enumerate(batch_results):
    if results:
        print(f"{queries[i]} → {results[0].full_name()}")
```

### Advanced Configuration

```python
# Fast search (speed optimized)
fast_results = searcher.find_quick("Berlin")

# Comprehensive search (accuracy optimized)
comprehensive_results = searcher.find_comprehensive("Cambridge")

# Custom configuration
config = heisenberg.SearchConfigBuilder() \
    .limit(5) \
    .place_importance(2) \
    .location_bias(40.7128, -74.0060) \
    .build()

results = searcher.find("Springfield", config)
```

### Working with Results

```python
results = searcher.find(["California", "San Francisco"])
result = results[0]

# Basic information
print(f"Name: {result.name}")
print(f"GeoNames ID: {result.geoname_id}")
print(f"Feature Code: {result.feature_code}")
print(f"Confidence Score: {result.score:.3f}")

# Geographic data
if result.latitude and result.longitude:
    print(f"Coordinates: ({result.latitude}, {result.longitude})")

if result.population:
    print(f"Population: {result.population:,}")

# Administrative hierarchy
hierarchy = result.admin_hierarchy()
print(f"Administrative context: {' → '.join(hierarchy)}")

# Full name with context
print(f"Full name: {result.full_name()}")

# Export to dictionary
data = result.to_dict()
```

## Data Sources

Choose different datasets based on your needs:

```python
# Embedded dataset (default - fastest startup)
searcher = heisenberg.LocationSearcher()

# Or specify data source
ds = heisenberg.DataSource.cities5000()  # Cities with pop > 5,000
searcher = heisenberg.LocationSearcher.with_data_source(ds)

# Available data sources:
# - cities15000() - Cities with pop > 15,000 (default, embedded)
# - cities5000()  - Cities with pop > 5,000
# - cities1000()  - Cities with pop > 1,000
# - cities500()   - Cities with pop > 500
# - all_countries() - Complete GeoNames dataset (~1GB)
```

## Use Cases

- **Data Cleaning**: Standardize messy location data in datasets
- **Address Validation**: Resolve and validate location components
- **Geocoding Preprocessing**: Prepare data for coordinate lookup
- **Business Intelligence**: Enrich location data for analytics
- **Import/ETL Pipelines**: Clean location data during data ingestion

## Performance

- **Startup**: Instant (embedded data, no downloads)
- **Search Speed**: ~1ms per query
- **Batch Processing**: 10-100x faster than individual queries
- **Memory Usage**: ~200MB RAM
- **Storage**: ~25MB embedded data + indexes

## API Reference

### Classes

- **`LocationSearcher`**: Main search interface
- **`DataSource`**: Data source selection
- **`LocationSearcherBuilder`**: Advanced searcher configuration
- **`SearchOptions`** & **`SearchConfigBuilder`**: Search configuration
- **`SearchResult`**: Search result with location data

### Functions

- **`find_location(query)`**: Convenience function for single searches
- **`find_locations_batch(queries)`**: Convenience function for batch processing

## Requirements

- Python 3.10+
- polars>=1.0

## Examples

See the [examples/](examples/) directory for complete examples including:

- Basic usage and configuration
- Batch processing patterns
- Advanced search features
- Error handling and edge cases

## Links

- **Documentation**: [GitHub Repository](https://github.com/SamBroomy/heisenberg)
- **Rust Crate**: [crates.io/crates/heisenberg](https://crates.io/crates/heisenberg)
- **Issues**: [GitHub Issues](https://github.com/SamBroomy/heisenberg/issues)
- **GeoNames Database**: [geonames.org](http://www.geonames.org/)

## License

MIT License - see [LICENSE](../LICENSE) for details.
