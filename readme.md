# Heisenberg

**Location enrichment library for converting unstructured location data into structured administrative hierarchies.**

[![Crates.io](https://img.shields.io/crates/v/heisenberg)](https://crates.io/crates/heisenberg)
[![PyPI](https://img.shields.io/pypi/v/heisenberg)](https://pypi.org/project/heisenberg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://docs.rs/heisenberg/badge.svg)](https://docs.rs/heisenberg)

Heisenberg transforms incomplete location data into complete administrative hierarchies using the GeoNames dataset. It resolves ambiguous place names, fills missing administrative context, and handles alternative names across 11+ million global locations.

## Features

- **Embedded dataset**: Ships with data included, no downloads required
- Fast full-text search with Tantivy indexing
- Complete administrative hierarchy resolution (country → state → county → place)
- Multiple data sources (cities15000, cities5000, etc.) with smart fallback
- Batch processing for high-throughput applications
- Python and Rust APIs
- Configurable search behavior and scoring
- Alternative name resolution (e.g., "Deutschland" → "Germany")

## Quick Start

### Python

```bash
pip install heisenberg
```

```python
import heisenberg

# Create searcher instance
searcher = heisenberg.LocationSearcher()

# Simple search
results = searcher.find("Tokyo")
print(f"Found: {results[0].name}")

# Multi-term search
results = searcher.find(["Paris", "France"])
print(f"Found: {results[0].name}")

# Resolve complete administrative hierarchy
resolved = searcher.resolve_location(["San Francisco", "California"])
context = resolved[0].context

print(f"Country: {context.admin0.name}")  # United States
print(f"State: {context.admin1.name}")    # California
print(f"County: {context.admin2.name}")   # San Francisco County
print(f"City: {context.place.name}")      # San Francisco
```

### Rust

```toml
[dependencies]
heisenberg = "0.1"
```

```rust
use heisenberg::{LocationSearcher, DataSource, GenericEntry};

// Create searcher using embedded data (fastest, no downloads)
let searcher = LocationSearcher::new_embedded()?;

// Or use specific data source with smart fallback
let searcher = LocationSearcher::initialize(DataSource::Cities15000)?;

// Simple search
let results = searcher.search(&["Tokyo"])?;
println!("Found: {}", results[0].name().unwrap_or("Unknown"));

// Resolve complete hierarchy
let resolved = searcher.resolve_location::<_, GenericEntry>(&["Berlin", "Germany"])?;
let context = &resolved[0].context;

if let Some(country) = &context.admin0 {
    println!("Country: {}", country.name());
}
if let Some(place) = &context.place {
    println!("City: {}", place.name());
}
```

## Examples

The problem: inconsistent and incomplete location data.

| Input | Output |
|-------|--------|
| `"Florida"` | United States → Florida |
| `["Paris", "France"]` | France → Île-de-France → Paris |
| `["San Francisco", "CA"]` | United States → California → San Francisco County → San Francisco |
| `"Deutschland"` | Germany (resolves alternative names) |

### Administrative Levels

- **Admin0**: Countries
- **Admin1**: States/Provinces
- **Admin2**: Counties/Regions
- **Admin3**: Local administrative divisions
- **Admin4**: Sub-local administrative divisions
- **Places**: Cities, towns, landmarks

## Usage Examples

### Batch Processing

```python
queries = [["Tokyo", "Japan"], ["London", "UK"], ["New York", "USA"]]
batch_results = searcher.find_batch(queries)
```

### Configuration

```python
# Fast search (fewer results, optimized for speed)
config = heisenberg.SearchConfigBuilder.fast().build()
results = searcher.find("Berlin", config)

# Comprehensive search (more results, higher accuracy)
config = heisenberg.SearchConfigBuilder.comprehensive().build()
results = searcher.find("Cambridge", config)
```

See [examples/](examples/) for complete Rust examples and [python/examples/](python/examples/) for Python examples.

## Installation

### Python

```bash
pip install heisenberg
```

### Rust

```toml
[dependencies]
heisenberg = "0.1"
```

## Data

**Embedded by Default**: Heisenberg ships with the Cities15000 dataset embedded (~25MB compressed), providing instant startup with no downloads required.

**Multiple Data Sources**: Choose from different datasets based on your needs:
- `Cities15000`: Cities with population > 15,000 (default, embedded)
- `Cities5000`: Cities with population > 5,000
- `Cities1000`: Cities with population > 1,000
- `Cities500`: Cities with population > 500
- `AllCountries`: Complete GeoNames dataset (~1GB)

**Smart Fallback**: When requesting non-embedded datasets, Heisenberg automatically downloads and processes data on first use, then caches locally.

**Development**:
```bash
# Use embedded test data for development
USE_TEST_DATA=true cargo test

# Force regeneration of embedded data at build time
GENERATE_EMBEDDED_DATA=1 cargo build

# Use specific data source
EMBEDDED_DATA_SOURCE=cities5000 cargo build
```

## Performance

- **Instant startup**: Using embedded data (no download/processing time)
- Search: ~1ms per query
- Batch processing: 10-100x faster than individual queries
- Memory: ~200MB RAM
- Storage: ~25MB embedded + indexes, or ~1GB for larger datasets

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://docs.rs/heisenberg)
- [Contributing](CONTRIBUTING.md)
- [GeoNames Database](http://www.geonames.org/)
