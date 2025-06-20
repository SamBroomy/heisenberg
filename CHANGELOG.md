# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public release preparation
- Comprehensive CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality
- Separate Python package README
- Publication-ready metadata for both Rust crate and Python package

## [0.1.0] - 2024-XX-XX

### Added

- **Core Functionality**
  - Fast location search using Tantivy full-text search engine
  - Complete administrative hierarchy resolution (country → state → county → place)
  - Support for multiple GeoNames datasets (cities15000, cities5000, cities1000, cities500, all_countries)
  - Embedded dataset support for instant startup (no downloads required)
  - Smart fallback system for data source loading
  - Alternative name resolution (e.g., "Deutschland" → "Germany")

- **Search Features**
  - Fuzzy string matching for typos and variations
  - Configurable search behavior and scoring weights
  - Location bias for disambiguating place names
  - Batch processing for high-throughput applications
  - Multiple search presets (fast, comprehensive, quality_places)

- **APIs**
  - **Rust API**: Complete native Rust interface with simplified, non-generic API
  - **Python API**: Clean Python wrappers with user-friendly interface
  - Raw Rust bindings accessible via `_internal` for debugging/testing
  - Comprehensive configuration builders for both APIs
  - Type-safe data source selection
  - Unified `LocationEntry` type eliminates complex trait system

- **Data Processing**
  - Automatic GeoNames data download and processing
  - Efficient parquet-based storage format
  - Tantivy search index generation
  - Build-time embedding of default dataset
  - Caching system for processed data

- **Performance Optimizations**
  - ~1ms search latency per query
  - 10-100x speedup for batch processing vs individual queries
  - ~200MB memory usage for default dataset
  - Embedded data eliminates startup time

- **Developer Experience**
  - Comprehensive examples for both Rust and Python
  - Extensive documentation and API reference
  - 43 automated tests covering all major functionality
  - Integration with Polars DataFrames for data manipulation
  - Error handling with detailed error messages

### Technical Details

- Built with Rust 2024 edition (MSRV: 1.89)
- Python support for 3.10+
- Uses GeoNames dataset for location data
- Tantivy for full-text search indexing
- Polars for efficient data processing
- PyO3 for Python bindings
- Supports Linux, macOS, and Windows

[Unreleased]: https://github.com/SamBroomy/heisenberg/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SamBroomy/heisenberg/releases/tag/v0.1.0
