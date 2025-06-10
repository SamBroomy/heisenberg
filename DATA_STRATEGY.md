# Data Distribution Strategy for Heisenberg

This document outlines the current data strategy and evaluates different approaches for distributing the GeoNames data required by Heisenberg.

## Current Approach: Download on First Use

### How It Works

Heisenberg currently uses a **download-on-first-use** strategy:

1. **First Run**: Downloads ~500MB of GeoNames data from official sources
2. **Processing**: Converts raw data to optimized Parquet format (~200MB)
3. **Indexing**: Builds Tantivy full-text search indexes (~300MB)
4. **Caching**: Stores processed data locally for subsequent runs
5. **Updates**: Data is cached indefinitely (manual refresh required)

### Implementation Details

```rust
// Controlled by features in Cargo.toml
#[cfg(feature = "download_data")]
pub fn fetch_geonames_data() -> Result<(), DataError> {
    // Downloads from http://download.geonames.org/
    // - allCountries.zip (~370MB → ~1.5GB uncompressed)
    // - countryInfo.txt (~12KB)
    // - featureCodes_en.txt (~30KB)
}

// Data directory structure
hberg_data/
├── processed/
│   ├── admin_search.parquet      # ~100MB
│   └── place_search.parquet      # ~150MB
├── tantivy_indexes/
│   ├── admin_search/             # ~150MB
│   └── places_search/            # ~200MB
└── raw/                          # ~500MB (deleted after processing)
```

### Advantages ✅

- **Always Up-to-Date**: Downloads latest GeoNames data
- **Smaller Package Size**: No bundled data in crate/wheel
- **Flexible**: Can update data independently of code
- **Storage Efficient**: Only downloads what's needed
- **License Friendly**: No redistribution concerns

### Disadvantages ❌

- **Slow First Run**: 2-5 minute initialization
- **Network Dependency**: Requires internet on first use
- **Reliability Risk**: Dependent on GeoNames server availability
- **User Friction**: Unexpected delay for new users
- **Bandwidth Usage**: Large download for every new user/environment

## Alternative Strategies Evaluated

### 1. Ship Pre-processed Data with Package

Bundle processed Parquet files directly in the package.

**Implementation**:
```toml
# Would add ~300MB to package size
[package]
include = [
    "data/admin_search.parquet",
    "data/place_search.parquet"
]
```

**Pros**: Instant startup, no network dependency, reliable
**Cons**: Large package size, infrequent updates, storage duplication
**Verdict**: ❌ Too large for PyPI/crates.io

### 2. Separate Data Package

Create companion data packages (e.g., `heisenberg-data`).

**Implementation**:
```bash
# Separate installation
pip install heisenberg heisenberg-data
cargo add heisenberg heisenberg-data
```

**Pros**: Modular, user choice, faster startup
**Cons**: Complex installation, version sync issues, user confusion
**Verdict**: ❌ Too complex for most users

### 3. Content Delivery Network (CDN)

Host optimized data on CDN for faster downloads.

**Implementation**:
```rust
const CDN_BASE_URL: &str = "https://cdn.heisenberg.dev/data/v1/";
// Download pre-processed files instead of raw GeoNames
```

**Pros**: Faster downloads, better reliability, version control
**Cons**: Infrastructure costs, maintenance overhead, still requires download
**Verdict**: ✅ Good for future optimization

### 4. Progressive Data Loading

Download and process data incrementally by region/importance.

**Implementation**:
```rust
// Start with major cities/countries, expand as needed
pub fn load_essential_data() -> Result<LocationSearcher, Error> {
    // Load top 1000 cities first (~1MB)
    // Background download of full dataset
}
```

**Pros**: Fast initial startup, progressive enhancement
**Cons**: Complex implementation, uneven coverage, cache complexity
**Verdict**: 🤔 Interesting for v2.0

### 5. Hybrid Approach

Combine embedded essential data with on-demand full dataset.

**Implementation**:
```rust
// Embed minimal dataset for instant functionality
const EMBEDDED_ESSENTIAL_DATA: &[u8] = include_bytes!("essential.parquet");

// Download full dataset in background or on-demand
pub fn enable_comprehensive_search() -> Result<(), Error> {
    // Download full dataset when user needs it
}
```

**Pros**: Instant basic functionality, full power on-demand
**Cons**: Complex implementation, unclear upgrade path
**Verdict**: 🤔 Promising for future

## Recommendation: Current + Optimizations

**Keep the current download-on-first-use strategy** with these improvements:

### Short-term Optimizations (v0.1.x)

1. **Pre-processed CDN**: Host processed Parquet files on CDN
   - Reduces download size: 500MB → 200MB
   - Faster download speed: ~30 seconds instead of 2-5 minutes
   - Better reliability than GeoNames servers

2. **Progress Indication**: Show download/processing progress
   ```rust
   pub fn new_with_progress<F>(callback: F) -> Result<LocationSearcher, Error>
   where F: Fn(ProgressUpdate)
   ```

3. **Parallel Processing**: Optimize data processing pipeline
   - Download and process in parallel
   - Use all CPU cores for data transformation

4. **Smart Caching**: Better cache management
   - Version checks against remote data
   - Incremental updates when possible
   - Cache validation and repair

### Medium-term Enhancements (v0.2.x)

1. **Essential Data Embedding**: 
   - Embed ~1MB of major cities/countries
   - Instant basic functionality
   - Full download for comprehensive search

2. **Regional Data Packages**:
   - Option to download specific regions
   - Useful for region-specific applications
   - Reduces download size and memory usage

3. **Data Versioning**:
   - Track GeoNames data versions
   - Automatic incremental updates
   - Rollback capability

### Long-term Vision (v1.0+)

1. **Distributed Data Network**:
   - P2P-style data sharing
   - Regional mirrors
   - Automatic failover

2. **Real-time Updates**:
   - Subscribe to GeoNames changes
   - Incremental data updates
   - Live index updates

## Current Testing Strategy

For testing and development, we use a **test data feature**:

```bash
# Use small embedded dataset for tests
USE_TEST_DATA=true cargo test --features test_data

# Test data is ~100 locations vs 11M+ in production
```

This allows:
- ✅ Fast CI/CD pipelines (< 10 seconds)
- ✅ Reliable offline development
- ✅ Consistent test environments
- ✅ No network dependencies in tests

## Migration Path

### Phase 1: CDN Optimization (Immediate)
- Set up CDN for processed data files
- Reduce download time by 80%
- Add progress indicators
- **Timeline**: 1-2 weeks

### Phase 2: Essential Data (v0.2)
- Embed 1MB essential dataset
- Instant basic functionality
- Background full download
- **Timeline**: 1-2 months

### Phase 3: Smart Loading (v0.3)
- Regional data packages
- Incremental updates
- Advanced caching
- **Timeline**: 3-6 months

## Decision Matrix

| Strategy | Startup Time | Package Size | Reliability | Complexity | Verdict |
|----------|--------------|--------------|-------------|------------|---------|
| Current | ❌ Slow | ✅ Small | ⚠️ Network | ✅ Simple | **Current** |
| Ship Data | ✅ Instant | ❌ Large | ✅ High | ✅ Simple | ❌ |
| Separate Package | ✅ Fast | ✅ Small | ✅ High | ❌ Complex | ❌ |
| CDN | ✅ Fast | ✅ Small | ✅ High | ⚠️ Medium | **Phase 1** |
| Progressive | ✅ Instant | ✅ Small | ⚠️ Complex | ❌ Complex | **Phase 2** |
| Hybrid | ✅ Instant | ⚠️ Medium | ✅ High | ❌ Complex | **Phase 3** |

## User Experience Goals

1. **First-time Users**: Should see functionality within 30 seconds
2. **Production Users**: Reliable, always-available data
3. **Developers**: Fast iteration with test data
4. **Enterprise**: Control over data sources and updates

The current strategy with planned optimizations best balances these competing requirements while maintaining simplicity and reliability.

---

**Status**: Current approach recommended with CDN optimization as immediate next step.
**Last Updated**: 2024-12-19
**Next Review**: Q1 2025