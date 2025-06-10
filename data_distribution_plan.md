# Heisenberg Data Distribution Strategy

## Problem Statement
- Full GeoNames dataset (allCountries.txt) is 391MB
- Library is unusable without data
- CI/testing needs reliable data access
- Users need offline functionality

## Proposed Solution: Hybrid Distribution

### Phase 1: Ship Curated Dataset (Immediate)
**Size:** ~2-5MB compressed (~10-15MB uncompressed)
**Contents:**
- 200-300 major countries/territories  
- 1000-2000 major cities (population > 100k)
- Major administrative divisions for key countries
- All feature codes and country info

**Benefits:**
- ✅ Works offline immediately
- ✅ Fast installation/CI
- ✅ Covers 80% of common use cases
- ✅ Same index/API as full dataset

### Phase 2: Progressive Enhancement (Future)
**API Extensions:**
```rust
// Current (uses curated data)
LocationSearcher::new(false)

// New options
LocationSearcher::new_with_full_data()      // Downloads 391MB full dataset
LocationSearcher::new_regional("US")        // Downloads region-specific data
LocationSearcher::new_cities_only()         // Downloads global cities only
```

## Implementation Plan

### Step 1: Create Curated Dataset
- Extract major countries, capitals, large cities from GeoNames
- Include proper administrative hierarchies
- Maintain all GeoName IDs for compatibility
- Store as compressed Parquet files in `data/embedded/`

### Step 2: Modify Data Loading Logic
```rust
// In src/data/mod.rs
pub enum DataSource {
    Embedded,     // Ships with package (~5MB)
    Cached,       // Previously downloaded full data
    Download,     // Download on demand
}

impl LocationSearchData {
    pub fn new_with_source(source: DataSource) -> Result<Self> {
        match source {
            DataSource::Embedded => load_embedded_data(),
            DataSource::Cached => load_cached_data().or_else(|_| load_embedded_data()),
            DataSource::Download => download_and_cache().or_else(|_| load_embedded_data()),
        }
    }
}
```

### Step 3: Backward Compatibility
- Current `LocationSearcher::new(false)` uses embedded data
- Current `LocationSearcher::new(true)` downloads full data (unchanged)
- All existing APIs work with any dataset size

## Curated Dataset Criteria

### Countries (200-300 entries)
- All UN member states
- Major territories and dependencies
- Countries with population > 1M

### Cities (1000-2000 entries)  
- All national capitals
- Cities with population > 100k
- Major tourist/business destinations
- Administrative centers

### Administrative Divisions
- First-level divisions for major countries (US states, Canadian provinces, etc.)
- Second-level for key regions (US counties, UK councils, etc.)

### Feature Codes
- All existing feature codes (small dataset)
- Complete coverage for proper classification

## Technical Implementation

### Storage Format
```
data/
├── embedded/           # Ships with package
│   ├── countries.parquet
│   ├── cities.parquet  
│   ├── admin.parquet
│   └── features.parquet
├── cached/             # Downloaded full data
│   └── full/
└── temp/               # Test data
```

### Size Optimization
- Parquet compression (~70% reduction)
- Remove unnecessary columns for embedded data
- Use smaller numeric types where possible
- Delta encoding for coordinates

### Index Compatibility
- Same Tantivy schema works with any dataset size
- GeoName IDs consistent across all dataset sizes
- Indexes rebuild automatically when data changes

## Benefits of This Approach

1. **Immediate Usability**: Library works out-of-the-box offline
2. **Progressive Enhancement**: Users can upgrade to full data when needed
3. **CI/Testing Friendly**: No external dependencies for basic functionality
4. **Backward Compatible**: Existing code continues to work
5. **Flexible**: Different dataset sizes for different use cases
6. **Reliable**: Fallback to embedded data if downloads fail

## Migration Path

### Phase 1 (Immediate)
- Create curated dataset from current GeoNames data
- Ship with next release
- Update documentation

### Phase 2 (Next release)
- Add new constructor methods
- Implement regional datasets
- Add data update mechanisms

### Phase 3 (Future)
- Auto-update embedded data
- Smart caching based on usage patterns
- Data validation and integrity checks

## Risk Mitigation

- **Data Staleness**: Update embedded data with each release
- **Size Creep**: Monitor embedded data size, maintain curation criteria
- **Regional Bias**: Include diverse global coverage in curation
- **Feature Parity**: Ensure embedded data covers common search patterns

This approach solves the core distribution problem while maintaining flexibility for power users who need the complete dataset.