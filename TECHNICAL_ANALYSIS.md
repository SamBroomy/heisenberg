# Deep Technical Analysis of Heisenberg Search and Backfill System

## System Overview

Heisenberg implements a sophisticated geographic location search and resolution system that combines full-text search with hierarchical administrative context resolution. The architecture transforms unstructured location inputs into complete geographic hierarchies through multi-phase search coordination.

## 1. Core Architecture (`src/core.rs`)

### LocationSearcher - The Central Orchestrator

The `LocationSearcher` coordinates all search operations through a layered architecture:

**Key Components:**

- `admin_fts_index: FTSIndex<AdminIndexDef>` - Tantivy-based administrative entity index
- `places_fts_index: FTSIndex<PlacesIndexDef>` - Tantivy-based places/POI index
- `data: LocationSearchData` - Cached GeoNames datasets in Polars DataFrames

**Architectural Layers:**

1. **Public API**: `search()`, `resolve_location()`, `resolve_location_batch()`
2. **Orchestration**: Multi-phase search coordination in `search_orchestration.rs`
3. **Specialized Search**: Administrative and place search modules
4. **Index Layer**: Tantivy FTS with custom configurations
5. **Data Layer**: Polars LazyFrames for efficient query processing

## 2. Search Orchestration (`src/search/search_orchestration.rs`)

### Intelligent Multi-Phase Search Strategy

The orchestration implements a sophisticated 4-phase pipeline:

#### Phase 1: Term Analysis and Preparation

```rust
fn prepare_search_terms(search_terms_raw: &[&str], max_sequential_admin_terms: usize)
-> Result<(Vec<String>, Option<String>, bool)>
```

**Algorithm:**

1. **Term Cleaning**: Removes empty/whitespace terms
2. **Admin Sequence Extraction**: First N terms treated as hierarchical admin candidates
3. **Place Candidate Identification**: Remaining/last term as potential place
4. **Overlap Detection**: Tracks when place candidate is also final admin term

#### Phase 2: Administrative Sequence Processing

**Hierarchical Context Building:**

```rust
let effective_start_level = match min_level_from_last_success {
    Some(min_prev) => std::cmp::max(natural_start_level, min_prev + 1),
    None => natural_start_level,
};
```

**Key Innovation - Dynamic Level Calculation:**

- Processes admin terms sequentially (country → state → county → local)
- Uses successful matches to constrain subsequent search levels
- Maintains hierarchical consistency (if "California" matches level 1, next term searches level 2+)
- Implements parent-child relationship validation

#### Phase 3: Proactive Administrative Search

Optional phase attempting to match place candidates as administrative entities before treating as places (handles "London" as London Borough vs. London city).

#### Phase 4: Place Search

Final phase searches places dataset using accumulated administrative context for geographic filtering and relevance boosting.

### Parallel Processing and Batching

**Bulk Operation Optimization:**

```rust
pub fn bulk_location_search_inner() -> Result<Vec<Vec<SearchResult>>>
```

**Deduplication Strategy:**

- Groups identical queries to avoid redundant processing
- Maps original indices to unique queries for result reconstruction
- Executes unique queries once, distributes results to original positions

**Parallel Execution:**

- Batches similar searches by term patterns and admin context
- Uses Rayon for parallel batch processing
- Implements progressive data filtering to narrow search space

## 3. Administrative Search (`src/search/admin_search.rs`)

### Hierarchical Administrative Entity Search

Handles countries (level 0) through local divisions (level 4) with sophisticated scoring:

#### Multi-Factor Scoring Algorithm

```rust
fn search_score_admin() -> Result<LazyFrame>
```

**Weighted Scoring Components:**

1. **Text Relevance (40%)**: FTS + fuzzy matching scores
2. **Population Importance (25%)**: Logarithmic scaling favoring larger entities
3. **Parent Context (20%)**: Boost from matching parent entities
4. **Feature Type Importance (15%)**: Countries > capitals > states > counties

**Feature Type Hierarchy:**

```rust
when(col("feature_code").eq(lit("PCLI")))     // Independent country: 1.0
when(col("feature_code").eq(lit("PPLC")))     // Capital city: 0.95
when(col("feature_code").str().starts_with(lit("ADM1")))  // State: 0.85
```

#### Context-Aware Filtering

```rust
fn filter_data_from_previous_results(
    data: LazyFrame,
    previous_result_df: LazyFrame,
    join_key_exprs: &[Expr]
) -> Result<LazyFrame>
```

Creates precise hierarchical filtering where "Los Angeles" after "California" only searches California's administrative boundaries.

### Parent Score Propagation

```rust
fn parent_factor(lf: LazyFrame) -> Result<LazyFrame>
```

**Sophisticated Parent Influence:**

1. Identifies parent score columns: `parent_score_admin_[0-4]`
2. Calculates mean of available parent scores
3. Normalizes to 0-1 range for consistent weighting
4. Boosts entities with highly-scored parents

## 4. Place Search (`src/search/place_search.rs`)

### Points of Interest and Location Search

Handles cities, landmarks, POIs, and geographic features:

#### Importance-Based Filtering

```rust
let data_filtered_by_tier = data.filter(
    col("importance_tier").lt_eq(lit(params.min_importance_tier.clamp(1, 5)))
);
```

Places pre-classified into importance tiers (1=highest, 5=lowest) for efficient filtering.

#### Geographic Distance Scoring

```rust
// Haversine distance calculation
let distance_km = lit(2.0_f64) * a.sqrt().arcsin() * lit(EARTH_RADIUS_KM);
```

Uses center coordinates from admin context for distance-based relevance scoring.

#### Feature Type Classification

```rust
when(col("feature_code").is_in(lit(Series::new("major_capitals", &["PPLC", "PPLA"]))))
    .then(lit(1.0_f32))  // Capitals: highest score
when(col("feature_code").is_in(lit(Series::new("landmarks", &["CSTL", "MNMT"]))))
    .then(lit(0.95_f32)) // Historic landmarks
```

Ensures capitals rank higher than towns, landmarks higher than generic features.

## 5. Full-Text Search Index (`src/index/mod.rs`)

### Tantivy-Based Search Infrastructure

#### Dual Index Architecture

- **AdminIndexDef**: Administrative entities with country codes (ISO, FIPS)
- **PlacesIndexDef**: Lightweight place names and alternates

#### Advanced Query Construction

```rust
fn build_base_query() -> Result<Box<dyn Query>>
```

**Multi-Strategy Querying:**

1. **Standard Text Search**: Multi-field with custom weights
2. **Fuzzy Matching**: Typo tolerance for longer terms
3. **Exact Code Matching**: High-boost for country codes ("USA", "GB")
4. **Document Subset Filtering**: Constrains to specific document sets

#### Field Weight Optimization

```rust
vec![
    (schema.get_field("official_name").unwrap(), 4.0),
    (schema.get_field("name").unwrap(), 3.0),
    (schema.get_field("asciiname").unwrap(), 2.0),
    (schema.get_field("alternatenames").unwrap(), 1.0),
]
```

Official names receive highest weight, decreasing through primary names, ASCII variants, to alternates.

## 6. Backfill and Resolution (`src/backfill/`)

### Complete Location Context Assembly

Transforms raw search results into complete hierarchical location contexts:

#### Target Location Code Extraction

```rust
pub struct TargetLocationAdminCodes {
    pub admin0_code: Option<String>,  // Country
    pub admin1_code: Option<String>,  // State/Province
    pub admin2_code: Option<String>,  // County/Region
    pub admin3_code: Option<String>,  // Local admin
    pub admin4_code: Option<String>,  // Sub-local admin
}
```

#### Hierarchical Context Building

```rust
fn backfill_administrative_context<E: LocationEntry>(
    target_codes: &[TargetLocationAdminCodes],
    admin_data_lf: &LazyFrame,
) -> Result<Vec<LocationContext<E>>>
```

**Resolution Algorithm:**

1. **Code Hierarchy Traversal**: Builds filters combining parent codes for each level
2. **Cumulative Filtering**: `admin0_code = "US" AND admin1_code = "CA" AND admin_level = 2`
3. **Parallel Collection**: Uses `collect_all()` for efficient admin entity gathering
4. **Context Assembly**: Reconstructs complete location hierarchies

#### Flexible Entity Type System

Supports customizable entity types through `LocationEntry` trait:

- **BasicEntry**: Minimal (geoname_id, name)
- **GenericEntry**: Rich (admin codes, coordinates, population, feature codes)
- **Custom Types**: User-implementable entry types

## 7. Scoring Algorithms and Ranking

### Text Relevance Scoring (`src/search/common.rs`)

**Hybrid Text Scoring:**

```rust
fn text_relevance_score(lf: LazyFrame, search_term: &str) -> LazyFrame
```

**Multi-Signal Approach:**

1. **FTS Contribution**: Z-score normalized Tantivy relevance
2. **Fuzzy Contribution**: Character-level similarity (RapidFuzz)
3. **Combined Score**: `0.3 * fts_score + 0.7 * fuzzy_score`

Fuzzy scoring catches alternate names and variations FTS might miss.

### Advanced Parent Factor Calculation

```rust
fn parent_factor(lf: LazyFrame) -> Result<LazyFrame>
```

**Sophisticated Context Propagation:**

1. Identifies parent score columns via regex pattern
2. Calculates horizontal mean across available parent scores
3. Normalizes relative to dataset maximum parent score
4. Provides 0.5 default when no parent context available

## 8. Data Flow and Architectural Decisions

### Complete Search Pipeline

```
Input Terms → Term Analysis → Admin Sequence → Proactive Admin → Place Search → Context Resolution → Final Results
```

### Key Architectural Decisions

1. **Lazy Evaluation**: Extensive Polars LazyFrames for efficient query planning
2. **Parallel Processing**: Rayon-based parallelism for bulk operations
3. **Context Propagation**: Results flow forward as context for subsequent searches
4. **Type Safety**: Strong typing with generics and wrapper types (AdminFrame, PlaceFrame)
5. **Hierarchical Validation**: Ensures administrative hierarchy consistency

### Performance Optimizations

1. **Index Persistence**: Tantivy indexes cached between runs
2. **Query Deduplication**: Identical queries processed once in bulk
3. **Progressive Filtering**: Admin data filtered by parent relationships
4. **Subset Search**: FTS limited to relevant document subsets vs. full corpus

### Error Handling and Robustness

- Graceful degradation on search failures
- Administrative hierarchy consistency validation
- Fallback scoring when context unavailable
- Comprehensive logging and instrumentation

## Conclusion

Heisenberg implements a sophisticated geographic location search system that transcends simple text matching through hierarchical, context-aware search strategies. Its strength lies in understanding geographic entity relationships and progressively refining searches using accumulated context to deliver complete location hierarchies rather than isolated name matches.

The system's multi-phase architecture, advanced scoring algorithms, and efficient parallel processing create a robust foundation capable of handling complex, ambiguous location queries with high accuracy and performance.
