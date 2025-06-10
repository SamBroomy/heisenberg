//! Heisenberg - Location Search and Enrichment Library
//!
//! Heisenberg is a powerful location enrichment library that transforms unstructured location data
//! into structured, complete administrative hierarchies. It uses the comprehensive GeoNames dataset
//! and advanced full-text search to provide fast, accurate location resolution.
//!
//! # Quick Start
//!
//! ```rust
//! use heisenberg::{GenericEntry, LocationEntryCore, LocationSearcher};
//!
//! // Create a searcher (downloads data on first run)
//! let searcher = LocationSearcher::new(false)?;
//!
//! // Simple search
//! let results = searcher.search(&["Tokyo"])?;
//! if let Some(result) = results.first() {
//!     println!("Found: {}", result.name().unwrap_or("Unknown"));
//! }
//!
//! // Resolve complete hierarchy
//! let resolved = searcher.resolve_location::<_, GenericEntry>(&["Berlin", "Germany"])?;
//! if let Some(result) = resolved.first() {
//!     let context = &result.context;
//!     if let Some(country) = &context.admin0 {
//!         println!("Country: {}", country.name());
//!     }
//!     if let Some(place) = &context.place {
//!         println!("City: {}", place.name());
//!     }
//! }
//! # Ok::<(), heisenberg::error::HeisenbergError>(())
//! ```
//!
//! # Features
//!
//! - **Intelligent Search**: Advanced text search with fuzzy matching and scoring
//! - **Structure Resolution**: Convert partial location data into complete administrative hierarchies
//! - **High Performance**: Built in Rust with Tantivy full-text search for speed
//! - **Global Coverage**: Based on GeoNames dataset with 11+ million locations
//! - **Flexible Configuration**: Customize search behavior for your specific needs
//! - **Batch Processing**: Efficiently process thousands of locations at once
//!
//! # Data
//!
//! Heisenberg ships with embedded geographic data (cities with population > 15,000)
//! that is processed at build time. This ensures the library works out of the box
//! without requiring external downloads or configuration.
#![feature(once_cell_try)]
use once_cell::sync::OnceCell;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

mod backfill;
mod config;
mod core;
mod data;
pub mod error;
mod index;
mod search;

// Re-export data processing from subcrate
pub use heisenberg_data_processing as data_processing;

pub extern crate polars;
pub use core::{LocationSearcher, ResolveSearchConfig};

pub use backfill::{
    BasicEntry, GenericEntry, LocationContext, LocationEntry, LocationEntryCore, ResolveConfig,
    ResolvedSearchResult,
};
pub use config::SearchConfigBuilder;
pub use index::FTSIndexSearchParams;
pub use search::{
    AdminFrame, AdminSearchParams, PlaceFrame, PlaceSearchParams, SearchConfig, SearchResult,
    SearchScoreAdminParams, SearchScorePlaceParams,
};

#[cfg(feature = "python")]
pub mod python;

static PLACES_DF_CACHE: OnceCell<()> = OnceCell::new();

/// Initialize logging for the Heisenberg library.
///
/// This sets up structured logging with configurable levels and filtering.
/// Call this once at the start of your application to enable detailed
/// logging output from Heisenberg operations.
///
/// # Arguments
///
/// * `level` - The minimum log level to display
///
/// # Examples
///
/// ```rust
/// use heisenberg::init_logging;
/// use tracing::Level;
///
/// // Initialize with info-level logging
/// init_logging(Level::INFO)?;
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
pub fn init_logging(level: impl Into<LevelFilter>) -> Result<&'static (), error::HeisenbergError> {
    PLACES_DF_CACHE.get_or_try_init(|| {
        let filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(level.into().to_string()))?
            .add_directive("tantivy=warn".parse().unwrap())
            .add_directive("hyper_util=warn".parse().unwrap());

        tracing_subscriber::fmt::fmt()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::CLOSE)
            .init();
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_env() {
        let _ = init_logging(tracing::Level::WARN);
    }

    #[test]
    fn test_searcher_creation() {
        setup_test_env();

        let searcher = LocationSearcher::new(false);
        assert!(
            searcher.is_ok(),
            "Should be able to create searcher with test data"
        );
    }

    #[test]
    fn test_basic_search() {
        setup_test_env();

        let searcher = LocationSearcher::new(false).unwrap();

        // Try a few different search terms that should exist in cities15000
        let test_terms = vec!["New York", "London", "Tokyo", "Berlin", "Paris"];

        for term in test_terms {
            let results = searcher.search(&[term]);
            assert!(results.is_ok(), "Basic search for '{term}' should work");
            let results = results.unwrap();
            if !results.is_empty() {
                println!("Found {} results for '{}'", results.len(), term);
                return; // Test passes if we find any results for any term
            }
        }

        panic!("Should find results for at least one major city");
    }

    #[test]
    fn test_multi_term_search() {
        setup_test_env();

        let searcher = LocationSearcher::new(false).unwrap();
        let results = searcher.search(&["New York", "USA"]);

        assert!(results.is_ok(), "Multi-term search should work");
        let results = results.unwrap();
        // Results may be empty if search is too specific, but shouldn't error
    }

    #[test]
    fn test_resolution() {
        setup_test_env();

        let searcher = LocationSearcher::new(false).unwrap();
        let resolved = searcher.resolve_location::<_, GenericEntry>(&["London"]);

        assert!(resolved.is_ok(), "Resolution should work");
        // Resolution may be empty with embedded data, but shouldn't error
    }

    #[test]
    fn test_batch_search() {
        setup_test_env();

        let searcher = LocationSearcher::new(false).unwrap();
        let queries = vec![vec!["London"], vec!["Paris"], vec!["Tokyo"]];
        let results = searcher.search_bulk(&queries);

        assert!(results.is_ok(), "Batch search should work");
        let results = results.unwrap();
        assert_eq!(results.len(), 3, "Should have results for all 3 queries");
    }

    #[test]
    fn test_configuration() {
        setup_test_env();

        // Test that configuration builder works
        let config = SearchConfigBuilder::fast()
            .limit(5)
            .place_importance_threshold(3)
            .build();

        assert_eq!(config.limit, 5);
        assert_eq!(config.place_min_importance_tier, 3);

        // Test search with configuration
        let searcher = LocationSearcher::new(false).unwrap();
        let results = searcher.search_with_config(&["London"], &config);

        assert!(results.is_ok(), "Search with config should work");
        let results = results.unwrap();
        assert!(results.len() <= 5, "Should respect limit in configuration");
    }

    #[test]
    fn test_empty_search() {
        setup_test_env();

        let searcher = LocationSearcher::new(false).unwrap();

        // Test empty query
        let results = searcher.search(&[""]);
        assert!(results.is_ok(), "Empty search should not error");

        // Test very specific non-existent location
        let results = searcher.search(&["XYZ123NONEXISTENT"]);
        assert!(
            results.is_ok(),
            "Non-existent location search should not error"
        );
    }
}
