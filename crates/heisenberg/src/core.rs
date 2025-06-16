//! Core location search functionality for the Heisenberg library.
//!
//! This module provides the main [`LocationSearcher`] interface for finding and resolving
//! geographic locations from unstructured text input. It combines full-text search with
//! hierarchical location resolution to provide accurate location matching.
//!
//! # Quick Start
//!
//! ```rust
//! use heisenberg::LocationSearcher;
//!
//! // Create a searcher (will build indexes on first run)
//! let searcher = LocationSearcher::new_embedded()?;
//!
//! // Simple search
//! let results = searcher.search(&["London"])?;
//!
//! // Resolve to structured location data
//! let resolved =
//!     searcher.resolve_location::<_, heisenberg::GenericEntry>(&["Paris", "France"])?;
//! # Ok::<() , heisenberg::error::HeisenbergError>(())
//! ```
//!
//! # Search Types
//!
//! The searcher provides several search modes:
//! - **Administrative search**: Find countries, states, provinces, etc.
//! - **Place search**: Find cities, towns, landmarks, and points of interest
//! - **Combined search**: Automatically determine the best search strategy
//! - **Resolution**: Convert search results into structured location hierarchies

pub use heisenberg_data_processing::DataSource;
use polars::prelude::*;
use tracing::{info, instrument};

use crate::{
    ResolveConfig,
    backfill::{
        LocationEntry, LocationResults, resolve_search_candidate, resolve_search_candidate_batches,
    },
    data::{LocationSearchData, embedded::METADATA},
    error::HeisenbergError,
    index::LocationSearchIndex,
    search::{
        AdminSearchParams, PlaceSearchParams, SearchConfig, SearchResult, admin_search_inner,
        bulk_location_search_inner, location_search_inner, place_search_inner,
    },
};
pub type SearchResults = Vec<SearchResult>;
pub type SearchResultsBatch = Vec<Vec<SearchResult>>;

#[derive(Debug, Clone, Default)]
pub struct ResolveSearchConfig {
    /// Configuration for the underlying search operations
    pub search_config: SearchConfig,
    /// Configuration for the resolution process
    pub resolve_config: ResolveConfig,
}

/// The main location searcher that provides all search and resolution functionality.
///
/// This struct handles loading and indexing of geographic data, and provides
/// methods for searching and resolving location information. It maintains
/// internal indexes for efficient text search and caches data for fast lookups.
///
/// # Examples
///
/// Basic usage:
/// ```rust
/// use heisenberg::LocationSearcher;
///
/// let searcher = LocationSearcher::new_embedded()?;
/// let results = searcher.search(&["Tokyo", "Japan"])?;
/// println!("Found {} search results", results.len());
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
///
/// With custom configuration:
/// ```rust
/// use heisenberg::{LocationSearcher, SearchConfig};
///
/// let config = SearchConfig::builder()
///     .limit(5)
///     .place_importance_threshold(2)
///     .build();
///
/// let searcher = LocationSearcher::new_embedded()?;
/// let results = searcher.search_with_config(&["Berlin"], &config)?;
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
#[derive(Clone)]
pub struct LocationSearcher {
    index: LocationSearchIndex,
    data: LocationSearchData,
}

impl LocationSearcher {
    /// Create a new `LocationSearcher` with smart initialization.
    ///
    /// This will try to load existing indexes if they're up-to-date, otherwise
    /// it will create new ones from the specified data source.
    ///
    /// # Arguments
    ///
    /// * `data_source` - The data source to use for geographic data
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{DataSource, LocationSearcher};
    ///
    /// // Use smart initialization with Cities15000 data
    /// let searcher = LocationSearcher::initialize(DataSource::Cities15000)?;
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    #[instrument(name = "Smart Initialize LocationSearcher", level = "info")]
    pub fn initialize(data_source: DataSource) -> Result<Self, HeisenbergError> {
        info!(
            "Smart initializing LocationSearcher with data source: {:?}",
            data_source
        );
        let t_init = std::time::Instant::now();

        let data = LocationSearchData::new(data_source);
        let index = LocationSearchIndex::initialize(&data)?;

        info!(
            elapsed_seconds = ?t_init.elapsed(),
            "LocationSearcher smart initialization complete"
        );

        Ok(Self { index, data })
    }

    /// Create a new `LocationSearcher`, forcing recreation of indexes.
    ///
    /// This will always rebuild search indexes from scratch, ensuring they're
    /// completely up-to-date but taking longer to initialize.
    ///
    /// # Arguments
    ///
    /// * `data_source` - The data source to use for geographic data
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{DataSource, LocationSearcher};
    ///
    /// // Force rebuild of all indexes
    /// let searcher = LocationSearcher::new_with_fresh_indexes(DataSource::Cities15000)?;
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    #[instrument(name = "Create LocationSearcher with Fresh Indexes", level = "info")]
    pub fn new_with_fresh_indexes(data_source: DataSource) -> Result<Self, HeisenbergError> {
        info!(
            "Creating LocationSearcher with fresh indexes for data source: {:?}",
            data_source
        );
        let t_init = std::time::Instant::now();

        let data = LocationSearchData::new(data_source);
        let index = LocationSearchIndex::new(&data, true)?; // overwrite = true

        info!(
            elapsed_seconds = ?t_init.elapsed(),
            "LocationSearcher creation with fresh indexes complete"
        );

        Ok(Self { index, data })
    }

    /// Create a `LocationSearcher` using embedded data.
    ///
    /// This uses the data that was embedded at compile time, providing the
    /// fastest initialization since no disk I/O or index building is required.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::LocationSearcher;
    ///
    /// // Use embedded data (fastest initialization)
    /// let searcher = LocationSearcher::new_embedded()?;
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    #[instrument(name = "Create LocationSearcher with Embedded Data", level = "info")]
    pub fn new_embedded() -> Result<Self, HeisenbergError> {
        info!("Creating LocationSearcher with embedded data");
        let t_init = std::time::Instant::now();

        let data = LocationSearchData::new_embedded();

        // For embedded data, we might not need traditional Tantivy indexes
        // since the data is already in memory. But for now, we'll create them.
        let index = LocationSearchIndex::initialize(&data)?;

        info!(
            elapsed_seconds = ?t_init.elapsed(),
            "LocationSearcher embedded initialization complete"
        );

        Ok(Self { index, data })
    }

    /// Try to load existing `LocationSearcher` from cached indexes.
    ///
    /// Returns None if indexes don't exist or are invalid. This is useful
    /// for checking if a searcher can be loaded quickly before falling back
    /// to slower initialization methods.
    ///
    /// # Arguments
    ///
    /// * `data_source` - The data source to check for existing indexes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{DataSource, LocationSearcher};
    ///
    /// // Try to load existing, fall back to initialization if needed
    /// let searcher = if let Some(existing) = LocationSearcher::load_existing(DataSource::Cities15000)?
    /// {
    ///     existing
    /// } else {
    ///     LocationSearcher::initialize(DataSource::Cities15000)?
    /// };
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    #[instrument(name = "Load Existing LocationSearcher", level = "info")]
    pub fn load_existing(data_source: DataSource) -> Result<Option<Self>, HeisenbergError> {
        info!(
            "Attempting to load existing LocationSearcher for data source: {:?}",
            data_source
        );

        let data = LocationSearchData::new(data_source);

        if let Some(index) = LocationSearchIndex::load_existing(&data_source)? {
            // Verify the indexes are up-to-date with the data
            if index.is_up_to_date(&data)? {
                info!("Successfully loaded existing up-to-date LocationSearcher");
                return Ok(Some(Self { index, data }));
            }
            info!("Existing indexes are out of date");
        } else {
            info!("No existing indexes found");
        }

        Ok(None)
    }

    /// Create a `LocationSearcher` from pre-built components.
    ///
    /// This is useful for advanced use cases where you want to customize
    /// the data loading or index creation process.
    ///
    /// # Arguments
    ///
    /// * `data` - Pre-configured `LocationSearchData`
    /// * `index` - Pre-built `LocationSearchIndex`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{DataSource, LocationSearchData, LocationSearchIndex, LocationSearcher};
    ///
    /// let data = LocationSearchData::new(DataSource::Cities15000);
    /// let index = LocationSearchIndex::new(&data, false)?;
    /// let searcher = LocationSearcher::from_components(data, index);
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    pub fn from_components(data: LocationSearchData, index: LocationSearchIndex) -> Self {
        Self { index, data }
    }

    /// Check if indexes exist for a given data source without loading them.
    ///
    /// # Arguments
    ///
    /// * `data_source` - The data source to check
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{DataSource, LocationSearcher};
    ///
    /// if LocationSearcher::indexes_exist(DataSource::Cities15000) {
    ///     println!("Indexes are available for fast loading");
    /// } else {
    ///     println!("Indexes will need to be built");
    /// }
    /// ```
    #[must_use]
    pub fn indexes_exist(data_source: DataSource) -> bool {
        LocationSearchIndex::exists_for_source(&data_source)
    }

    /// Legacy constructor for backward compatibility.
    ///
    /// This method maintains the old API while using the new internal structure.
    ///
    /// # Arguments
    ///
    /// * `data_source` - The data source to use
    /// * `overwrite_fts_indexes` - If true, rebuild indexes from scratch
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{DataSource, LocationSearcher};
    ///
    /// // Legacy API (backward compatible)
    /// let searcher = LocationSearcher::initialize(DataSource::Cities15000)?;
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    #[instrument(name = "Initialize LocationSearcher (Legacy)", level = "info")]
    pub fn new(
        data_source: DataSource,
        overwrite_fts_indexes: bool,
    ) -> Result<Self, HeisenbergError> {
        if overwrite_fts_indexes {
            Self::new_with_fresh_indexes(data_source)
        } else {
            Self::initialize(data_source)
        }
    }

    /// Get information about the searcher's configuration.
    pub fn info(&self) -> SearcherInfo {
        SearcherInfo {
            data_source: *self.data.data_source(),
            has_admin_index: true, // We always have both with LocationSearchIndex
            has_places_index: true,
            embedded_metadata: METADATA.clone(),
        }
    }

    /// Check if the searcher's indexes are up-to-date with the underlying data.
    pub fn is_up_to_date(&self) -> Result<bool, HeisenbergError> {
        self.index.is_up_to_date(&self.data).map_err(From::from)
    }

    /// Rebuild the searcher's indexes from scratch.
    pub fn rebuild_indexes(&mut self) -> Result<(), HeisenbergError> {
        info!("Rebuilding indexes for LocationSearcher");
        self.index = LocationSearchIndex::new(&self.data, true)?;
        Ok(())
    }

    // === Low-level searches (unchanged but with custom return types) ===

    /// Search for administrative entities at specific levels.
    ///
    /// Administrative entities include countries (level 0), states/provinces (level 1),
    /// counties/regions (level 2), and local administrative divisions (levels 3-4).
    ///
    /// # Arguments
    ///
    /// * `term` - The search term to look for
    /// * `levels` - Administrative levels to search (0-4, where 0 is country level)
    /// * `previous_result` - Optional previous search result to filter by
    /// * `params` - Search parameters controlling behavior and scoring
    ///
    /// # Returns
    ///
    /// An optional `DataFrame` containing matching administrative entities,
    /// or None if no matches were found.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{AdminSearchParams, LocationSearcher};
    ///
    /// let searcher = LocationSearcher::new_embedded()?;
    ///
    /// // Search for countries
    /// let countries = searcher.admin_search("France", &[0], None, &AdminSearchParams::default())?;
    ///
    /// // Search for states/provinces
    /// let states = searcher.admin_search("California", &[1], None, &AdminSearchParams::default())?;
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    #[instrument(name = "Admin Search", level = "debug", skip_all)]
    pub fn admin_search(
        &self,
        term: impl AsRef<str>,
        levels: &[u8],
        previous_result: Option<DataFrame>,
        params: &AdminSearchParams,
    ) -> Result<Option<DataFrame>, HeisenbergError> {
        admin_search_inner(
            term.as_ref(),
            levels,
            &self.index.admin,
            self.data.admin_search_df(),
            previous_result.map(From::from),
            params,
        )
        .map(|result| result.map(From::from))
        .map_err(From::from)
    }

    /// Search for places like cities, towns, landmarks, and points of interest.
    ///
    /// This searches the places dataset which includes populated places,
    /// landmarks, geographic features, and other location points of interest.
    ///
    /// # Arguments
    ///
    /// * `term` - The search term to look for
    /// * `previous_result` - Optional previous search result to filter by
    /// * `params` - Search parameters controlling behavior and scoring
    ///
    /// # Returns
    ///
    /// An optional `DataFrame` containing matching places,
    /// or None if no matches were found.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use heisenberg::{LocationSearcher, PlaceSearchParams};
    ///
    /// let searcher = LocationSearcher::new_embedded()?;
    /// let places = searcher.place_search("Tokyo", None, &PlaceSearchParams::default())?;
    ///
    /// if let Some(results) = places {
    ///     println!("Found {} places matching 'Tokyo'", results.height());
    /// }
    /// # Ok::<(), heisenberg::error::HeisenbergError>(())
    /// ```
    #[instrument(name = "Place Search", level = "debug", skip_all)]
    pub fn place_search(
        &self,
        term: impl AsRef<str>,
        previous_result: Option<DataFrame>,
        params: &PlaceSearchParams,
    ) -> Result<Option<DataFrame>, HeisenbergError> {
        place_search_inner(
            term.as_ref(),
            &self.index.places,
            self.data.place_search_df(),
            previous_result.map(From::from),
            params,
        )
        .map(|result| result.map(From::from))
        .map_err(From::from)
    }

    // === Simplified mid-level search methods ===

    /// Search for locations using the provided terms.
    ///
    /// **Important**: Input terms should be provided in descending 'size' order
    /// (largest to smallest location) for optimal results:
    /// `['Country', 'State', 'County', 'Place']`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use heisenberg::LocationSearcher;
    /// # let searcher = LocationSearcher::new_embedded().unwrap();
    /// // Single term
    /// let results = searcher.search(&["Tokyo"]).unwrap();
    ///
    /// // Multi-term (largest to smallest)
    /// let results = searcher.search(&["Japan", "Tokyo"]).unwrap();
    /// let results = searcher
    ///     .search(&["United States", "California", "San Francisco"])
    ///     .unwrap();
    /// ```
    pub fn search<Term>(&self, input_terms: &[Term]) -> Result<SearchResults, HeisenbergError>
    where
        Term: AsRef<str>,
    {
        self.search_with_config(input_terms, &SearchConfig::default())
    }

    /// Search for locations using the provided terms and custom configuration.
    ///
    /// **Important**: Input terms should be provided in descending 'size' order
    /// (largest to smallest location) for optimal results:
    /// `['Country', 'State', 'County', 'Place']`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use heisenberg::{LocationSearcher, SearchConfig};
    /// # let searcher = LocationSearcher::new_embedded().unwrap();
    /// let config = SearchConfig::default();
    /// let results = searcher
    ///     .search_with_config(&["Germany", "Berlin"], &config)
    ///     .unwrap();
    /// ```
    pub fn search_with_config<Term>(
        &self,
        input_terms: &[Term],
        config: &SearchConfig,
    ) -> Result<SearchResults, HeisenbergError>
    where
        Term: AsRef<str>,
    {
        let input_terms = input_terms.iter().map(AsRef::as_ref).collect::<Vec<_>>();

        location_search_inner(
            &input_terms,
            &self.index.admin,
            self.data.admin_search_df(),
            &self.index.places,
            self.data.place_search_df(),
            config,
        )
        .map_err(From::from)
    }

    /// Search for multiple location queries in batch for improved performance.
    ///
    /// **Important**: Each batch of input terms should be provided in descending 'size' order
    /// (largest to smallest location) for optimal results:
    /// `['Country', 'State', 'County', 'Place']`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use heisenberg::LocationSearcher;
    /// # let searcher = LocationSearcher::new_embedded().unwrap();
    /// let queries = vec![
    ///     vec!["Japan", "Tokyo"],
    ///     vec!["France", "Paris"],
    ///     vec!["United States", "New York"],
    /// ];
    /// let batch_results = searcher.search_bulk(&queries).unwrap();
    /// ```
    pub fn search_bulk<Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
    ) -> Result<SearchResultsBatch, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
    {
        self.search_bulk_with_config(all_raw_input_batches, &SearchConfig::default())
    }

    pub fn search_bulk_with_config<Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
        config: &SearchConfig,
    ) -> Result<SearchResultsBatch, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
    {
        let all_raw_input_batches = all_raw_input_batches
            .iter()
            .map(|batch| batch.as_ref().iter().map(AsRef::as_ref).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let all_raw_input_batches = all_raw_input_batches
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();

        bulk_location_search_inner(
            &all_raw_input_batches,
            &self.index.admin,
            self.data.admin_search_df(),
            &self.index.places,
            self.data.place_search_df(),
            config,
        )
        .map_err(From::from)
    }

    pub fn resolve<Entry: LocationEntry>(
        &self,
        search_results: &SearchResults,
    ) -> Result<LocationResults<Entry>, HeisenbergError> {
        self.resolve_with_config(search_results, &ResolveConfig::default())
    }

    pub fn resolve_with_config<Entry: LocationEntry>(
        &self,
        search_results: &SearchResults,
        config: &ResolveConfig,
    ) -> Result<LocationResults<Entry>, HeisenbergError> {
        resolve_search_candidate(search_results, &self.data.admin_search_df(), config)
            .map_err(From::from)
    }

    pub fn resolve_batch<Entry: LocationEntry>(
        &self,
        search_results_batches: SearchResultsBatch,
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError> {
        self.resolve_batch_with_config(search_results_batches, &ResolveConfig::default())
    }

    pub fn resolve_batch_with_config<Entry: LocationEntry>(
        &self,
        search_results_batches: SearchResultsBatch,
        config: &ResolveConfig,
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError> {
        resolve_search_candidate_batches(
            search_results_batches,
            &self.data.admin_search_df(),
            config,
        )
        .map_err(From::from)
    }

    // === High-level search and resolve methods ===

    /// Resolve location terms into complete administrative hierarchies.
    ///
    /// **Important**: Input terms should be provided in descending 'size' order
    /// (largest to smallest location) for optimal results:
    /// `['Country', 'State', 'County', 'Place']`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use heisenberg::{LocationSearcher, GenericEntry};
    /// # let searcher = LocationSearcher::new_embedded().unwrap();
    /// // Resolve with proper ordering (largest to smallest)
    /// let resolved = searcher
    ///     .resolve_location::<_, GenericEntry>(&["United States", "California", "San Francisco"])
    ///     .unwrap();
    /// let context = &resolved[0].context;
    /// ```
    pub fn resolve_location<Term, Entry>(
        &self,
        input_terms: &[Term],
    ) -> Result<LocationResults<Entry>, HeisenbergError>
    where
        Term: AsRef<str>,
        Entry: LocationEntry,
    {
        self.resolve_location_with_config(input_terms, &ResolveSearchConfig::default())
    }

    /// Resolve location terms into complete administrative hierarchies with custom configuration.
    ///
    /// **Important**: Input terms should be provided in descending 'size' order
    /// (largest to smallest location) for optimal results:
    /// `['Country', 'State', 'County', 'Place']`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use heisenberg::{LocationSearcher, GenericEntry, ResolveSearchConfig};
    /// # let searcher = LocationSearcher::new_embedded().unwrap();
    /// let config = ResolveSearchConfig::default();
    /// let resolved = searcher
    ///     .resolve_location_with_config::<_, GenericEntry>(&["France", "Paris"], &config)
    ///     .unwrap();
    /// ```
    pub fn resolve_location_with_config<Term, Entry>(
        &self,
        input_terms: &[Term],
        config: &ResolveSearchConfig,
    ) -> Result<LocationResults<Entry>, HeisenbergError>
    where
        Term: AsRef<str>,
        Entry: LocationEntry,
    {
        let search_results = self.search_with_config(input_terms, &config.search_config)?;

        resolve_search_candidate(
            &search_results,
            &self.data.admin_search_df(),
            &config.resolve_config,
        )
        .map_err(From::from)
    }

    pub fn resolve_location_batch<Entry, Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
        Entry: LocationEntry,
    {
        self.resolve_location_batch_with_config(
            all_raw_input_batches,
            &ResolveSearchConfig::default(),
        )
    }

    pub fn resolve_location_batch_with_config<Entry, Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
        config: &ResolveSearchConfig,
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
        Entry: LocationEntry,
    {
        let search_results_batches =
            self.search_bulk_with_config(all_raw_input_batches, &config.search_config)?;

        resolve_search_candidate_batches(
            search_results_batches,
            &self.data.admin_search_df(),
            &config.resolve_config,
        )
        .map_err(From::from)
    }

    // === Utility Methods ===

    /// Access the underlying `LocationSearchIndex` for advanced operations.
    pub fn index(&self) -> &LocationSearchIndex {
        &self.index
    }

    /// Access the underlying `LocationSearchData` for advanced operations.
    pub fn data(&self) -> &LocationSearchData {
        &self.data
    }
}

impl Default for LocationSearcher {
    fn default() -> Self {
        // Default to embedded data if available
        Self::new_embedded().expect("Failed to create default LocationSearcher")
    }
}

/// Create a `LocationSearcher` from pre-built components.
///
/// This is useful for advanced use cases where you want to customize
/// the data loading or index creation process.
///
/// # Arguments
///
/// * `data` - Pre-configured `LocationSearchData`
/// * `index` - Pre-built `LocationSearchIndex`
///
/// # Examples
///
/// ```rust
/// use heisenberg::{DataSource, LocationSearchData, LocationSearchIndex, LocationSearcher};
///
/// let data = LocationSearchData::new(DataSource::Cities15000);
/// let index = LocationSearchIndex::new(&data, false)?;
/// let searcher = LocationSearcher::from((index, data));
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
impl From<(LocationSearchIndex, LocationSearchData)> for LocationSearcher {
    fn from((index, data): (LocationSearchIndex, LocationSearchData)) -> Self {
        Self { index, data }
    }
}

/// Information about a `LocationSearcher`'s configuration and state.
#[derive(Debug, Clone)]
pub struct SearcherInfo {
    pub data_source: DataSource,
    pub has_admin_index: bool,
    pub has_places_index: bool,
    pub embedded_metadata: heisenberg_data_processing::embedded::EmbeddedMetadata,
}

impl SearcherInfo {
    /// Get a human-readable summary of the searcher.
    pub fn summary(&self) -> String {
        format!(
            "LocationSearcher using {:?} with {} admin entries and {} place entries",
            self.data_source,
            self.embedded_metadata.admin_df.rows,
            self.embedded_metadata.place_df.rows
        )
    }

    /// Check if the searcher is using embedded data.
    pub fn is_embedded(&self) -> bool {
        self.data_source == self.embedded_metadata.source
    }

    /// Get the total number of indexed locations.
    pub fn total_locations(&self) -> usize {
        self.embedded_metadata.admin_df.rows + self.embedded_metadata.place_df.rows
    }
}

// === Builder Pattern (Optional) ===

/// Builder for creating `LocationSearcher` with custom configuration.
#[derive(Debug, Clone)]
pub struct LocationSearcherBuilder {
    data_source: Option<DataSource>,
    force_rebuild: bool,
    embedded_fallback: bool,
}

impl LocationSearcherBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_source: None,
            force_rebuild: false,
            embedded_fallback: true,
        }
    }

    /// Set the data source.
    #[must_use]
    pub fn data_source(mut self, source: DataSource) -> Self {
        self.data_source = Some(source);
        self
    }

    /// Force rebuilding of indexes.
    #[must_use]
    pub fn force_rebuild(mut self, rebuild: bool) -> Self {
        self.force_rebuild = rebuild;
        self
    }

    /// Enable or disable fallback to embedded data.
    #[must_use]
    pub fn embedded_fallback(mut self, fallback: bool) -> Self {
        self.embedded_fallback = fallback;
        self
    }

    /// Build the `LocationSearcher`.
    pub fn build(self) -> Result<LocationSearcher, HeisenbergError> {
        let data_source = self.data_source.unwrap_or_default();

        if self.force_rebuild {
            LocationSearcher::new_with_fresh_indexes(data_source)
        } else if self.embedded_fallback && data_source == METADATA.source {
            LocationSearcher::new_embedded()
        } else {
            LocationSearcher::initialize(data_source)
        }
    }
}

impl Default for LocationSearcherBuilder {
    fn default() -> Self {
        Self::new()
    }
}
