//! Intelligent search orchestration that automatically determines optimal search strategies.
//!
//! This module implements the core search logic that analyzes input terms and decides
//! whether to perform administrative search, place search, or a combination of both.
//! It handles complex multi-term queries and coordinates between different search types.

use std::{
    cmp::{max, min},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};

use ahash::{AHashMap as HashMap, AHasher as DefaultHasher};
use polars::prelude::*;
use rayon::prelude::*;
use tracing::{debug, info, instrument, warn};

use super::{
    Result,
    admin_search::{AdminFrame, AdminSearchParams, SearchScoreAdminParams, admin_search_inner},
    place_search::{PlaceFrame, PlaceSearchParams, SearchScorePlaceParams, place_search_inner},
};
use crate::{
    SearchConfigBuilder,
    index::{AdminIndexDef, FTSIndex, FTSIndexSearchParams, PlacesIndexDef},
};

const MAX_ADMIN_LEVELS: usize = 5; // Admin levels 0 through 4

/// Represents a search result from either administrative or place search.
///
/// This enum wraps the result of a search operation and provides type safety
/// to distinguish between administrative entities (countries, states, etc.)
/// and places (cities, landmarks, etc.).
///
/// # Examples
///
/// ```rust
/// use heisenberg::{LocationSearcher, SearchResult};
///
/// let searcher = LocationSearcher::new_embedded()?;
/// let results = searcher.search(&["Tokyo"])?;
///
/// for result in results {
///     match result {
///         SearchResult::Admin(_) => println!("Found administrative entity"),
///         SearchResult::Place(_) => println!("Found place"),
///     }
/// }
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
#[derive(Debug, Clone)]
pub enum SearchResult {
    /// Administrative entity (country, state, province, etc.)
    Admin(AdminFrame),
    /// Place (city, town, landmark, point of interest, etc.)
    Place(PlaceFrame),
}
use SearchResult::*;
impl SearchResult {
    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(DataFrame) -> DataFrame,
    {
        match self {
            Self::Admin(df) => Self::Admin(df.map(f)),
            Self::Place(df) => Self::Place(df.map(f)),
        }
    }

    pub fn is_admin(&self) -> bool {
        matches!(self, Self::Admin(_))
    }

    pub fn is_place(&self) -> bool {
        matches!(self, Self::Place(_))
    }

    pub fn into_df(self) -> DataFrame {
        match self {
            Self::Admin(df) => df.into(),
            Self::Place(df) => df.into(),
        }
    }

    /// Get the name of the location
    pub fn name(&self) -> Option<&str> {
        self.column("name")
            .ok()
            .and_then(|s| s.str().ok())
            .and_then(|ca| ca.get(0))
    }

    /// Get the search relevance score
    pub fn score(&self) -> Option<f64> {
        self.column("score")
            .ok()
            .and_then(|s| s.f64().ok())
            .and_then(|ca| ca.get(0))
    }

    /// Get the feature code (geographic feature type)
    pub fn feature_code(&self) -> Option<&str> {
        self.column("feature_code")
            .ok()
            .and_then(|s| s.str().ok())
            .and_then(|ca| ca.get(0))
    }

    /// Get the GeoNames ID
    pub fn geoname_id(&self) -> Option<u32> {
        self.column("geonameId")
            .ok()
            .and_then(|s| s.u32().ok())
            .and_then(|ca| ca.get(0))
    }
}

impl Deref for SearchResult {
    type Target = DataFrame;

    fn deref(&self) -> &Self::Target {
        match self {
            SearchResult::Admin(df) => df,
            SearchResult::Place(df) => df,
        }
    }
}

impl DerefMut for SearchResult {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            SearchResult::Admin(df) => df,
            SearchResult::Place(df) => df,
        }
    }
}

/// Configuration for location search operations.
///
/// This struct controls all aspects of the search behavior, including result limits,
/// scoring weights, and search strategies. Use [`SearchConfigBuilder`] for an
/// ergonomic way to create configurations with sensible defaults.
///
/// # Examples
///
/// Creating a custom configuration:
/// ```rust
/// use heisenberg::SearchConfig;
///
/// let config = SearchConfig::builder()
///     .limit(10)
///     .place_importance_threshold(2)
///     .build();
/// ```
///
/// Using preset configurations:
/// ```rust
/// use heisenberg::SearchConfigBuilder;
///
/// // Fast search with fewer results
/// let fast_config = SearchConfigBuilder::fast().build();
///
/// // Comprehensive search with more results
/// let comprehensive_config = SearchConfigBuilder::comprehensive().build();
/// ```
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of results to return
    pub limit: usize,
    /// Include all data columns in results (useful for debugging)
    pub all_cols: bool,
    /// Maximum number of sequential administrative terms to process
    pub max_sequential_admin_terms: usize,
    /// Whether to attempt admin search for place candidates before place search
    pub attempt_place_candidate_as_admin_before_place_search: bool,
    /// Search parameters for administrative entity full-text search
    pub admin_fts_search_params: FTSIndexSearchParams,
    /// Scoring parameters for administrative entity search
    pub admin_search_score_params: SearchScoreAdminParams,
    /// Search parameters for place full-text search
    pub place_fts_search_params: FTSIndexSearchParams,
    /// Scoring parameters for place search
    pub place_search_score_params: SearchScorePlaceParams,
    /// Minimum importance tier for places (1=most important, 5=least important)
    pub place_min_importance_tier: u8,
}
impl SearchConfig {
    pub fn builder() -> SearchConfigBuilder {
        SearchConfigBuilder::default()
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        let default_limit = 20;
        Self {
            limit: default_limit,
            all_cols: false,
            max_sequential_admin_terms: 5,
            attempt_place_candidate_as_admin_before_place_search: true,
            admin_fts_search_params: FTSIndexSearchParams {
                limit: default_limit * 3, // Fetch more for ranking
                ..Default::default()
            },
            admin_search_score_params: Default::default(),
            place_fts_search_params: FTSIndexSearchParams {
                limit: default_limit * 3, // Fetch more for ranking
                ..Default::default()
            },
            place_search_score_params: Default::default(),
            place_min_importance_tier: 5, // Default from PlaceSearchParams
        }
    }
}

/// Represents the state of a location search query during processing
#[derive(Debug, Clone)]
struct QueryState {
    _unique_id: usize,
    admin_terms_for_main_sequence: Vec<String>,
    place_candidate_term: Option<String>,
    is_place_candidate_also_last_admin_term_in_sequence: bool,
    current_admin_term_idx: usize,
    last_successful_admin_context_df: Option<AdminFrame>,
    min_level_from_last_success: Option<u8>,
    admin_sequence_complete: bool,
    proactive_admin_search_complete: bool,
    place_search_complete: bool,
    results_for_this_query: Vec<SearchResult>,
}

impl QueryState {
    fn new(
        _unique_id: usize, // For debugging and tracking
        admin_terms_for_main_sequence: Vec<String>,
        place_candidate_term: Option<String>,
        is_place_candidate_also_last_admin_term_in_sequence: bool,
    ) -> Self {
        Self {
            _unique_id,
            admin_terms_for_main_sequence,
            place_candidate_term,
            is_place_candidate_also_last_admin_term_in_sequence,
            current_admin_term_idx: 0,
            last_successful_admin_context_df: None,
            min_level_from_last_success: None,
            admin_sequence_complete: false,
            proactive_admin_search_complete: false,
            place_search_complete: false,
            results_for_this_query: Vec::new(),
        }
    }
}

/// Represents a batch of similar admin searches that can be processed together
#[derive(Debug)]
struct AdminSearchBatch {
    term: String,
    levels: Vec<u8>,
    context_df: Option<AdminFrame>,
    query_indices: Vec<usize>,
}

/// Represents a batch of similar place searches
#[derive(Debug)]
struct PlaceSearchBatch {
    term: String,
    context_df: Option<AdminFrame>,
    query_indices: Vec<usize>,
}

const INVALID_QUERY_ID_PLACEHOLDER: usize = usize::MAX; // Marker for invalid queries

/// Cleans and validates search terms, removing empty entries
fn clean_search_terms(raw_terms: &[&str]) -> Vec<String> {
    raw_terms
        .iter()
        .filter_map(|s| {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                Some(trimmed.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Determines which terms should be treated as administrative entities
fn extract_admin_terms(cleaned_terms: &[String], max_admin_terms: usize) -> Vec<String> {
    let num_admin_terms = std::cmp::min(cleaned_terms.len(), max_admin_terms);
    cleaned_terms[0..num_admin_terms].to_vec()
}

/// Identifies a potential place candidate from remaining terms or last admin term
fn identify_place_candidate(
    cleaned_terms: &[String],
    admin_terms: &[String],
) -> (Option<String>, bool) {
    let num_admin_terms = admin_terms.len();

    if cleaned_terms.len() > num_admin_terms {
        // Extra terms beyond admin terms become place candidate
        let mut place_term = cleaned_terms[num_admin_terms].clone();
        if cleaned_terms.len() > num_admin_terms + 1 {
            let extra_terms = cleaned_terms[(num_admin_terms + 1)..].join(" ");
            place_term = format!("{place_term} {extra_terms}");
            debug!(
                "Concatenated extra terms into place candidate: '{}'",
                place_term
            );
        }
        (Some(place_term), false)
    } else if !admin_terms.is_empty() {
        // Use last admin term as place candidate when no extra terms
        (admin_terms.last().cloned(), true)
    } else {
        (None, false)
    }
}

/// Prepares search terms by cleaning and categorizing them
///
/// This function:
/// 1. Cleans and validates search terms
/// 2. Separates terms into admin sequence and place candidate
/// 3. Identifies if the place candidate is also part of the admin sequence
///
/// # Returns
/// A tuple containing:
/// * Admin terms for main sequence
/// * Optional place candidate term
/// * Boolean indicating if place candidate is also the last admin term
pub fn prepare_search_terms(
    search_terms_raw: &[&str],
    max_sequential_admin_terms: usize,
) -> Result<(Vec<String>, Option<String>, bool)> {
    let cleaned_terms = clean_search_terms(search_terms_raw);

    if cleaned_terms.is_empty() {
        warn!("No valid search terms provided after cleaning");
        return Ok((Vec::new(), None, false));
    }

    let admin_terms = extract_admin_terms(&cleaned_terms, max_sequential_admin_terms);
    let (place_candidate, is_place_candidate_also_last_admin) =
        identify_place_candidate(&cleaned_terms, &admin_terms);

    debug!("Admin terms for main sequence: {:?}", admin_terms);
    if let Some(ref pct) = place_candidate {
        debug!(
            "Place candidate term: '{}' (Is also last admin in sequence: {})",
            pct, is_place_candidate_also_last_admin
        );
    } else {
        debug!("No distinct place candidate term identified");
    }

    Ok((
        admin_terms,
        place_candidate,
        is_place_candidate_also_last_admin,
    ))
}

/// Processes the main administrative sequence of search terms
///
/// This function:
/// 1. Iteratively processes each admin term in sequence
/// 2. Determines appropriate admin levels for each term
/// 3. Uses context from previous terms to refine searches
/// 4. Collects results and updates context
///
/// # Returns
/// A tuple containing:
/// * Vector of DataFrames with admin search results
/// * Optional DataFrame with last successful context
fn process_admin_sequence(
    admin_terms: &[String],
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_data_lf: LazyFrame,
    initial_context: Option<AdminFrame>,
    config: &SearchConfig,
) -> Result<(Vec<AdminFrame>, Option<AdminFrame>)> {
    let mut results = Vec::new();
    let mut last_context = initial_context;
    let mut min_level_from_last_success: Option<u8> = None;

    let num_terms = admin_terms.len();
    let empty_admin_slots = MAX_ADMIN_LEVELS.saturating_sub(num_terms);

    // Create parameters for admin search
    let admin_params = AdminSearchParams {
        limit: config.limit,
        all_cols: config.all_cols,
        fts_search_params: config.admin_fts_search_params,
        search_score_params: config.admin_search_score_params,
    };

    // Process each admin term in sequence
    for (i, term) in admin_terms.iter().enumerate() {
        let natural_start_level = i as u8;
        let effective_start_level = match min_level_from_last_success {
            Some(min_prev) => std::cmp::max(natural_start_level, min_prev + 1),
            None => natural_start_level,
        };

        let current_search_window_end_level = std::cmp::min(
            (MAX_ADMIN_LEVELS - 1) as u8, // Max level is 4
            natural_start_level + (empty_admin_slots as u8),
        );

        // Calculate valid admin levels for this term
        let current_search_levels: Vec<u8> =
            if effective_start_level <= current_search_window_end_level {
                (effective_start_level..=current_search_window_end_level).collect()
            } else {
                Vec::new()
            };

        if current_search_levels.is_empty() {
            warn!(
                "Term '{}': No valid admin levels to search (effective range: {}-{}). Skipping.",
                term, effective_start_level, current_search_window_end_level
            );
            continue;
        }

        debug!(
            "Searching admin term '{}' for levels {:?} with current context",
            term, current_search_levels
        );

        // Execute admin search with context
        match admin_search_inner(
            term,
            &current_search_levels,
            admin_fts_index,
            admin_data_lf.clone(),
            last_context.clone(),
            &admin_params,
        ) {
            Ok(Some(df)) if !df.is_empty() => {
                info!(
                    "Found {} results for admin term '{}' in levels {:?}",
                    df.height(),
                    term,
                    current_search_levels
                );

                // Update results and context
                results.push(df.clone());
                last_context = Some(df.clone());

                // Extract min admin level from results for next search
                min_level_from_last_success = extract_min_admin_level(&df);
            }
            Ok(_) => {
                debug!(
                    "No results for admin term '{}' in levels {:?}",
                    term, current_search_levels
                );
                // min_level_from_last_success remains from previous successful search
            }
            Err(e) => {
                warn!("Error searching for admin term '{}': {:?}", term, e);
            }
        }
    }

    Ok((results, last_context))
}

/// Processes proactive admin search for a place candidate
///
/// This function attempts to find the place candidate in the administrative
/// data before treating it as a place. This is useful for places that are
/// also administrative entities (e.g., "London" as both a city and county).
///
/// # Returns
/// A tuple containing:
/// * Vector of DataFrames with proactive search results
/// * Optional DataFrame with last successful context
fn process_proactive_admin_search(
    place_term: &str,
    admin_terms_count: usize,
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_data_lf: LazyFrame,
    last_context: Option<AdminFrame>,
    config: &SearchConfig,
) -> Result<(Vec<AdminFrame>, Option<AdminFrame>)> {
    let additional_admin_start_level = admin_terms_count as u8;

    // Check if there are any valid admin levels to search
    if additional_admin_start_level >= MAX_ADMIN_LEVELS as u8 {
        debug!(
            "Skipping proactive admin search for '{}': no subsequent admin levels available (start_level: {})",
            place_term, additional_admin_start_level
        );
        return Ok((Vec::new(), last_context));
    }

    // Calculate valid admin levels for proactive search
    let additional_admin_levels: Vec<u8> =
        (additional_admin_start_level..(MAX_ADMIN_LEVELS as u8)).collect();

    if additional_admin_levels.is_empty() {
        return Ok((Vec::new(), last_context));
    }

    debug!(
        "Proactively searching place candidate '{}' as ADMIN at levels {:?} using current context",
        place_term, additional_admin_levels
    );

    // Create parameters for admin search
    let admin_params = AdminSearchParams {
        limit: config.limit,
        all_cols: config.all_cols,
        fts_search_params: config.admin_fts_search_params,
        search_score_params: config.admin_search_score_params,
    };

    // Execute admin search
    match admin_search_inner(
        place_term,
        &additional_admin_levels,
        admin_fts_index,
        admin_data_lf,
        last_context.clone(),
        &admin_params,
    ) {
        Ok(Some(df)) if !df.is_empty() => {
            info!(
                "Found {} results for place candidate '{}' as proactive ADMIN in levels {:?}",
                df.height(),
                place_term,
                additional_admin_levels
            );
            Ok((vec![df.clone()], Some(df)))
        }
        Ok(_) => {
            debug!(
                "No results for place candidate '{}' as proactive ADMIN in levels {:?}",
                place_term, additional_admin_levels
            );
            Ok((Vec::new(), last_context))
        }
        Err(e) => {
            warn!(
                "Error during proactive admin search for '{}': {:?}",
                place_term, e
            );
            Ok((Vec::new(), last_context))
        }
    }
}

/// Processes place search for a candidate term
///
/// This function searches for the place candidate in the places data,
/// using context from previous administrative searches to refine results.
///
/// # Returns
/// Optional DataFrame with place search results, if any
fn process_place_search(
    place_term: &str,
    places_fts_index: &FTSIndex<PlacesIndexDef>,
    places_data_lf: LazyFrame,
    last_context: Option<AdminFrame>,
    config: &SearchConfig,
) -> Result<Option<PlaceFrame>> {
    debug!(
        "Searching for place candidate '{}' as PLACE entity using context",
        place_term
    );

    // Create parameters for place search
    let place_params = PlaceSearchParams {
        limit: config.limit,
        all_cols: config.all_cols,
        min_importance_tier: config.place_min_importance_tier,
        center_lat: None, // place_search derives from previous_result if Some
        center_lon: None,
        fts_search_params: config.place_fts_search_params,
        search_score_params: config.place_search_score_params,
    };

    // Execute place search
    match place_search_inner(
        place_term,
        places_fts_index,
        places_data_lf,
        last_context,
        &place_params,
    ) {
        Ok(Some(df)) if !df.is_empty() => {
            info!(
                "Found {} results for place candidate '{}' as PLACE",
                df.height(),
                place_term
            );
            Ok(Some(df))
        }
        Ok(_) => {
            debug!("No results for place candidate '{}' as PLACE", place_term);
            Ok(None)
        }
        Err(e) => {
            warn!("Error searching for place '{}': {:?}", place_term, e);
            Ok(None)
        }
    }
}

/// This function takes a sequence of search terms and processes them to find matching
/// locations in a hierarchical manner:
///
/// 1. First, it processes administrative entities (countries, states, counties, etc.)
/// 2. Then, it processes place candidates (specific locations like landmarks, cities, etc.)
/// 3. It uses previous search context to refine subsequent searches
///
/// The function follows a flexible approach to match location hierarchies that may not
/// perfectly align with administrative boundaries.
///
/// # Arguments
/// * `search_terms_raw` - A slice of search terms to process (e.g., ["USA", "California", "Los Angeles"])
/// * `admin_fts_index` - FTS index for administrative entities
/// * `admin_data_lf` - LazyFrame containing administrative data
/// * `places_fts_index` - FTS index for places
/// * `places_data_lf` - LazyFrame containing place data
/// * `config` - Configuration parameters for the search process
///
/// # Returns
/// A vector of DataFrames containing the search results, with each DataFrame
/// representing a layer in the location hierarchy
#[instrument(name = "Location Search", level = "info", skip_all, fields(search_terms = ?search_terms_raw))]
pub fn location_search_inner(
    search_terms_raw: &[&str],
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_data_lf: LazyFrame,
    places_fts_index: &FTSIndex<PlacesIndexDef>,
    places_data_lf: LazyFrame,
    config: &SearchConfig,
) -> Result<Vec<SearchResult>> {
    let t_start = std::time::Instant::now();

    let (admin_terms, place_candidate_term, is_place_candidate_also_last_admin_term) =
        prepare_search_terms(search_terms_raw, config.max_sequential_admin_terms)?;

    if admin_terms.is_empty() && place_candidate_term.is_none() {
        warn!("No valid search terms provided after cleaning");
        return Ok(Vec::new());
    }

    let mut all_results: Vec<SearchResult> = Vec::new();
    let mut last_successful_context: Option<AdminFrame> = None;

    // --- 2. Process main administrative sequence ---
    if !admin_terms.is_empty() {
        let admin_seq_start = std::time::Instant::now();
        let (admin_results, context) = process_admin_sequence(
            &admin_terms,
            admin_fts_index,
            admin_data_lf.clone(),
            last_successful_context,
            config,
        )?;

        // Update results and context
        all_results.extend(admin_results.into_iter().map(SearchResult::Admin));
        last_successful_context = context;

        debug!(
            elapsed_ms = ?admin_seq_start.elapsed(),
            "Admin sequence processing complete"
        );
    }

    // --- 3. Process proactive admin search (if applicable) ---
    // For single-term inputs, skip proactive admin search to avoid confusion
    if let Some(ref place_term) = place_candidate_term {
        if search_terms_raw.len() == 1 {
            debug!(
                "Skipping proactive admin search for single-term input '{}' - already processed as admin",
                place_term
            );
        } else if config.attempt_place_candidate_as_admin_before_place_search
            && !is_place_candidate_also_last_admin_term
        {
            let proactive_start = std::time::Instant::now();
            let (proactive_results, context) = process_proactive_admin_search(
                place_term,
                admin_terms.len(),
                admin_fts_index,
                admin_data_lf.clone(),
                last_successful_context.clone(),
                config,
            )?;

            // Update results and context if found
            if !proactive_results.is_empty() {
                all_results.extend(proactive_results.into_iter().map(SearchResult::Admin));
                last_successful_context = context;

                debug!(
                    elapsed_ms = proactive_start.elapsed().as_millis(),
                    "Proactive admin search found results"
                );
            } else {
                debug!("No results from proactive admin search");
            }
        } else if is_place_candidate_also_last_admin_term {
            debug!(
                "Skipping proactive admin search for '{}': already processed as last admin term",
                place_term
            );
        }
    }

    // --- 4. Process place search (if place candidate exists) ---
    // For single-term inputs, skip place search to avoid confusion between admin and place results
    if let Some(ref place_term) = place_candidate_term {
        if search_terms_raw.len() == 1 {
            debug!(
                "Skipping place search for single-term input '{}' - treating as admin-only search",
                place_term
            );
        } else {
            let place_search_start = std::time::Instant::now();

            if let Some(place_result) = process_place_search(
                place_term,
                places_fts_index,
                places_data_lf,
                last_successful_context,
                config,
            )? {
                all_results.push(SearchResult::Place(place_result));

                debug!(
                    elapsed_ms = place_search_start.elapsed().as_millis(),
                    "Place search found results"
                );
            } else {
                debug!("No results from place search");
            }
        }
    }

    info!(
        total_elapsed_ms = t_start.elapsed().as_millis(),
        num_results = all_results.len(),
        "Location search complete"
    );

    Ok(all_results)
}

/// Generates a hash signature for a context DataFrame (for batching similar contexts)
///
/// This function creates a hash signature based on the geonameId values in the DataFrame,
/// allowing for efficient grouping of queries with similar contexts.
fn generate_context_signature(context_df: Option<&AdminFrame>) -> Option<u64> {
    context_df.and_then(|df| {
        df.column("geonameId")
            .ok()
            .and_then(|s| s.u32().ok())
            .map(|ca| {
                let mut ids: Vec<u32> = ca.into_no_null_iter().collect();
                ids.sort_unstable();
                let mut hasher = DefaultHasher::default();
                ids.hash(&mut hasher);
                hasher.finish()
            })
    })
}

/// Parses search terms into admin terms and place candidate
///
/// This function:
/// 1. Extracts the administrative terms based on max_sequential_admin_terms
/// 2. Identifies a potential place candidate term (either beyond admin terms or last admin term)
/// 3. Tracks if the place candidate is also the last admin term
fn parse_search_terms(
    terms: &[String],
    max_sequential_admin_terms: usize,
) -> (Vec<String>, Option<String>, bool) {
    let num_terms = terms.len();
    let num_admin_terms = min(num_terms, max_sequential_admin_terms);

    let admin_terms = terms[0..num_admin_terms].to_vec();

    let mut place_candidate_term: Option<String> = None;
    let mut is_place_candidate_also_last_admin_term_in_sequence = false;

    if num_terms > num_admin_terms {
        // If there are extra terms, combine them into a place candidate
        let mut pct = terms[num_admin_terms].clone();
        if num_terms > num_admin_terms + 1 {
            let extra_terms = terms[(num_admin_terms + 1)..].join(" ");
            pct = format!("{pct} {extra_terms}");
        }
        place_candidate_term = Some(pct);
    } else if !admin_terms.is_empty() {
        // If no extra terms but at least one admin term exists,
        // the last admin term can also be a place candidate
        place_candidate_term = admin_terms.last().cloned();
        is_place_candidate_also_last_admin_term_in_sequence = true;
    }
    (
        admin_terms,
        place_candidate_term,
        is_place_candidate_also_last_admin_term_in_sequence,
    )
}

/// Prepares query states by deduplicating and parsing the input batches
///
/// This function:
/// 1. Deduplicates identical input batches to avoid redundant processing
/// 2. Parses each input batch into administrative terms and place candidates
/// 3. Creates a QueryState object for each unique input batch
/// 4. Maps original input indices to unique query IDs for result reconstruction
fn prepare_query_states(
    all_raw_input_batches: &[&[&str]],
    max_sequential_admin_terms: usize,
) -> Result<(Vec<QueryState>, Vec<usize>)> {
    let mut unique_queries_map: HashMap<Vec<String>, usize> = HashMap::new();
    let mut query_states: Vec<QueryState> = Vec::new();
    let mut original_input_to_unique_id: Vec<usize> =
        Vec::with_capacity(all_raw_input_batches.len());

    for raw_input_batch in all_raw_input_batches {
        let cleaned_terms: Vec<String> = clean_search_terms(raw_input_batch);

        if cleaned_terms.is_empty() {
            original_input_to_unique_id.push(INVALID_QUERY_ID_PLACEHOLDER);
            continue;
        }

        let unique_id = *unique_queries_map
            .entry(cleaned_terms.clone())
            .or_insert_with(|| {
                let id = query_states.len();

                // Parse input into admin terms and place candidate
                let (
                    admin_terms_for_main_sequence,
                    place_candidate_term,
                    is_place_candidate_also_last_admin_term_in_sequence,
                ) = parse_search_terms(&cleaned_terms, max_sequential_admin_terms);
                // Create new query state
                query_states.push(QueryState::new(
                    id,
                    admin_terms_for_main_sequence,
                    place_candidate_term,
                    is_place_candidate_also_last_admin_term_in_sequence,
                ));
                id
            });
        original_input_to_unique_id.push(unique_id);
    }

    Ok((query_states, original_input_to_unique_id))
}

/// Calculates the target admin levels for a query state
///
/// This function determines the appropriate administrative levels to search for
/// the current term in a query state, considering:
/// 1. The current term index in the admin sequence
/// 2. Previous admin level matches (min_level_from_last_success)
/// 3. Available admin level slots
fn calculate_target_levels(qs: &QueryState, max_admin_levels: usize) -> Vec<u8> {
    if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
        return vec![]; // No more terms to search
    }

    let natural_start_level = qs.current_admin_term_idx as u8;
    let effective_start_level = match qs.min_level_from_last_success {
        Some(min_prev) => max(natural_start_level, min_prev + 1),
        None => natural_start_level,
    };

    let num_terms_in_main_seq = qs.admin_terms_for_main_sequence.len();
    let empty_admin_slots = max_admin_levels.saturating_sub(num_terms_in_main_seq);

    let current_search_window_end_level = min(
        (max_admin_levels - 1) as u8, // Max level is 4 for 5 levels (0-4)
        natural_start_level + (empty_admin_slots as u8),
    );

    if effective_start_level <= current_search_window_end_level {
        (effective_start_level..=current_search_window_end_level).collect()
    } else {
        vec![] // No valid levels
    }
}

/// Prepares admin search batches by grouping similar queries
///
/// This function groups query states that:
/// 1. Are searching for the same term
/// 2. Target the same admin levels
/// 3. Have similar context (if any)
fn prepare_admin_search_batches(
    query_states: &[QueryState],
    max_admin_levels: usize,
) -> Vec<AdminSearchBatch> {
    let mut search_batches = HashMap::new();
    // First loop: Collect batches based on current query states
    for (idx, qs) in query_states.iter().enumerate() {
        if qs.admin_sequence_complete
            || qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len()
        {
            continue;
        }

        let term = &qs.admin_terms_for_main_sequence[qs.current_admin_term_idx];
        let target_levels = calculate_target_levels(qs, max_admin_levels);

        if target_levels.is_empty() {
            continue;
        }

        let context_signature =
            generate_context_signature(qs.last_successful_admin_context_df.as_ref());
        let batch_key = (term.clone(), target_levels.clone(), context_signature);

        // Group by term, levels, and context signature
        search_batches
            .entry(batch_key)
            .or_insert_with(|| AdminSearchBatch {
                term: term.clone(),
                levels: target_levels,
                context_df: qs.last_successful_admin_context_df.clone(),
                query_indices: Vec::new(),
            })
            .query_indices
            .push(idx);
    }

    search_batches.into_values().collect()
}

/// Advances "stuck" queries that couldn't form search batches
///
/// This function:
/// 1. Identifies queries that couldn't form batches (e.g., due to empty target levels)
/// 2. Advances their current_admin_term_idx
/// 3. Updates admin_sequence_complete flag if needed
/// 4. Returns true if any queries were advanced
fn advance_stuck_queries(query_states: &mut [QueryState], max_admin_levels: usize) -> bool {
    let mut made_progress = false;

    for qs in query_states.iter_mut() {
        if !qs.admin_sequence_complete
            && qs.current_admin_term_idx < qs.admin_terms_for_main_sequence.len()
        {
            // Check if this query was part of any batch (not strictly necessary here if batching is exhaustive, but good for logic)
            // The primary condition for advancing here is if target_levels was empty.
            let target_levels = calculate_target_levels(qs, max_admin_levels);
            if target_levels.is_empty() {
                qs.current_admin_term_idx += 1;
                made_progress = true;
                if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                    qs.admin_sequence_complete = true;
                }
            }
        }
    }
    made_progress
}

/// Extracts the minimum admin level from a DataFrame
fn extract_min_admin_level(df: &DataFrame) -> Option<u8> {
    match df.column("admin_level") {
        Ok(admin_level_series) => match admin_level_series.cast(&DataType::UInt8) {
            Ok(casted_series) => casted_series.u8().ok().and_then(|ca| ca.min()),
            Err(_) => {
                warn!("Could not cast 'admin_level' to UInt8 for min_level extraction.");
                None
            }
        },
        Err(_) => {
            warn!("'admin_level' column not found for min_level extraction.");
            None
        }
    }
}

type AdminSearchResult = Vec<(Vec<usize>, Option<AdminFrame>, Option<u8>)>;

/// Executes admin search batches in parallel
///
/// This function:
/// 1. Converts batches to parallel tasks
/// 2. Executes admin_search_inner for each batch
/// 3. Processes and collects the results
/// 4. Returns a vector of (query_indices, result_df, min_level) tuples
fn execute_admin_search_batches(
    batches: Vec<AdminSearchBatch>,
    admin_data_lf: &LazyFrame,
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_search_params: &AdminSearchParams,
) -> Result<AdminSearchResult> {
    // Execute batches in parallel using rayon
    batches
        .into_par_iter()
        .map(|batch| {
            let AdminSearchBatch {
                term,
                levels,
                context_df,
                query_indices,
            } = batch;

            debug!(
                "Searching admin term '{}' for levels {:?} for {} queries.",
                term,
                levels,
                query_indices.len()
            );

            let search_result = admin_search_inner(
                &term,
                &levels,
                admin_fts_index,
                admin_data_lf.clone(),
                context_df,
                admin_search_params,
            );

            match search_result {
                Ok(Some(result_df)) if !result_df.is_empty() => {
                    let min_level = extract_min_admin_level(&result_df);
                    Ok((query_indices, Some(result_df), min_level))
                }
                Ok(_) => Ok((query_indices, None, None)),
                Err(e) => {
                    warn!("Error in admin search for term '{}': {:?}", term, e);
                    Ok((query_indices, None, None))
                }
            }
        })
        .collect()
}

/// Updates the admin filter for the next pass based on results from the current pass
///
/// This function:
/// 1. Extracts parent links from search results
/// 2. Creates a filter expression to narrow the search space
/// 3. Applies the filter to the admin data LazyFrame
fn update_admin_filter_for_next_pass(
    all_results_this_pass: &[AdminFrame],
    admin_data_lf: &LazyFrame,
    original_admin_data_height: u32,
) -> Result<LazyFrame> {
    // Concat all results from this pass
    let concatenated_results = {
        let lazy_results: Vec<_> = all_results_this_pass
            .iter()
            .map(|df| df.clone().lazy())
            .collect();

        if lazy_results.is_empty() {
            return Ok(admin_data_lf.clone());
        }

        match concat(
            &lazy_results,
            UnionArgs {
                diagonal: true,
                ..Default::default()
            },
        )
        .and_then(|lf| lf.collect())
        {
            Ok(df) => df,
            Err(e) => {
                warn!("Error concatenating results: {:?}", e);
                return Ok(admin_data_lf.clone());
            }
        }
    };

    if concatenated_results.is_empty() {
        debug!("No results to use for filtering next pass.");
        return Ok(admin_data_lf.clone());
    }

    // Get unique parent links based on geonameId
    let unique_parents = match concatenated_results.unique_stable(
        Some(&["geonameId".to_string()]),
        UniqueKeepStrategy::First,
        None,
    ) {
        Ok(df) => df,
        Err(e) => {
            warn!("Error getting unique parents: {:?}", e);
            return Ok(admin_data_lf.clone());
        }
    };

    // Build filter expressions based on parent-child relationships
    let mut filter_exprs = Vec::new();

    // Extract admin level and code information
    if let Ok(admin_level_ca) = unique_parents.column("admin_level").and_then(|s| s.u8()) {
        for i in 0..unique_parents.height() {
            if let Some(parent_level) = admin_level_ca.get(i) {
                // Skip if at max admin level (no children possible)
                if parent_level >= (MAX_ADMIN_LEVELS - 1) as u8 {
                    continue;
                }

                // Build the complete hierarchical path for this parent
                let mut hierarchical_path = Vec::new();
                for level in 0..=parent_level {
                    let code_col_name = format!("admin{level}_code");
                    if let Ok(code_series) = unique_parents.column(&code_col_name)
                        && let Ok(code_ca) = code_series.str()
                        && let Some(code) = code_ca.get(i)
                        && !code.is_empty()
                    {
                        hierarchical_path.push((level, code.to_string()));
                    }
                }

                // Only create filters for immediate children (parent_level + 1)
                let child_level = parent_level + 1;
                if child_level < MAX_ADMIN_LEVELS as u8 {
                    // Children must match ALL parent codes in the hierarchical path
                    let mut child_filter_parts = vec![col("admin_level").eq(lit(child_level))];

                    for (code_level, code_value) in &hierarchical_path {
                        let code_col_name = format!("admin{code_level}_code");
                        child_filter_parts.push(col(&code_col_name).eq(lit(code_value.clone())));
                    }

                    // Combine all filter parts with AND
                    if let Some(combined_child_filter) = child_filter_parts
                        .into_iter()
                        .reduce(|acc, expr| acc.and(expr))
                    {
                        debug!(
                            "Adding filter for child level {} with {} parent codes",
                            child_level,
                            hierarchical_path.len()
                        );
                        filter_exprs.push(combined_child_filter);
                    }
                }
            }
        }
    }

    // If no filter expressions, use original data
    if filter_exprs.is_empty() {
        debug!("No parent-child relationships found for filtering next pass.");
        return Ok(admin_data_lf.clone());
    }

    debug!("Created {} filter expressions", filter_exprs.len());

    // Combine filters with OR to include all possible children
    let combined_filter = filter_exprs
        .into_iter()
        .reduce(|acc, expr| acc.or(expr))
        .unwrap_or(lit(true));

    // Apply the filter to create the new globally filtered LazyFrame
    let potential_next_active_lf = admin_data_lf.clone().filter(combined_filter);

    let t0 = std::time::Instant::now();
    let potential_next_active_lf = potential_next_active_lf.collect();
    info!("Lf collect took {:.5}s", t0.elapsed().as_secs_f32());

    let active_admin_lf_for_pass = match potential_next_active_lf {
        Ok(active_admin_df) if !active_admin_df.is_empty() => {
            debug!(
                "Filtered admin data from {} to {} rows ({}%)",
                original_admin_data_height,
                active_admin_df.height(),
                active_admin_df.height() as f64 / original_admin_data_height as f64 * 100.0
            );
            active_admin_df.lazy()
        }
        Ok(_) => {
            debug!("Filtered LazyFrame is empty, using original LazyFrame.");
            admin_data_lf.clone()
        }
        Err(e) => {
            warn!("Error collecting potential_next_active_lf: {:?}", e);
            admin_data_lf.clone()
        }
    };

    Ok(active_admin_lf_for_pass)
}
/// Processes the main administrative sequence of searches in parallel
///
/// This function:
/// 1. Iteratively processes administrative search terms in batches
/// 2. Groups similar searches to maximize efficiency
/// 3. Executes searches in parallel
/// 4. Updates query states with results
/// 5. Continues until all admin sequences are complete
fn process_main_admin_sequence(
    query_states: &mut [QueryState],
    admin_data_lf: &LazyFrame,
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_search_params: &AdminSearchParams,
) -> Result<()> {
    // Main processing loop continues until all query admin sequences are complete
    let mut iteration_count = 0;
    let original_admin_data_height = admin_data_lf
        .clone()
        .select([len().alias("count")])
        .collect()?
        .column("count")?
        .u32()
        .unwrap()
        .get(0)
        .unwrap();
    // active_admin_lf_for_pass will be the globally filtered LazyFrame for the current pass
    let mut active_admin_lf_for_pass = admin_data_lf.clone(); // Initialize with the full dataset for the first pass
    let mut all_results_this_pass: Vec<AdminFrame> = Vec::new(); // Collect all DFs from a pass

    loop {
        iteration_count += 1;
        debug!(iteration = iteration_count, "Main admin sequence iteration");
        all_results_this_pass.clear();

        // Determine which queries need processing and batch similar searches
        let search_batches = prepare_admin_search_batches(query_states, MAX_ADMIN_LEVELS);

        // If no batches to process and all queries complete, we're done
        if search_batches.is_empty() {
            let all_complete = query_states.iter().all(|qs| qs.admin_sequence_complete);
            if all_complete {
                debug!(
                    "All admin sequences are complete after {} iterations.",
                    iteration_count
                );
                break;
            }

            // Handle stuck queries by advancing them if needed
            let advanced_any = advance_stuck_queries(query_states, MAX_ADMIN_LEVELS);
            if !advanced_any {
                debug!(
                    "No queries could be advanced. Breaking loop to prevent infinite iteration."
                );
                break;
            }
            continue;
        }

        debug!(
            "Main admin sequence pass {}: Processing {} search batches in parallel.",
            iteration_count,
            search_batches.len()
        );

        // Execute searches in parallel and collect results
        let batch_results = execute_admin_search_batches(
            search_batches,
            &active_admin_lf_for_pass,
            admin_fts_index,
            admin_search_params,
        )?;

        // Process results and update query states
        for (query_indices, result_opt, min_level) in batch_results {
            if let Some(result_df) = &result_opt {
                all_results_this_pass.push(result_df.clone());
            }

            for &query_idx in &query_indices {
                let qs = &mut query_states[query_idx];

                if let Some(result_df) = &result_opt {
                    qs.results_for_this_query.push(Admin(result_df.clone()));
                    qs.last_successful_admin_context_df = Some(result_df.clone());
                    qs.min_level_from_last_success = min_level;
                }

                // Advance to next term
                qs.current_admin_term_idx += 1;
                if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                    qs.admin_sequence_complete = true;
                }
            }
        }
        // Update active_admin_lf_for_pass for the next iteration using parent links
        if !all_results_this_pass.is_empty() {
            active_admin_lf_for_pass = update_admin_filter_for_next_pass(
                &all_results_this_pass,
                admin_data_lf,
                original_admin_data_height,
            )?;
        }

        // Check if all admin sequences are complete
        if query_states.iter().all(|qs| qs.admin_sequence_complete) {
            debug!(
                "All admin sequences complete after {} iterations.",
                iteration_count
            );
            break;
        }
    }

    Ok(())
}

/// Processes proactive admin searches for place candidates
///
/// This function:
/// 1. Identifies queries with place candidates that should be searched as admin entities
/// 2. Groups similar searches for parallel execution
/// 3. Executes the searches and processes results
/// 4. Updates query states with the results
fn process_proactive_admin_searches(
    query_states: &mut [QueryState],
    config: &SearchConfig,
    admin_data_lf: &LazyFrame,
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_search_params: &AdminSearchParams,
) -> Result<()> {
    // Group similar proactive searches for parallel execution
    let mut proactive_batches = HashMap::new();

    for (idx, qs) in query_states.iter().enumerate() {
        if qs.proactive_admin_search_complete {
            continue;
        }

        // Skip proactive admin search for single-term queries to avoid confusion
        if qs.admin_terms_for_main_sequence.len() <= 1
            && qs.is_place_candidate_also_last_admin_term_in_sequence
        {
            debug!(
                "Skipping proactive admin search for single-term query '{}' - already processed as admin",
                qs.place_candidate_term
                    .as_ref()
                    .unwrap_or(&"<unknown>".to_string())
            );
            continue;
        }

        let should_run_proactive = qs.place_candidate_term.is_some()
            && config.attempt_place_candidate_as_admin_before_place_search
            && !qs.is_place_candidate_also_last_admin_term_in_sequence;

        if should_run_proactive && let Some(ref pct) = qs.place_candidate_term {
            let admin_start_level = qs.admin_terms_for_main_sequence.len() as u8;

            if admin_start_level < MAX_ADMIN_LEVELS as u8 {
                let admin_levels: Vec<u8> = (admin_start_level..(MAX_ADMIN_LEVELS as u8)).collect();

                if !admin_levels.is_empty() {
                    let context_sig =
                        generate_context_signature(qs.last_successful_admin_context_df.as_ref());
                    let batch_key = (pct.clone(), admin_levels.clone(), context_sig);

                    proactive_batches
                        .entry(batch_key)
                        .or_insert_with(|| AdminSearchBatch {
                            term: pct.clone(),
                            levels: admin_levels,
                            context_df: qs.last_successful_admin_context_df.clone(),
                            query_indices: Vec::new(),
                        })
                        .query_indices
                        .push(idx);
                }
            }
        }
    }

    let proactive_batches: Vec<_> = proactive_batches.into_values().collect();

    if !proactive_batches.is_empty() {
        info!(
            "Proactive admin search: Processing {} unique search batches in parallel.",
            proactive_batches.len()
        );

        let proactive_results = execute_admin_search_batches(
            proactive_batches,
            admin_data_lf,
            admin_fts_index,
            admin_search_params,
        )?;

        // Process results
        for (query_indices, result_opt, _) in proactive_results {
            if let Some(result_df) = &result_opt {
                for &query_idx in &query_indices {
                    let qs = &mut query_states[query_idx];
                    qs.results_for_this_query.push(Admin(result_df.clone()));
                    qs.last_successful_admin_context_df = Some(result_df.clone());
                }
            }
        }
    } else {
        debug!("No proactive admin searches to process.");
    }

    // Mark all queries as having completed proactive admin search
    for qs in query_states.iter_mut() {
        qs.proactive_admin_search_complete = true;
    }

    Ok(())
}

/// Processes place searches for all queries with place candidates
///
/// This function:
/// 1. Identifies queries with place candidates
/// 2. Groups similar searches for parallel execution
/// 3. Executes the searches and processes results
/// 4. Updates query states with the results
fn process_place_searches(
    query_states: &mut [QueryState],
    places_data_lf: &LazyFrame,
    places_fts_index: &FTSIndex<PlacesIndexDef>,
    place_search_params: &PlaceSearchParams,
) -> Result<()> {
    for (idx, qs) in query_states.iter().enumerate() {
        if let Some(ref context) = qs.last_successful_admin_context_df {
            debug!("Query {}: Admin context has {} rows", idx, context.height());
            // Log the actual admin results being used
            if let Ok(names) = context.column("name") {
                debug!("Query {}: Admin names: {:?}", idx, names);
            }
        }
    }
    // Group similar place searches for parallel execution
    let mut place_batches = HashMap::new();

    for (idx, qs) in query_states.iter().enumerate() {
        if qs.place_search_complete || qs.place_candidate_term.is_none() {
            continue;
        }

        // Skip place search for single-term queries to avoid confusion between admin and place results
        if qs.admin_terms_for_main_sequence.len() <= 1
            && qs.is_place_candidate_also_last_admin_term_in_sequence
        {
            debug!(
                "Skipping place search for single-term query '{}' - treating as admin-only search",
                qs.place_candidate_term.as_ref().unwrap()
            );
            continue;
        }

        let term = qs.place_candidate_term.as_ref().unwrap();
        let context_sig = generate_context_signature(qs.last_successful_admin_context_df.as_ref());

        place_batches
            .entry((term.clone(), context_sig))
            .or_insert_with(|| PlaceSearchBatch {
                term: term.clone(),
                context_df: qs.last_successful_admin_context_df.clone(),
                query_indices: Vec::new(),
            })
            .query_indices
            .push(idx);
    }

    let place_batches: Vec<_> = place_batches.into_values().collect();

    if !place_batches.is_empty() {
        info!(
            "Place search: Processing {} unique search batches in parallel.",
            place_batches.len()
        );

        // Execute place searches in parallel
        let place_results: Vec<_> = place_batches
            .into_par_iter()
            .map(|batch| {
                let PlaceSearchBatch {
                    term,
                    context_df,
                    query_indices,
                } = batch;

                debug!(
                    "Searching place term '{}' for {} queries.",
                    term,
                    query_indices.len()
                );

                let search_result = place_search_inner(
                    &term,
                    places_fts_index,
                    places_data_lf.clone(),
                    context_df,
                    place_search_params,
                );

                match search_result {
                    Ok(Some(result_df)) if !result_df.is_empty() => {
                        Ok((query_indices, Some(result_df)))
                    }
                    Ok(_) => Ok((query_indices, None)),
                    Err(e) => {
                        warn!("Error in place search for term '{}': {:?}", term, e);
                        Ok((query_indices, None))
                    }
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Process results
        for (query_indices, result_opt) in place_results {
            if let Some(result_df) = result_opt {
                for &query_idx in &query_indices {
                    query_states[query_idx]
                        .results_for_this_query
                        .push(Place(result_df.clone()));
                }
            }
        }
    } else {
        debug!("No place searches to process.");
    }

    // Mark all queries as having completed place search
    for qs in query_states.iter_mut() {
        qs.place_search_complete = true;
    }

    Ok(())
}

/// Reconstructs final results in the original order
///
/// This function:
/// 1. Maps query results back to their original input indices
/// 2. Handles invalid query placeholders
/// 3. Returns a vector of vectors of DataFrames in the original input order
fn reconstruct_results(
    query_states: &[QueryState],
    original_input_to_unique_id: &[usize],
    original_count: usize,
) -> Vec<Vec<SearchResult>> {
    let mut final_results = Vec::with_capacity(original_count);

    for &unique_id in original_input_to_unique_id {
        if unique_id == INVALID_QUERY_ID_PLACEHOLDER {
            final_results.push(Vec::new());
        } else {
            final_results.push(query_states[unique_id].results_for_this_query.clone());
        }
    }

    final_results
}

#[instrument(name = "Bulk Location Search", level = "info", skip_all, fields(num_batches = all_raw_input_batches.len()))]
pub fn bulk_location_search_inner(
    all_raw_input_batches: &[&[&str]],
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_data_lf: impl IntoLazy,
    places_fts_index: &FTSIndex<PlacesIndexDef>,
    places_data_lf: impl IntoLazy,
    config: &SearchConfig,
) -> Result<Vec<Vec<SearchResult>>> {
    let t_start = std::time::Instant::now();
    // Convert to LazyFrames once to avoid repeated conversion
    let admin_data_lf = admin_data_lf.lazy();
    let places_data_lf = places_data_lf.lazy();
    info!(
        "Starting Bulk Location Search for {} input batches.",
        all_raw_input_batches.len()
    );

    // --- 1. Deduplication and Initialization ---
    let dedup_start = std::time::Instant::now();
    let (mut query_states, original_input_to_unique_id) =
        prepare_query_states(all_raw_input_batches, config.max_sequential_admin_terms)?;

    info!(
        elapsed = ?dedup_start.elapsed(),
        unique_queries = query_states.len(),
        "Deduplicated {} inputs to {} unique queries.",
        all_raw_input_batches.len(),
        query_states.len()
    );

    // --- 2. Process main admin sequence with parallel batching ---
    let admin_seq_start = std::time::Instant::now();
    info!("Starting main admin sequence processing...");
    let admin_search_params = AdminSearchParams {
        limit: config.limit,
        all_cols: config.all_cols,
        fts_search_params: config.admin_fts_search_params,
        search_score_params: config.admin_search_score_params,
    };
    // --- 2. Iterative Admin Search (Main Sequence - Horizontal Passes) ---
    process_main_admin_sequence(
        &mut query_states,
        &admin_data_lf,
        admin_fts_index,
        &admin_search_params,
    )?;
    info!(
        elapsed_ms = ?admin_seq_start.elapsed(),
        "Main admin sequence processing complete."
    );

    // --- 3. Process proactive admin searches with parallel execution ---
    let proactive_start = std::time::Instant::now();
    info!("Starting proactive admin search for place candidates (batched).");

    process_proactive_admin_searches(
        &mut query_states,
        config,
        &admin_data_lf,
        admin_fts_index,
        &admin_search_params,
    )?;

    info!(
        elapsed_ms = ?proactive_start.elapsed(),
        "Proactive admin search complete."
    );

    // --- 4. Process place searches with parallel execution ---
    let place_search_start = std::time::Instant::now();
    info!("Starting final place search (batched & parallel).");

    let place_search_params = PlaceSearchParams {
        limit: config.limit,
        all_cols: config.all_cols,
        min_importance_tier: config.place_min_importance_tier,
        center_lat: None,
        center_lon: None,
        fts_search_params: config.place_fts_search_params,
        search_score_params: config.place_search_score_params,
    };

    process_place_searches(
        &mut query_states,
        &places_data_lf,
        places_fts_index,
        &place_search_params,
    )?;

    info!(
        elapsed_ms = ?place_search_start.elapsed(),
        "Final place search complete."
    );

    // --- 5. Reconstruct Results for Original Inputs ---
    let reconstruct_start = std::time::Instant::now();
    info!("Reconstructing results for original inputs.");

    let final_bulk_results = reconstruct_results(
        &query_states,
        &original_input_to_unique_id,
        all_raw_input_batches.len(),
    );

    let avg_results_per_batch = if !final_bulk_results.is_empty() {
        final_bulk_results.iter().map(|v| v.len()).sum::<usize>() as f64
            / final_bulk_results.len() as f64
    } else {
        0.0
    };

    info!(
        elapsed_ms = ?reconstruct_start.elapsed(),
        avg_results_per_batch = avg_results_per_batch,
        "Reconstructed results complete."
    );

    info!(
        total_elapsed_ms = ?t_start.elapsed(),
        avg_ms_per_batch =
            (t_start.elapsed().as_millis() as f64) / (all_raw_input_batches.len().max(1) as f64),
        "Bulk Location Search finished. Returning results for {} original inputs.",
        final_bulk_results.len()
    );

    Ok(final_bulk_results)
}
