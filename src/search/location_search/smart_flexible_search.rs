use crate::index::{AdminIndexDef, FTSIndex, FTSIndexSearchParams, PlacesIndexDef};
use crate::search::location_search::admin_search::{
    admin_search_inner, AdminSearchParams, SearchScoreAdminParams,
};
use crate::search::location_search::place_search::{
    place_search_inner, PlaceSearchParams, SearchScorePlaceParams,
};
use ahash::AHashMap as HashMap;
use ahash::AHasher as DefaultHasher;
use anyhow::{Context, Result};
use polars::prelude::*;
use rayon::prelude::*;
use std::cmp::{max, min};
use std::hash::{Hash, Hasher};
use tracing::{debug, info, instrument, warn};
const MAX_ADMIN_LEVELS_COUNT: usize = 5; // Admin levels 0 through 4

#[derive(Debug, Clone)]
pub struct SmartFlexibleSearchConfig {
    pub limit: usize,
    pub all_cols: bool,
    pub max_sequential_admin_terms: usize,
    pub attempt_place_candidate_as_admin_before_place_search: bool,
    pub admin_fts_search_params: FTSIndexSearchParams,
    pub admin_search_score_params: SearchScoreAdminParams,
    pub place_fts_search_params: FTSIndexSearchParams,
    pub place_search_score_params: SearchScorePlaceParams,
    pub place_min_importance_tier: u8,
}

impl Default for SmartFlexibleSearchConfig {
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

#[instrument(name = "Location Search", level = "info", skip_all, fields(search_terms = ?search_terms_raw))]
pub fn location_search_inner(
    search_terms_raw: &[&str],
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_data_lf: LazyFrame,
    places_fts_index: &FTSIndex<PlacesIndexDef>,
    places_data_lf: LazyFrame,
    config: &SmartFlexibleSearchConfig,
) -> Result<Vec<DataFrame>> {
    // --- 1. Input Cleaning & Term Definition ---
    let cleaned_terms = search_terms_raw
        .iter()
        .filter_map(|s| {
            if !s.trim().is_empty() {
                Some(s.trim().to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if cleaned_terms.is_empty() {
        warn!("No valid search terms provided after cleaning.");
        return Ok(Vec::new());
    }

    let mut place_candidate_term: Option<String> = None;
    let mut is_place_candidate_also_last_admin_term_in_sequence = false;

    let num_cleaned_terms = cleaned_terms.len();
    let num_admin_terms_in_sequence = min(num_cleaned_terms, config.max_sequential_admin_terms);

    let admin_terms_for_main_sequence: Vec<String> =
        cleaned_terms[0..num_admin_terms_in_sequence].to_vec();

    if num_cleaned_terms > num_admin_terms_in_sequence {
        let mut pct = cleaned_terms[num_admin_terms_in_sequence].clone();
        if num_cleaned_terms > num_admin_terms_in_sequence + 1 {
            let extra_terms = cleaned_terms[(num_admin_terms_in_sequence + 1)..].join(" ");
            pct = format!("{} {}", pct, extra_terms);
            info!(
                "Concatenated extra input terms into place candidate. New place candidate: '{}'",
                pct
            );
        }
        place_candidate_term = Some(pct);
    } else if !admin_terms_for_main_sequence.is_empty() {
        place_candidate_term = admin_terms_for_main_sequence.last().cloned();
        is_place_candidate_also_last_admin_term_in_sequence = true;
    }

    debug!(
        "Admin terms for main sequence: {:?}",
        admin_terms_for_main_sequence
    );
    if let Some(ref pct) = place_candidate_term {
        debug!(
            "Place candidate term: '{}' (Is also last admin in sequence: {})",
            pct, is_place_candidate_also_last_admin_term_in_sequence
        );
    } else {
        debug!("No distinct place candidate term identified.");
    }

    let mut all_found_results_list: Vec<DataFrame> = Vec::new();
    let mut last_successful_context: Option<DataFrame> = None;

    // --- 2. Iterative Flexible Admin Search (Main Sequence) ---
    if !admin_terms_for_main_sequence.is_empty() {
        let num_terms_in_main_seq = admin_terms_for_main_sequence.len();
        let empty_admin_slots = MAX_ADMIN_LEVELS_COUNT.saturating_sub(num_terms_in_main_seq);
        let mut min_level_from_last_success: Option<u8> = None;

        for (i, term_to_search) in admin_terms_for_main_sequence.iter().enumerate() {
            let natural_start_level = i as u8;
            let effective_start_level = match min_level_from_last_success {
                Some(min_prev) => max(natural_start_level, min_prev + 1),
                None => natural_start_level,
            };

            let current_search_window_end_level = min(
                (MAX_ADMIN_LEVELS_COUNT - 1) as u8, // Max level is 4
                natural_start_level + (empty_admin_slots as u8),
            );

            let current_search_levels: Vec<u8> =
                if effective_start_level <= current_search_window_end_level {
                    (effective_start_level..=current_search_window_end_level).collect()
                } else {
                    Vec::new()
                };

            if current_search_levels.is_empty() {
                warn!(
                    "Term '{}': No valid admin levels to search (effective range: {}-{}). Skipping.",
                    term_to_search, effective_start_level, current_search_window_end_level
                );
                continue;
            }

            debug!(
                "Searching main admin sequence term '{}' for admin levels {:?} with current context.",
                term_to_search, current_search_levels
            );

            let admin_params = AdminSearchParams {
                limit: config.limit,
                all_cols: config.all_cols,
                fts_search_params: config.admin_fts_search_params,
                search_score_params: config.admin_search_score_params,
            };

            match admin_search_inner(
                term_to_search,
                &current_search_levels,
                admin_fts_index,
                admin_data_lf.clone(),
                last_successful_context.clone(), // Clones Option<DataFrame>
                &admin_params,
            ) {
                Ok(Some(df)) if !df.is_empty() => {
                    info!(
                        "Found {} results for main admin term '{}' in levels {:?}.",
                        df.height(),
                        term_to_search,
                        current_search_levels
                    );
                    all_found_results_list.push(df.clone());
                    last_successful_context = Some(df.clone());

                    // Update min_level_from_last_success
                    if let Ok(admin_level_series) = df.column("admin_level") {
                        if let Ok(casted_series) = admin_level_series.cast(&DataType::UInt8) {
                            min_level_from_last_success = casted_series.u8().unwrap().min();
                        } else {
                            warn!("Could not cast 'admin_level' to UInt8.");
                            min_level_from_last_success = None; // Reset if problematic
                        }
                    } else {
                        warn!("'admin_level' column not found in admin search results.");
                        min_level_from_last_success = None; // Reset if not found
                    }
                }
                Ok(_) => {
                    debug!(
                        "No results for main admin term '{}' in levels {:?}.",
                        term_to_search, current_search_levels
                    );
                    // min_level_from_last_success remains from the *actual* last successful search.
                }
                Err(e) => {
                    warn!(
                        "Error searching for admin term '{}': {:?}",
                        term_to_search, e
                    );
                }
            }
        }
    }

    // --- 3. Proactive Admin Search for Place Candidate (if applicable) ---
    let should_run_proactive_admin_search = place_candidate_term.is_some()
        && config.attempt_place_candidate_as_admin_before_place_search
        && !is_place_candidate_also_last_admin_term_in_sequence;

    if should_run_proactive_admin_search {
        if let Some(ref pct_for_admin) = place_candidate_term {
            let additional_admin_start_level = admin_terms_for_main_sequence.len() as u8;

            if additional_admin_start_level < MAX_ADMIN_LEVELS_COUNT as u8 {
                let additional_admin_search_levels: Vec<u8> =
                    (additional_admin_start_level..(MAX_ADMIN_LEVELS_COUNT as u8)).collect();

                if !additional_admin_search_levels.is_empty() {
                    debug!(
                        "Proactively searching place candidate '{}' as ADMIN at levels {:?} using current context.",
                        pct_for_admin, additional_admin_search_levels
                    );
                    let admin_params = AdminSearchParams {
                        limit: config.limit,
                        all_cols: config.all_cols,
                        fts_search_params: config.admin_fts_search_params,
                        search_score_params: config.admin_search_score_params,
                    };
                    match admin_search_inner(
                        pct_for_admin,
                        &additional_admin_search_levels,
                        admin_fts_index,
                        admin_data_lf.clone(),
                        last_successful_context.clone(),
                        &admin_params,
                    ) {
                        Ok(Some(df)) if !df.is_empty() => {
                            info!(
                                "Found {} results for place candidate '{}' as proactive ADMIN in levels {:?}.",
                                df.height(), pct_for_admin, additional_admin_search_levels
                            );
                            all_found_results_list.push(df.clone());
                            last_successful_context = Some(df);
                        }
                        Ok(_) => {
                            debug!(
                                "No results for place candidate '{}' as proactive ADMIN in levels {:?}.",
                                pct_for_admin, additional_admin_search_levels
                            );
                        }
                        Err(e) => {
                            warn!(
                                "Error during proactive admin search for '{}': {:?}",
                                pct_for_admin, e
                            );
                        }
                    }
                }
            } else {
                debug!(
                    "Skipping proactive admin search for '{}': no subsequent admin levels available (start_level: {}).",
                    pct_for_admin, additional_admin_start_level
                );
            }
        }
    } else if place_candidate_term.is_some()
        && config.attempt_place_candidate_as_admin_before_place_search
        && is_place_candidate_also_last_admin_term_in_sequence
    {
        debug!(
            "Skipping proactive admin search for '{}': it was already processed as the last admin term in the main sequence.",
            place_candidate_term.as_ref().unwrap_or(&String::new())
        );
    }

    // --- 4. Final Place Search (if a place_candidate_term exists) ---
    if let Some(ref final_pct) = place_candidate_term {
        debug!(
            "Searching for place candidate '{}' as PLACE entity using final context.",
            final_pct
        );

        let place_params = PlaceSearchParams {
            limit: config.limit,
            all_cols: config.all_cols,
            min_importance_tier: config.place_min_importance_tier,
            center_lat: None, // place_search derives from previous_result if Some
            center_lon: None,
            fts_search_params: config.place_fts_search_params,
            search_score_params: config.place_search_score_params,
        };

        match place_search_inner(
            final_pct,
            places_fts_index,
            places_data_lf.clone(),
            last_successful_context.clone(),
            &place_params,
        ) {
            Ok(Some(df)) if !df.is_empty() => {
                info!(
                    "Found {} results for place candidate '{}' as PLACE.",
                    df.height(),
                    final_pct
                );
                all_found_results_list.push(df);
            }
            Ok(_) => {
                debug!("No results for place candidate '{}' as PLACE.", final_pct);
            }
            Err(e) => {
                warn!("Error searching for place '{}': {:?}", final_pct, e);
            }
        }
    }

    info!(
        "Smart Flexible Search finished. Returning {} DataFrame(s).",
        all_found_results_list.len()
    );
    Ok(all_found_results_list)
}

#[derive(Debug)]
struct BulkQueryState {
    unique_id: usize,
    //original_input_terms: Vec<String>, // For reference/debugging
    admin_terms_for_main_sequence: Vec<String>,
    place_candidate_term: Option<String>,
    is_place_candidate_also_last_admin_term_in_sequence: bool,
    current_admin_term_idx: usize,
    last_successful_admin_context_df: Option<DataFrame>,
    min_level_from_last_success: Option<u8>,
    admin_sequence_complete: bool,
    proactive_admin_search_complete: bool,
    place_search_complete: bool,
    results_for_this_query: Vec<DataFrame>,
}

fn calculate_target_levels_for_query_state(qs: &BulkQueryState) -> Vec<u8> {
    if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
        return vec![]; // No more terms to search
    }

    let natural_start_level = qs.current_admin_term_idx as u8;
    let effective_start_level = match qs.min_level_from_last_success {
        Some(min_prev) => max(natural_start_level, min_prev + 1),
        None => natural_start_level,
    };

    let num_terms_in_main_seq = qs.admin_terms_for_main_sequence.len();
    let empty_admin_slots = MAX_ADMIN_LEVELS_COUNT.saturating_sub(num_terms_in_main_seq);

    let current_search_window_end_level = min(
        (MAX_ADMIN_LEVELS_COUNT - 1) as u8, // Max level is 4
        natural_start_level + (empty_admin_slots as u8),
    );

    if effective_start_level <= current_search_window_end_level {
        (effective_start_level..=current_search_window_end_level).collect()
    } else {
        vec![] // No valid levels
    }
}

fn extract_min_admin_level_from_df(df: &DataFrame) -> Option<u8> {
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

fn generate_context_signature(context_df: Option<&DataFrame>) -> Option<u64> {
    context_df.and_then(|df| {
        df.column("geonameId")
            .ok()
            .and_then(|s| s.u32().ok()) // Assuming geonameId is u32, adjust if different
            .map(|ca| {
                let mut ids: Vec<u32> = ca.into_no_null_iter().collect();
                ids.sort_unstable();
                let mut hasher = DefaultHasher::default();
                ids.hash(&mut hasher);
                hasher.finish()
            })
    })
}

#[instrument(name = "Bulk Smart Flexible Search", level = "info", skip_all)]
pub fn bulk_location_search_inner(
    all_raw_input_batches: &[&[&str]],
    admin_fts_index: &FTSIndex<AdminIndexDef>,
    admin_data_lf: impl IntoLazy,
    places_fts_index: &FTSIndex<PlacesIndexDef>,
    places_data_lf: impl IntoLazy,
    config: &SmartFlexibleSearchConfig,
) -> Result<Vec<Vec<DataFrame>>> {
    let admin_data_lf = admin_data_lf.lazy();
    let places_data_lf = places_data_lf.lazy();
    info!(
        "Starting Bulk Location Search for {} input batches.",
        all_raw_input_batches.len()
    );

    // --- 1. Deduplication and Initialization ---
    let mut unique_queries_map: HashMap<Vec<String>, usize> = HashMap::new();
    let mut query_states: Vec<BulkQueryState> = Vec::new();
    let mut original_input_to_unique_id: Vec<usize> =
        Vec::with_capacity(all_raw_input_batches.len());
    const INVALID_QUERY_ID_PLACEHOLDER: usize = usize::MAX;

    for raw_input_batch in all_raw_input_batches {
        let cleaned_terms: Vec<String> = raw_input_batch
            .iter()
            .filter_map(|s| {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    Some(trimmed.to_string())
                } else {
                    None
                }
            })
            .collect();

        if cleaned_terms.is_empty() {
            original_input_to_unique_id.push(INVALID_QUERY_ID_PLACEHOLDER);
            continue;
        }

        let unique_id = *unique_queries_map
            .entry(cleaned_terms.clone())
            .or_insert_with(|| {
                let id = query_states.len();
                // (Logic for determining admin_terms, place_candidate_term, etc. as before)
                let num_cleaned_terms = cleaned_terms.len();
                let num_admin_terms_in_sequence =
                    min(num_cleaned_terms, config.max_sequential_admin_terms);

                let admin_terms_for_main_sequence: Vec<String> =
                    cleaned_terms[0..num_admin_terms_in_sequence].to_vec();

                let mut place_candidate_term: Option<String> = None;
                let mut is_place_candidate_also_last_admin_term_in_sequence = false;

                if num_cleaned_terms > num_admin_terms_in_sequence {
                    let mut pct = cleaned_terms[num_admin_terms_in_sequence].clone();
                    if num_cleaned_terms > num_admin_terms_in_sequence + 1 {
                        let extra_terms =
                            cleaned_terms[(num_admin_terms_in_sequence + 1)..].join(" ");
                        pct = format!("{} {}", pct, extra_terms);
                    }
                    place_candidate_term = Some(pct);
                } else if !admin_terms_for_main_sequence.is_empty() {
                    place_candidate_term = admin_terms_for_main_sequence.last().cloned();
                    is_place_candidate_also_last_admin_term_in_sequence = true;
                }

                query_states.push(BulkQueryState {
                    unique_id: id,
                    //original_input_terms: cleaned_terms, // Keep original for reference
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
                });
                id
            });
        original_input_to_unique_id.push(unique_id);
    }
    info!(
        "Deduplicated {} inputs to {} unique queries.",
        all_raw_input_batches.len(),
        query_states.len()
    );

    let admin_search_params = AdminSearchParams {
        limit: config.limit,
        all_cols: config.all_cols,
        fts_search_params: config.admin_fts_search_params,
        search_score_params: config.admin_search_score_params,
    };

    // --- 2. Iterative Admin Search (Main Sequence - Horizontal Passes) ---
    info!("Starting main admin sequence processing...");
    let original_admin_data_height = admin_data_lf.clone().collect()?.height();

    // active_admin_lf_for_pass will be the globally filtered LazyFrame for the current pass
    let mut active_admin_lf_for_pass = admin_data_lf.clone(); // Initialize with the full dataset for the first pass
    let mut all_results_this_pass_dfs: Vec<DataFrame> = Vec::new(); // Collect all DFs from a pass

    loop {
        let mut searches_to_batch_this_pass: HashMap<(String, Vec<u8>, Option<u64>), Vec<usize>> =
            HashMap::new();
        let mut advanced_any_query_this_pass = false;
        all_results_this_pass_dfs.clear(); // Clear for the current pass

        // First loop: Collect batches based on current query states
        for qs in &query_states {
            if qs.admin_sequence_complete
                || qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len()
            {
                continue;
            }
            advanced_any_query_this_pass = true;
            let term_to_search = &qs.admin_terms_for_main_sequence[qs.current_admin_term_idx];
            let target_levels = calculate_target_levels_for_query_state(qs);

            if target_levels.is_empty() {
                continue;
            }
            let context_signature =
                generate_context_signature(qs.last_successful_admin_context_df.as_ref());
            searches_to_batch_this_pass
                .entry((term_to_search.clone(), target_levels, context_signature))
                .or_default()
                .push(qs.unique_id);
        }

        // Second loop: Advance queries that couldn't form a batch key for this pass
        // (e.g., no valid target_levels for their current_admin_term_idx)
        for qs in &mut query_states {
            if qs.admin_sequence_complete {
                continue;
            }
            if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                qs.admin_sequence_complete = true;
                continue;
            }
            // Check if this query was part of any batch (not strictly necessary here if batching is exhaustive, but good for logic)
            // The primary condition for advancing here is if target_levels was empty.
            let target_levels = calculate_target_levels_for_query_state(qs);
            if target_levels.is_empty() && !qs.admin_terms_for_main_sequence.is_empty() {
                qs.current_admin_term_idx += 1;
                if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                    qs.admin_sequence_complete = true;
                }
            }
        }

        let all_admin_sequences_done_before_batch =
            query_states.iter().all(|qs| qs.admin_sequence_complete);

        if (!advanced_any_query_this_pass || searches_to_batch_this_pass.is_empty())
            && all_admin_sequences_done_before_batch
        {
            debug!("Main admin sequence loop: All queries complete or no batches to process.");
            break;
        }
        if searches_to_batch_this_pass.is_empty() && !all_admin_sequences_done_before_batch {
            debug!("Main admin sequence loop: No batches this pass, but some queries still active. Checking for advancement.");
            let mut made_progress_in_advancing_stuck_queries = false;
            for qs in &mut query_states {
                if !qs.admin_sequence_complete
                    && qs.current_admin_term_idx < qs.admin_terms_for_main_sequence.len()
                {
                    let target_levels = calculate_target_levels_for_query_state(qs);
                    if target_levels.is_empty() {
                        qs.current_admin_term_idx += 1;
                        made_progress_in_advancing_stuck_queries = true;
                        if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                            qs.admin_sequence_complete = true;
                        }
                    }
                }
            }
            if !made_progress_in_advancing_stuck_queries
                && query_states.iter().all(|qs| qs.admin_sequence_complete)
            {
                debug!(
                    "Main admin sequence loop: All queries now complete after checking stuck ones."
                );
                break;
            }
            if !made_progress_in_advancing_stuck_queries
                && !query_states.iter().all(|qs| qs.admin_sequence_complete)
            {
                debug!("Main admin sequence loop: No batches and no queries could be advanced. Breaking to prevent infinite loop.");
                break;
            }
            // If progress was made, continue to the next iteration to form new batches or break if all complete.
            if query_states.iter().all(|qs| qs.admin_sequence_complete) {
                break;
            }
            if searches_to_batch_this_pass.is_empty() {
                continue;
            } // If stuck queries advanced, new batches might form next iteration.
        }

        info!(
            "Main admin sequence pass: Preparing {} unique search batches for parallel execution.",
            searches_to_batch_this_pass.len()
        );
        // Prepare tasks for parallel execution
        let tasks_for_parallel_execution = searches_to_batch_this_pass
            .into_iter() // Consumes the HashMap
            .map(|((term, levels, _context_sig), batch_query_indices)| {
                // It's crucial that query_states is not mutably borrowed here.
                // We get the context_df_for_batch based on the representative query.
                let context_df_for_batch = if !batch_query_indices.is_empty() {
                    query_states[batch_query_indices[0]]
                        .last_successful_admin_context_df
                        .clone()
                } else {
                    None
                };
                (
                    term,
                    levels,
                    context_df_for_batch,
                    batch_query_indices,
                    active_admin_lf_for_pass.clone(), // Clone LazyFrame (cheap)
                    admin_fts_index,                  // FTSIndex is Sync, can be referenced
                    admin_search_params,              // AdminSearchParams is Copy
                )
            })
            .collect::<Vec<_>>();

        // Parallel execution of search tasks
        let batch_processing_results = tasks_for_parallel_execution
            .par_iter()
            .filter_map(
                |(
                    term,
                    levels,
                    context_df_for_batch,
                    batch_query_indices,
                    task_active_admin_lf,
                    task_admin_fts_index,
                    task_admin_search_params,
                )| {
                    if batch_query_indices.is_empty() {
                        return None;
                    }
                    debug!(
                        "Parallel task: Searching admin term '{}' for levels {:?} for {} queries.",
                        term,
                        levels,
                        batch_query_indices.len()
                    );
                    match admin_search_inner(
                        term,
                        levels,
                        task_admin_fts_index,
                        task_active_admin_lf.clone(),
                        context_df_for_batch.clone(),
                        task_admin_search_params,
                    ) {
                        Ok(Some(result_df)) if !result_df.is_empty() => {
                            let min_level = extract_min_admin_level_from_df(&result_df);
                            Some((
                                batch_query_indices.clone(), // Pass indices for update
                                Ok(Some(result_df)),
                                min_level,
                            ))
                        }
                        Ok(opt_df) => {
                            // Handles Ok(None) or Ok(Some(empty_df))
                            Some((batch_query_indices.clone(), Ok(opt_df), None))
                        }
                        Err(e) => {
                            warn!(
                                "Error in parallel admin search for term '{}': {:?}",
                                term, e
                            );
                            Some((batch_query_indices.clone(), Err(e), None))
                        }
                    }
                },
            )
            .collect::<Vec<_>>();

        // Sequential update of query_states and all_results_this_pass_dfs
        for (batch_query_indices, result_outcome, new_min_level_option) in batch_processing_results
        {
            match result_outcome {
                Ok(Some(result_df)) => {
                    all_results_this_pass_dfs.push(result_df.clone());
                    for query_idx_ref in &batch_query_indices {
                        let qs = &mut query_states[*query_idx_ref];
                        qs.results_for_this_query.push(result_df.clone());
                        qs.last_successful_admin_context_df = Some(result_df.clone());
                        qs.min_level_from_last_success = new_min_level_option;
                        qs.current_admin_term_idx += 1;
                        if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                            qs.admin_sequence_complete = true;
                        }
                    }
                }
                Ok(_) => {
                    // No results found for this batch
                    for query_idx_ref in &batch_query_indices {
                        let qs = &mut query_states[*query_idx_ref];
                        qs.current_admin_term_idx += 1;
                        if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                            qs.admin_sequence_complete = true;
                        }
                    }
                }
                Err(_e) => {
                    // Error already logged
                    for query_idx_ref in &batch_query_indices {
                        let qs = &mut query_states[*query_idx_ref];
                        qs.current_admin_term_idx += 1; // Advance on error
                        if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
                            qs.admin_sequence_complete = true;
                        }
                    }
                }
            }
        }

        // --- Update active_admin_lf_for_pass for the NEXT pass ---
        let mut next_pass_filter_exprs: Vec<Expr> = Vec::new();
        if !all_results_this_pass_dfs.is_empty() {
            match concat(
                all_results_this_pass_dfs
                    .iter()
                    .map(|df| df.clone().lazy())
                    .collect::<Vec<_>>(),
                UnionArgs {
                    diagonal: true,
                    ..Default::default()
                },
            )
            .and_then(|df| df.collect())
            {
                Ok(concatenated_results_df) if !concatenated_results_df.is_empty() => {
                    // We need 'code' and 'admin_level' of the found parents to filter children in the next pass.
                    // Assuming 'geonameId' is a unique identifier for entities.
                    match concatenated_results_df.unique_stable(
                        Some(&["geonameId".to_string()]),
                        UniqueKeepStrategy::First,
                        None,
                    ) {
                        Ok(unique_parents_df) if !unique_parents_df.is_empty() => {
                            if unique_parents_df.column("admin_level").is_ok() {
                                let admin_level_ca = unique_parents_df
                                    .column("admin_level")? // Propagate error if column not found
                                    .u8()?; // Propagate error if not u8
                                for i in 0..unique_parents_df.height() {
                                    if let Some(parent_level) = admin_level_ca.get(i) {
                                        let admin_code_col_name_for_parent = // Column in unique_parents_df to get the parent's specific code
                                            format!("admin{}_code", parent_level);
                                        if let Ok(parent_code_series_ref) = unique_parents_df
                                            .column(&admin_code_col_name_for_parent)
                                        {
                                            let parent_code_ca = parent_code_series_ref.str()?;
                                            if let Some(parent_code_val) = parent_code_ca.get(i) {
                                                // Children (in admin_data_lf) will have this parent_code_val
                                                // in their column corresponding to the parent's level.
                                                // e.g., if parent is level 0 (country), children (states) link via their admin0_code.
                                                if parent_level < (MAX_ADMIN_LEVELS_COUNT - 1) as u8
                                                {
                                                    // The column in admin_data_lf (children table) that links to this parent level's code
                                                    let child_link_col_name_in_admin_table =
                                                        format!("admin{}_code", parent_level);
                                                    next_pass_filter_exprs.push(
                                                        col(&child_link_col_name_in_admin_table)
                                                            .eq(lit(parent_code_val.to_string())),
                                                    );
                                                }
                                            } else {
                                                debug!("Parent code is null for {} at row {} (geonameId: {:?})",
                                                    admin_code_col_name_for_parent, i, unique_parents_df.column("geonameId").ok().and_then(|s| s.u64().ok().and_then(|ca| ca.get(i))));
                                            }
                                        } else {
                                            warn!("Admin code column '{}' not found in unique_parents_df for parent_level {}.", admin_code_col_name_for_parent, parent_level);
                                        }
                                    } else {
                                        debug!("Parent admin_level is null in unique_parents_df at row {} (geonameId: {:?})",
                                            i, unique_parents_df.column("geonameId").ok().and_then(|s| s.u64().ok().and_then(|ca| ca.get(i))));
                                    }
                                }
                            } else {
                                warn!("'admin_level' column missing in unique parent results. Cannot build next pass filter.");
                            }
                        }
                        Ok(_) => debug!("No unique parents found in current pass results."),
                        Err(e) => warn!("Error getting unique parents from pass results: {:?}", e),
                    }
                }
                Ok(_) => debug!("Concatenated results for the pass are empty."),
                Err(e) => warn!("Error concatenating results for the pass: {:?}", e),
            }
        }

        if !next_pass_filter_exprs.is_empty() {
            // Deduplicate expressions (optional, but can simplify large OR clauses)
            // A more robust deduplication might be needed if Expr does not derive a suitable Hash/Eq for HashMap keys.
            // For now, simple sort and dedup on string representation.
            next_pass_filter_exprs.sort_by_key(|e| format!("{:?}", e));
            next_pass_filter_exprs.dedup_by_key(|e| format!("{:?}", e));

            let combined_filter = next_pass_filter_exprs
                .into_iter()
                .reduce(|acc, e| acc.or(e))
                .expect("next_pass_filter_exprs was not empty"); // Should be safe due to check

            // Attempt to create the new globally filtered LazyFrame for the next pass
            // by applying the new filter to the *original* full admin dataset.
            let potential_next_active_lf = admin_data_lf.clone().filter(combined_filter);
            let t0 = std::time::Instant::now();
            let potential_next_active_lf = potential_next_active_lf
                .collect()
                .with_context(|| "Failed to collect potential_next_active_lf");
            warn!("Lf collect took {:.5}s", t0.elapsed().as_secs_f32());

            active_admin_lf_for_pass = match potential_next_active_lf {
                Ok(active_admin_df) if !active_admin_df.is_empty() => {
                    warn!(
                        "New global filter for the next admin pass based. Search space reduced by {} rows ({:.3}%)",
                        original_admin_data_height - active_admin_df.height(),
                        ((original_admin_data_height - active_admin_df.height()) as f64 / original_admin_data_height as f64) * 100.0
                    );

                    active_admin_df.lazy()
                }
                Ok(_) => {
                    // The new global filter resulted in an empty dataset
                    debug!("No results from the current pass after applying global filter.");
                    admin_data_lf.clone().filter(lit(false))
                }
                Err(e) => {
                    warn!("Error collecting potential_next_active_lf: {:?}. Setting active_admin_lf_for_pass to effectively empty.", e);
                    admin_data_lf.clone().filter(lit(false))
                }
            };

            debug!(
                "Applied global filter for the next admin pass based on {} unique parent links.",
                active_admin_lf_for_pass
                    .collect_schema()
                    .map_or(0, |s| s.len())
            ); // A bit of a proxy for filter complexity
        } else {
            // No new parent links were found in this pass to create a *new* global filter.
            // This means the global context for the next pass cannot be further refined based on *this* pass's results.
            // `active_admin_lf_for_pass` will retain its current state, preserving any global filtering from previous passes.
            debug!("No new parent links found in this pass to further refine active_admin_lf_for_pass. It remains as is for the next pass (if any).");
        }
        // End of main admin sequence loop iteration
    }
    info!("Main admin sequence processing complete.");

    // --- 3. Proactive Admin Search for Place Candidate (Batched) ---
    // (This section and step 4 will use admin_data_lf.clone() or places_data_lf.clone() directly,
    // as the `active_admin_lf_for_pass` optimization is specific to the main admin sequence's horizontal passes.)
    info!("Starting proactive admin search for place candidates (batched).");
    let mut proactive_searches_to_batch: HashMap<(String, Vec<u8>, Option<u64>), Vec<usize>> =
        HashMap::new();

    for qs in &query_states {
        if qs.proactive_admin_search_complete {
            continue;
        }
        let should_run_proactive = qs.place_candidate_term.is_some()
            && config.attempt_place_candidate_as_admin_before_place_search
            && !qs.is_place_candidate_also_last_admin_term_in_sequence;

        if should_run_proactive {
            if let Some(ref pct_for_admin) = qs.place_candidate_term {
                let additional_admin_start_level = qs.admin_terms_for_main_sequence.len() as u8;
                if additional_admin_start_level < MAX_ADMIN_LEVELS_COUNT as u8 {
                    let additional_admin_search_levels: Vec<u8> =
                        (additional_admin_start_level..(MAX_ADMIN_LEVELS_COUNT as u8)).collect();
                    if !additional_admin_search_levels.is_empty() {
                        let context_signature = generate_context_signature(
                            qs.last_successful_admin_context_df.as_ref(),
                        );
                        proactive_searches_to_batch
                            .entry((
                                pct_for_admin.clone(),
                                additional_admin_search_levels,
                                context_signature,
                            ))
                            .or_default()
                            .push(qs.unique_id);
                    }
                }
            }
        }
    }

    if !proactive_searches_to_batch.is_empty() {
        info!(
            "Proactive admin search: Preparing {} unique search batches for parallel execution.",
            proactive_searches_to_batch.len()
        );

        let proactive_tasks: Vec<_> = proactive_searches_to_batch
            .into_iter()
            .map(|((term, levels, _context_sig), batch_query_indices)| {
                let context_df_for_batch = if !batch_query_indices.is_empty() {
                    query_states[batch_query_indices[0]]
                        .last_successful_admin_context_df
                        .clone()
                } else {
                    None
                };
                (
                    term,
                    levels,
                    context_df_for_batch,
                    batch_query_indices,
                    // These are captured by the closure for par_iter
                    // admin_data_lf.clone(), // Use full admin data
                    // admin_fts_index,
                    // admin_search_params,
                )
            })
            .collect();

        let proactive_results: Vec<_> = proactive_tasks
            .par_iter()
            .filter_map(
                |(
                    term,
                    levels,
                    context_df_for_batch,
                    batch_query_indices,
                    // task_admin_data_lf, // These are implicitly captured or passed directly
                    // task_admin_fts_index,
                    // task_admin_search_params
                )| {
                    if batch_query_indices.is_empty() {
                        return None;
                    }
                    debug!(
                        "Parallel proactive admin search: term '{}', levels {:?} for {} queries.",
                        term,
                        levels,
                        batch_query_indices.len()
                    );
                    match admin_search_inner(
                        term,
                        levels,
                        admin_fts_index,       // Captured
                        admin_data_lf.clone(), // Captured and cloned
                        context_df_for_batch.clone(),
                        &admin_search_params, // Captured
                    ) {
                        Ok(Some(result_df)) if !result_df.is_empty() => {
                            Some((batch_query_indices.clone(), Ok(Some(result_df))))
                        }
                        Ok(opt_df) => Some((batch_query_indices.clone(), Ok(opt_df))),
                        Err(e) => {
                            warn!(
                                "Error in parallel proactive admin search for term '{}': {:?}",
                                term, e
                            );
                            Some((batch_query_indices.clone(), Err(e)))
                        }
                    }
                },
            )
            .collect();

        // Sequential update for proactive search results
        for (batch_query_indices, result_outcome) in proactive_results {
            match result_outcome {
                Ok(Some(result_df)) => {
                    // info!(
                    //     "Batch proactive admin search for term (deduced from indices) found {} results.",
                    //     result_df.height()
                    // );
                    for query_idx_ref in &batch_query_indices {
                        let qs = &mut query_states[*query_idx_ref];
                        qs.results_for_this_query.push(result_df.clone());
                        qs.last_successful_admin_context_df = Some(result_df.clone());
                    }
                }
                Ok(None) => {
                    // debug!("Batch proactive admin search for term (deduced) found no results.");
                }
                Err(_e) => { // Error already logged
                }
            }
        }
    }
    for qs in &mut query_states {
        qs.proactive_admin_search_complete = true;
    }
    info!("Proactive admin search complete.");

    // --- 4. Final Place Search (Batched) ---
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
    let mut final_place_searches_to_batch: HashMap<(String, Option<u64>), Vec<usize>> =
        HashMap::new();

    for qs in &query_states {
        if qs.place_search_complete {
            continue;
        }
        if let Some(ref final_pct) = qs.place_candidate_term {
            let context_signature =
                generate_context_signature(qs.last_successful_admin_context_df.as_ref());
            final_place_searches_to_batch
                .entry((final_pct.clone(), context_signature))
                .or_default()
                .push(qs.unique_id);
        }
    }

    if !final_place_searches_to_batch.is_empty() {
        info!(
            "Final place search: Preparing {} unique search batches for parallel execution.",
            final_place_searches_to_batch.len()
        );

        let place_tasks: Vec<_> = final_place_searches_to_batch
            .into_iter()
            .map(|((term, _context_sig), batch_query_indices)| {
                let context_df_for_batch = if !batch_query_indices.is_empty() {
                    query_states[batch_query_indices[0]]
                        .last_successful_admin_context_df
                        .clone()
                } else {
                    None
                };
                (
                    term,
                    context_df_for_batch,
                    batch_query_indices,
                    // places_data_lf.clone(), // Captured
                    // places_fts_index,      // Captured
                    // place_search_params,   // Captured
                )
            })
            .collect();

        let place_results: Vec<_> = place_tasks
            .par_iter()
            .filter_map(
                |(
                    term,
                    context_df_for_batch,
                    batch_query_indices,
                    // task_places_data_lf,
                    // task_places_fts_index,
                    // task_place_search_params
                )| {
                    if batch_query_indices.is_empty() {
                        return None;
                    }
                    debug!(
                        "Parallel final place search: term '{}' for {} queries.",
                        term,
                        batch_query_indices.len()
                    );
                    match place_search_inner(
                        term,
                        places_fts_index,       // Captured
                        places_data_lf.clone(), // Captured and cloned
                        context_df_for_batch.clone(),
                        &place_search_params, // Captured
                    ) {
                        Ok(Some(result_df)) if !result_df.is_empty() => {
                            Some((batch_query_indices.clone(), Ok(Some(result_df))))
                        }
                        Ok(opt_df) => Some((batch_query_indices.clone(), Ok(opt_df))),
                        Err(e) => {
                            warn!(
                                "Error in parallel final place search for term '{}': {:?}",
                                term, e
                            );
                            Some((batch_query_indices.clone(), Err(e)))
                        }
                    }
                },
            )
            .collect();

        // Sequential update for final place search results
        for (batch_query_indices, result_outcome) in place_results {
            match result_outcome {
                Ok(Some(result_df)) => {
                    // info!(
                    //     "Batch final place search for term (deduced) found {} results.",
                    //     result_df.height()
                    // );
                    for query_idx_ref in &batch_query_indices {
                        query_states[*query_idx_ref]
                            .results_for_this_query
                            .push(result_df.clone());
                    }
                }
                Ok(None) => {
                    // debug!("Batch final place search for term (deduced) found no results.");
                }
                Err(_e) => { // Error already logged
                }
            }
        }
    }
    for qs in &mut query_states {
        qs.place_search_complete = true;
    }
    info!("Final place search complete.");

    // --- 5. Reconstruct Results for Original Inputs ---
    info!("Reconstructing results for original inputs.");
    let mut final_bulk_results: Vec<Vec<DataFrame>> =
        Vec::with_capacity(all_raw_input_batches.len());
    for unique_id_or_placeholder in original_input_to_unique_id {
        if unique_id_or_placeholder == INVALID_QUERY_ID_PLACEHOLDER {
            final_bulk_results.push(Vec::new());
        } else {
            final_bulk_results.push(
                query_states[unique_id_or_placeholder]
                    .results_for_this_query
                    .clone(),
            );
        }
    }

    info!(
        "Bulk Location Search finished. Returning results for {} original inputs.",
        final_bulk_results.len()
    );
    Ok(final_bulk_results)
}

// pub fn bulk_smart_flexible_search(
//     all_raw_input_batches: &[&[&str]],
//     admin_fts_index: &FTSIndex<AdminIndexDef>,
//     admin_data_lf: LazyFrame, // Original, full admin data
//     places_fts_index: &FTSIndex<PlacesIndexDef>,
//     places_data_lf: LazyFrame,
//     config: &SmartFlexibleSearchConfig,
// ) -> Result<Vec<Vec<DataFrame>>> {
//     info!(
//         "Starting bulk_smart_flexible_search for {} input batches.",
//         all_raw_input_batches.len()
//     );

//     // --- 1. Deduplication and Initialization ---
//     let mut unique_queries_map: HashMap<Vec<String>, usize> = HashMap::new();
//     let mut query_states: Vec<BulkQueryState> = Vec::new();
//     let mut original_input_to_unique_id: Vec<usize> =
//         Vec::with_capacity(all_raw_input_batches.len());
//     const INVALID_QUERY_ID_PLACEHOLDER: usize = usize::MAX;

//     for raw_input_batch in all_raw_input_batches {
//         let cleaned_terms: Vec<String> = raw_input_batch
//             .iter()
//             .filter_map(|s| {
//                 let trimmed = s.trim();
//                 if !trimmed.is_empty() {
//                     Some(trimmed.to_string())
//                 } else {
//                     None
//                 }
//             })
//             .collect();

//         if cleaned_terms.is_empty() {
//             original_input_to_unique_id.push(INVALID_QUERY_ID_PLACEHOLDER);
//             continue;
//         }

//         let unique_id = *unique_queries_map
//             .entry(cleaned_terms.clone())
//             .or_insert_with(|| {
//                 let id = query_states.len();
//                 // (Logic for determining admin_terms, place_candidate_term, etc. as before)
//                 let num_cleaned_terms = cleaned_terms.len();
//                 let num_admin_terms_in_sequence =
//                     min(num_cleaned_terms, config.max_sequential_admin_terms);

//                 let admin_terms_for_main_sequence: Vec<String> =
//                     cleaned_terms[0..num_admin_terms_in_sequence].to_vec();

//                 let mut place_candidate_term: Option<String> = None;
//                 let mut is_place_candidate_also_last_admin_term_in_sequence = false;

//                 if num_cleaned_terms > num_admin_terms_in_sequence {
//                     let mut pct = cleaned_terms[num_admin_terms_in_sequence].clone();
//                     if num_cleaned_terms > num_admin_terms_in_sequence + 1 {
//                         let extra_terms =
//                             cleaned_terms[(num_admin_terms_in_sequence + 1)..].join(" ");
//                         pct = format!("{} {}", pct, extra_terms);
//                     }
//                     place_candidate_term = Some(pct);
//                 } else if !admin_terms_for_main_sequence.is_empty() {
//                     place_candidate_term = admin_terms_for_main_sequence.last().cloned();
//                     is_place_candidate_also_last_admin_term_in_sequence = true;
//                 }

//                 query_states.push(BulkQueryState {
//                     unique_id: id,
//                     original_input_terms: cleaned_terms, // Keep original for reference
//                     admin_terms_for_main_sequence,
//                     place_candidate_term,
//                     is_place_candidate_also_last_admin_term_in_sequence,
//                     current_admin_term_idx: 0,
//                     last_successful_admin_context_df: None,
//                     min_level_from_last_success: None,
//                     admin_sequence_complete: false,
//                     proactive_admin_search_complete: false,
//                     place_search_complete: false,
//                     results_for_this_query: Vec::new(),
//                 });
//                 id
//             });
//         original_input_to_unique_id.push(unique_id);
//     }
//     info!(
//         "Deduplicated {} inputs to {} unique queries.",
//         all_raw_input_batches.len(),
//         query_states.len()
//     );

//     let admin_search_params = AdminSearchParams {
//         limit: config.limit,
//         all_cols: config.all_cols,
//         fts_search_params: config.admin_fts_search_params,
//         search_score_params: config.admin_search_score_params,
//     };

//     // --- 2. Iterative Admin Search (Main Sequence - Horizontal Passes) ---
//     info!("Starting main admin sequence processing...");
//     let original_admin_data_height = admin_data_lf.clone().collect()?.height();

//     // active_admin_lf_for_pass will be the globally filtered LazyFrame for the current pass
//     let mut active_admin_lf_for_pass = admin_data_lf.clone(); // Initialize with the full dataset for the first pass
//     let mut all_results_this_pass_dfs: Vec<DataFrame> = Vec::new(); // Collect all DFs from a pass

//     loop {
//         let mut searches_to_batch_this_pass: HashMap<(String, Vec<u8>, Option<u64>), Vec<usize>> =
//             HashMap::new();
//         let mut advanced_any_query_this_pass = false;
//         all_results_this_pass_dfs.clear(); // Clear for the current pass

//         // First loop: Collect batches based on current query states
//         for qs in &query_states {
//             if qs.admin_sequence_complete
//                 || qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len()
//             {
//                 continue;
//             }
//             advanced_any_query_this_pass = true;
//             let term_to_search = &qs.admin_terms_for_main_sequence[qs.current_admin_term_idx];
//             let target_levels = calculate_target_levels_for_query_state(qs, config);

//             if target_levels.is_empty() {
//                 continue;
//             }
//             let context_signature =
//                 generate_context_signature(qs.last_successful_admin_context_df.as_ref());
//             searches_to_batch_this_pass
//                 .entry((term_to_search.clone(), target_levels, context_signature))
//                 .or_default()
//                 .push(qs.unique_id);
//         }

//         // Second loop: Advance queries that couldn't form a batch key for this pass
//         // (e.g., no valid target_levels for their current_admin_term_idx)
//         for qs in &mut query_states {
//             if qs.admin_sequence_complete {
//                 continue;
//             }
//             if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
//                 qs.admin_sequence_complete = true;
//                 continue;
//             }
//             // Check if this query was part of any batch (not strictly necessary here if batching is exhaustive, but good for logic)
//             // The primary condition for advancing here is if target_levels was empty.
//             let target_levels = calculate_target_levels_for_query_state(qs, config);
//             if target_levels.is_empty() && !qs.admin_terms_for_main_sequence.is_empty() {
//                 qs.current_admin_term_idx += 1;
//                 if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
//                     qs.admin_sequence_complete = true;
//                 }
//             }
//         }

//         if !advanced_any_query_this_pass || searches_to_batch_this_pass.is_empty() {
//             debug!("Main admin sequence loop: No more queries to advance or batches to process for this term index.");
//             // Check if all queries have completed their admin sequence
//             let all_admin_sequences_done = query_states.iter().all(|qs| qs.admin_sequence_complete);
//             if all_admin_sequences_done {
//                 debug!("All admin sequences are complete.");
//                 break; // Exit main admin loop
//             }
//             // If not all done, but no batches this pass, it might mean some queries are stuck or completed.
//             // The loop condition `advanced_any_query_this_pass` should handle breaking if no progress.
//             // If there are still queries not marked admin_sequence_complete but no batches,
//             // it implies they might have exhausted terms or levels. The outer loop should eventually break.
//             // One more check: if no batches but some queries are still active, we might need to advance them if they didn't get advanced above.
//             let mut made_progress_in_advancing_stuck_queries = false;
//             for qs in &mut query_states {
//                 if !qs.admin_sequence_complete
//                     && qs.current_admin_term_idx < qs.admin_terms_for_main_sequence.len()
//                 {
//                     let target_levels = calculate_target_levels_for_query_state(qs, config);
//                     if target_levels.is_empty() {
//                         // If it's still stuck with no levels
//                         qs.current_admin_term_idx += 1;
//                         made_progress_in_advancing_stuck_queries = true;
//                         if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
//                             qs.admin_sequence_complete = true;
//                         }
//                     }
//                 }
//             }
//             if !made_progress_in_advancing_stuck_queries && !all_admin_sequences_done {
//                 // If no batches and no queries could be advanced, then break to avoid infinite loop.
//                 debug!("Main admin sequence loop: No batches and no queries could be advanced. Breaking.");
//                 break;
//             }
//             if all_admin_sequences_done {
//                 break;
//             } // Re-check after potential advancement
//               // Continue to the next pass; active_admin_lf_for_pass will be updated below.
//         }
//         info!(
//             "Main admin sequence pass: Processing {} unique search batches.",
//             searches_to_batch_this_pass.len()
//         );

//         for ((term, levels, _context_sig), batch_of_query_indices) in searches_to_batch_this_pass {
//             if batch_of_query_indices.is_empty() {
//                 continue;
//             }
//             let representative_query_idx = batch_of_query_indices[0];
//             // Each batch uses its own specific context if available, but searches within the globally filtered active_admin_lf_for_pass
//             let context_df_for_batch = query_states[representative_query_idx]
//                 .last_successful_admin_context_df
//                 .clone();

//             debug!(
//                 "Batch searching admin term '{}' for levels {:?} (rep_idx: {}) for {} queries.",
//                 term,
//                 levels,
//                 representative_query_idx,
//                 batch_of_query_indices.len()
//             );
//             match admin_search(
//                 &term,
//                 &levels,
//                 admin_fts_index,
//                 active_admin_lf_for_pass.clone(), // Use the globally pre-filtered LF for this pass
//                 context_df_for_batch,             // Individual context for this batch
//                 &admin_search_params,
//             ) {
//                 Ok(Some(result_df)) if !result_df.is_empty() => {
//                     info!(
//                         "Batch admin search for term '{}' found {} results.",
//                         term,
//                         result_df.height()
//                     );
//                     all_results_this_pass_dfs.push(result_df.clone()); // Collext for next pass's global context
//                     let new_min_level = extract_min_admin_level_from_df(&result_df);
//                     for query_idx_ref in &batch_of_query_indices {
//                         let qs = &mut query_states[*query_idx_ref];
//                         qs.results_for_this_query.push(result_df.clone());
//                         qs.last_successful_admin_context_df = Some(result_df.clone());
//                         qs.min_level_from_last_success = new_min_level;
//                         qs.current_admin_term_idx += 1;
//                         if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
//                             qs.admin_sequence_complete = true;
//                         }
//                     }
//                 }
//                 Ok(_) => {
//                     debug!("Batch admin search for term '{}' found no results.", term);
//                     for query_idx_ref in &batch_of_query_indices {
//                         let qs = &mut query_states[*query_idx_ref];
//                         qs.current_admin_term_idx += 1;
//                         if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
//                             qs.admin_sequence_complete = true;
//                         }
//                     }
//                 }
//                 Err(e) => {
//                     warn!("Error in batch admin search for term '{}': {:?}", term, e);
//                     for query_idx_ref in &batch_of_query_indices {
//                         let qs = &mut query_states[*query_idx_ref];
//                         qs.current_admin_term_idx += 1;
//                         if qs.current_admin_term_idx >= qs.admin_terms_for_main_sequence.len() {
//                             qs.admin_sequence_complete = true;
//                         }
//                     }
//                 }
//             }
//         }

//         // --- Update active_admin_lf_for_pass for the NEXT pass ---
//         let mut next_pass_filter_exprs: Vec<Expr> = Vec::new();
//         if !all_results_this_pass_dfs.is_empty() {
//             match concat(
//                 all_results_this_pass_dfs
//                     .iter()
//                     .map(|df| df.clone().lazy())
//                     .collect::<Vec<_>>(),
//                 UnionArgs {
//                     diagonal: true,
//                     ..Default::default()
//                 },
//             )
//             .and_then(|df| df.collect())
//             {
//                 Ok(concatenated_results_df) if !concatenated_results_df.is_empty() => {
//                     // We need 'code' and 'admin_level' of the found parents to filter children in the next pass.
//                     // Assuming 'geonameId' is a unique identifier for entities.
//                     match concatenated_results_df.unique_stable(
//                         Some(&["geonameId".to_string()]),
//                         UniqueKeepStrategy::First,
//                         None,
//                     ) {
//                         Ok(unique_parents_df) if !unique_parents_df.is_empty() => {
//                             if unique_parents_df.column("admin_level").is_ok() {
//                                 let admin_level_ca = unique_parents_df
//                                     .column("admin_level")? // Propagate error if column not found
//                                     .u8()?; // Propagate error if not u8
//                                 for i in 0..unique_parents_df.height() {
//                                     if let Some(parent_level) = admin_level_ca.get(i) {
//                                         let admin_code_col_name_for_parent = // Column in unique_parents_df to get the parent's specific code
//                                             format!("admin{}_code", parent_level);
//                                         if let Ok(parent_code_series_ref) = unique_parents_df
//                                             .column(&admin_code_col_name_for_parent)
//                                         {
//                                             let parent_code_ca = parent_code_series_ref.str()?;
//                                             if let Some(parent_code_val) = parent_code_ca.get(i) {
//                                                 // Children (in admin_data_lf) will have this parent_code_val
//                                                 // in their column corresponding to the parent's level.
//                                                 // e.g., if parent is level 0 (country), children (states) link via their admin0_code.
//                                                 if parent_level < (MAX_ADMIN_LEVELS_COUNT - 1) as u8
//                                                 {
//                                                     // The column in admin_data_lf (children table) that links to this parent level's code
//                                                     let child_link_col_name_in_admin_table =
//                                                         format!("admin{}_code", parent_level);
//                                                     next_pass_filter_exprs.push(
//                                                         col(&child_link_col_name_in_admin_table)
//                                                             .eq(lit(parent_code_val.to_string())),
//                                                     );
//                                                 }
//                                             } else {
//                                                 debug!("Parent code is null for {} at row {} (geonameId: {:?})",
//                                                     admin_code_col_name_for_parent, i, unique_parents_df.column("geonameId").ok().and_then(|s| s.u64().ok().and_then(|ca| ca.get(i))));
//                                             }
//                                         } else {
//                                             warn!("Admin code column '{}' not found in unique_parents_df for parent_level {}.", admin_code_col_name_for_parent, parent_level);
//                                         }
//                                     } else {
//                                         debug!("Parent admin_level is null in unique_parents_df at row {} (geonameId: {:?})",
//                                             i, unique_parents_df.column("geonameId").ok().and_then(|s| s.u64().ok().and_then(|ca| ca.get(i))));
//                                     }
//                                 }
//                             } else {
//                                 warn!("'admin_level' column missing in unique parent results. Cannot build next pass filter.");
//                             }
//                         }
//                         Ok(_) => debug!("No unique parents found in current pass results."),
//                         Err(e) => warn!("Error getting unique parents from pass results: {:?}", e),
//                     }
//                 }
//                 Ok(_) => debug!("Concatenated results for the pass are empty."),
//                 Err(e) => warn!("Error concatenating results for the pass: {:?}", e),
//             }
//         }

//         if !next_pass_filter_exprs.is_empty() {
//             // Deduplicate expressions (optional, but can simplify large OR clauses)
//             // A more robust deduplication might be needed if Expr does not derive a suitable Hash/Eq for HashMap keys.
//             // For now, simple sort and dedup on string representation.
//             next_pass_filter_exprs.sort_by_key(|e| format!("{:?}", e));
//             next_pass_filter_exprs.dedup_by_key(|e| format!("{:?}", e));

//             let combined_filter = next_pass_filter_exprs
//                 .into_iter()
//                 .reduce(|acc, e| acc.or(e))
//                 .expect("next_pass_filter_exprs was not empty"); // Should be safe due to check

//             // Attempt to create the new globally filtered LazyFrame for the next pass
//             // by applying the new filter to the *original* full admin dataset.
//             let potential_next_active_lf = admin_data_lf.clone().filter(combined_filter);
//             let t0 = std::time::Instant::now();
//             let potential_next_active_lf = potential_next_active_lf
//                 .collect()
//                 .with_context(|| "Failed to collect potential_next_active_lf");
//             warn!("Lf collect took {:.5}s", t0.elapsed().as_secs_f32());

//             active_admin_lf_for_pass = match potential_next_active_lf {
//                 Ok(active_admin_df) if !active_admin_df.is_empty() => {
//                     warn!(
//                         "New global filter for the next admin pass based. Search space reduced by {} rows ({:.3}%)",
//                         original_admin_data_height - active_admin_df.height(),
//                         ((original_admin_data_height - active_admin_df.height()) as f64 / original_admin_data_height as f64) * 100.0
//                     );

//                     active_admin_df.lazy()
//                 }
//                 Ok(_) => {
//                     // The new global filter resulted in an empty dataset
//                     debug!("No results from the current pass after applying global filter.");
//                     admin_data_lf.clone().filter(lit(false))
//                 }
//                 Err(e) => {
//                     warn!("Error collecting potential_next_active_lf: {:?}. Setting active_admin_lf_for_pass to effectively empty.", e);
//                     admin_data_lf.clone().filter(lit(false))
//                 }
//             };

//             debug!(
//                 "Applied global filter for the next admin pass based on {} unique parent links.",
//                 active_admin_lf_for_pass
//                     .collect_schema()
//                     .map_or(0, |s| s.len())
//             ); // A bit of a proxy for filter complexity
//         } else {
//             // No new parent links were found in this pass to create a *new* global filter.
//             // This means the global context for the next pass cannot be further refined based on *this* pass's results.
//             // `active_admin_lf_for_pass` will retain its current state, preserving any global filtering from previous passes.
//             debug!("No new parent links found in this pass to further refine active_admin_lf_for_pass. It remains as is for the next pass (if any).");
//         }
//         // End of main admin sequence loop iteration
//     }
//     info!("Main admin sequence processing complete.");

//     // --- 3. Proactive Admin Search for Place Candidate (Batched) ---
//     // (This section and step 4 will use admin_data_lf.clone() or places_data_lf.clone() directly,
//     // as the `active_admin_lf_for_pass` optimization is specific to the main admin sequence's horizontal passes.)
//     info!("Starting proactive admin search for place candidates (batched).");
//     let mut proactive_searches_to_batch: HashMap<(String, Vec<u8>, Option<u64>), Vec<usize>> =
//         HashMap::new();
//     for qs in &query_states {
//         if qs.proactive_admin_search_complete {
//             continue;
//         }
//         let should_run_proactive = qs.place_candidate_term.is_some()
//             && config.attempt_place_candidate_as_admin_before_place_search
//             && !qs.is_place_candidate_also_last_admin_term_in_sequence;

//         if should_run_proactive {
//             if let Some(ref pct_for_admin) = qs.place_candidate_term {
//                 let additional_admin_start_level = qs.admin_terms_for_main_sequence.len() as u8;
//                 if additional_admin_start_level < MAX_ADMIN_LEVELS_COUNT as u8 {
//                     let additional_admin_search_levels: Vec<u8> =
//                         (additional_admin_start_level..(MAX_ADMIN_LEVELS_COUNT as u8)).collect();
//                     if !additional_admin_search_levels.is_empty() {
//                         let context_signature = generate_context_signature(
//                             qs.last_successful_admin_context_df.as_ref(),
//                         );
//                         proactive_searches_to_batch
//                             .entry((
//                                 pct_for_admin.clone(),
//                                 additional_admin_search_levels,
//                                 context_signature,
//                             ))
//                             .or_default()
//                             .push(qs.unique_id);
//                     }
//                 }
//             }
//         }
//     }

//     if !proactive_searches_to_batch.is_empty() {
//         info!(
//             "Proactive admin search: Processing {} unique search batches.",
//             proactive_searches_to_batch.len()
//         );
//         for ((term, levels, _context_sig), batch_of_query_indices) in proactive_searches_to_batch {
//             if batch_of_query_indices.is_empty() {
//                 continue;
//             }
//             let representative_query_idx = batch_of_query_indices[0];
//             let context_df_for_batch = query_states[representative_query_idx]
//                 .last_successful_admin_context_df
//                 .clone();

//             debug!("Batch proactive admin search for term '{}', levels {:?} (rep_idx: {}) for {} queries.", term, levels, representative_query_idx, batch_of_query_indices.len());
//             match admin_search(
//                 &term,
//                 &levels,
//                 admin_fts_index,
//                 admin_data_lf.clone(),
//                 context_df_for_batch,
//                 &admin_search_params,
//             ) {
//                 Ok(Some(result_df)) if !result_df.is_empty() => {
//                     info!(
//                         "Batch proactive admin search for term '{}' found {} results.",
//                         term,
//                         result_df.height()
//                     );
//                     for query_idx_ref in &batch_of_query_indices {
//                         let qs = &mut query_states[*query_idx_ref];
//                         qs.results_for_this_query.push(result_df.clone());
//                         qs.last_successful_admin_context_df = Some(result_df.clone());
//                     }
//                 }
//                 Ok(_) => {
//                     debug!(
//                         "Batch proactive admin search for term '{}' found no results.",
//                         term
//                     );
//                 }
//                 Err(e) => {
//                     warn!(
//                         "Error in batch proactive admin search for term '{}': {:?}",
//                         term, e
//                     );
//                 }
//             }
//         }
//     }
//     for qs in &mut query_states {
//         qs.proactive_admin_search_complete = true;
//     }
//     info!("Proactive admin search complete.");

//     // --- 4. Final Place Search (Batched) ---
//     info!("Starting final place search (batched).");
//     let place_search_params = PlaceSearchParams {
//         limit: config.limit,
//         all_cols: config.all_cols,
//         min_importance_tier: config.place_min_importance_tier,
//         center_lat: None,
//         center_lon: None,
//         fts_search_params: config.place_fts_search_params,
//         search_score_params: config.place_search_score_params,
//     };
//     let mut final_place_searches_to_batch: HashMap<(String, Option<u64>), Vec<usize>> =
//         HashMap::new();

//     for qs in &query_states {
//         if qs.place_search_complete {
//             continue;
//         }
//         if let Some(ref final_pct) = qs.place_candidate_term {
//             let context_signature =
//                 generate_context_signature(qs.last_successful_admin_context_df.as_ref());
//             final_place_searches_to_batch
//                 .entry((final_pct.clone(), context_signature))
//                 .or_default()
//                 .push(qs.unique_id);
//         }
//     }

//     if !final_place_searches_to_batch.is_empty() {
//         info!(
//             "Final place search: Processing {} unique search batches.",
//             final_place_searches_to_batch.len()
//         );
//         for ((term, _context_sig), batch_of_query_indices) in final_place_searches_to_batch {
//             if batch_of_query_indices.is_empty() {
//                 continue;
//             }
//             let representative_query_idx = batch_of_query_indices[0];
//             let context_df_for_batch = query_states[representative_query_idx]
//                 .last_successful_admin_context_df
//                 .clone();

//             debug!(
//                 "Batch final place search for term '{}' (rep_idx: {}) for {} queries.",
//                 term,
//                 representative_query_idx,
//                 batch_of_query_indices.len()
//             );
//             match place_search(
//                 &term,
//                 places_fts_index,
//                 places_data_lf.clone(),
//                 context_df_for_batch,
//                 &place_search_params,
//             ) {
//                 Ok(Some(result_df)) if !result_df.is_empty() => {
//                     info!(
//                         "Batch final place search for term '{}' found {} results.",
//                         term,
//                         result_df.height()
//                     );
//                     for query_idx_ref in &batch_of_query_indices {
//                         query_states[*query_idx_ref]
//                             .results_for_this_query
//                             .push(result_df.clone());
//                     }
//                 }
//                 Ok(_) => {
//                     debug!(
//                         "Batch final place search for term '{}' found no results.",
//                         term
//                     );
//                 }
//                 Err(e) => {
//                     warn!(
//                         "Error in batch final place search for term '{}': {:?}",
//                         term, e
//                     );
//                 }
//             }
//         }
//     }
//     for qs in &mut query_states {
//         qs.place_search_complete = true;
//     }
//     info!("Final place search complete.");

//     // --- 5. Reconstruct Results for Original Inputs ---
//     info!("Reconstructing results for original inputs.");
//     let mut final_bulk_results: Vec<Vec<DataFrame>> =
//         Vec::with_capacity(all_raw_input_batches.len());
//     for unique_id_or_placeholder in original_input_to_unique_id {
//         if unique_id_or_placeholder == INVALID_QUERY_ID_PLACEHOLDER {
//             final_bulk_results.push(Vec::new());
//         } else {
//             final_bulk_results.push(
//                 query_states[unique_id_or_placeholder]
//                     .results_for_this_query
//                     .clone(),
//             );
//         }
//     }

//     info!(
//         "bulk_smart_flexible_search finished. Returning results for {} original inputs.",
//         final_bulk_results.len()
//     );
//     Ok(final_bulk_results)
// }
