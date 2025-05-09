use crate::search::fts_search::{AdminIndexDef, FTSIndex, FTSIndexSearchParams, PlacesIndexDef};
use crate::search::location_search::admin_search::{
    admin_search, AdminSearchParams, SearchScoreAdminParams,
};
use crate::search::location_search::place_search::{
    place_search, PlaceSearchParams, SearchScorePlaceParams,
};
use anyhow::Result;
use polars::prelude::*;
use std::cmp::{max, min};
use tracing::{debug, info, warn};

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
            place_min_importance_tier: 4, // Default from PlaceSearchParams
        }
    }
}

pub fn smart_flexible_search(
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
        warn!("Smart Flexible Search: No valid search terms provided after cleaning.");
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
                "Smart Flexible Search: Concatenated extra input terms into place candidate. New place candidate: '{}'",
                pct
            );
        }
        place_candidate_term = Some(pct);
    } else if !admin_terms_for_main_sequence.is_empty() {
        place_candidate_term = admin_terms_for_main_sequence.last().cloned();
        is_place_candidate_also_last_admin_term_in_sequence = true;
    }

    debug!(
        "Smart Flexible Search: Admin terms for main sequence: {:?}",
        admin_terms_for_main_sequence
    );
    if let Some(ref pct) = place_candidate_term {
        debug!(
            "Smart Flexible Search: Place candidate term: '{}' (Is also last admin in sequence: {})",
            pct, is_place_candidate_also_last_admin_term_in_sequence
        );
    } else {
        debug!("Smart Flexible Search: No distinct place candidate term identified.");
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
                    "Smart Flexible Search: Term '{}': No valid admin levels to search (effective range: {}-{}). Skipping.",
                    term_to_search, effective_start_level, current_search_window_end_level
                );
                continue;
            }

            debug!(
                "Smart Flexible Search: Searching main admin sequence term '{}' for admin levels {:?} with current context.",
                term_to_search, current_search_levels
            );

            let admin_params = AdminSearchParams {
                limit: config.limit,
                all_cols: config.all_cols,
                fts_search_params: config.admin_fts_search_params,
                search_score_params: config.admin_search_score_params,
            };

            match admin_search(
                term_to_search,
                &current_search_levels,
                admin_fts_index,
                admin_data_lf.clone(),
                last_successful_context.clone(), // Clones Option<DataFrame>
                &admin_params,
            ) {
                Ok(Some(df)) if !df.is_empty() => {
                    info!(
                        "Smart Flexible Search: Found {} results for main admin term '{}' in levels {:?}.",
                        df.height(), term_to_search, current_search_levels
                    );
                    all_found_results_list.push(df.clone());
                    last_successful_context = Some(df.clone());

                    // Update min_level_from_last_success
                    if let Ok(admin_level_series) = df.column("admin_level") {
                        if let Ok(casted_series) = admin_level_series.cast(&DataType::UInt8) {
                            min_level_from_last_success = casted_series.u8().unwrap().min();
                        } else {
                            warn!("Smart Flexible Search: Could not cast 'admin_level' to UInt8.");
                            min_level_from_last_success = None; // Reset if problematic
                        }
                    } else {
                        warn!("Smart Flexible Search: 'admin_level' column not found in admin search results.");
                        min_level_from_last_success = None; // Reset if not found
                    }
                }
                Ok(_) => {
                    debug!(
                        "Smart Flexible Search: No results for main admin term '{}' in levels {:?}.",
                        term_to_search, current_search_levels
                    );
                    // min_level_from_last_success remains from the *actual* last successful search.
                }
                Err(e) => {
                    warn!(
                        "Smart Flexible Search: Error searching for admin term '{}': {:?}",
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
                        "Smart Flexible Search: Proactively searching place candidate '{}' as ADMIN at levels {:?} using current context.",
                        pct_for_admin, additional_admin_search_levels
                    );
                    let admin_params = AdminSearchParams {
                        limit: config.limit,
                        all_cols: config.all_cols,
                        fts_search_params: config.admin_fts_search_params,
                        search_score_params: config.admin_search_score_params,
                    };
                    match admin_search(
                        pct_for_admin,
                        &additional_admin_search_levels,
                        admin_fts_index,
                        admin_data_lf.clone(),
                        last_successful_context.clone(),
                        &admin_params,
                    ) {
                        Ok(Some(df)) if !df.is_empty() => {
                            info!(
                                "Smart Flexible Search: Found {} results for place candidate '{}' as proactive ADMIN in levels {:?}.",
                                df.height(), pct_for_admin, additional_admin_search_levels
                            );
                            all_found_results_list.push(df.clone());
                            last_successful_context = Some(df);
                        }
                        Ok(_) => {
                            debug!(
                                "Smart Flexible Search: No results for place candidate '{}' as proactive ADMIN in levels {:?}.",
                                pct_for_admin, additional_admin_search_levels
                            );
                        }
                        Err(e) => {
                            warn!(
                                "Smart Flexible Search: Error during proactive admin search for '{}': {:?}",
                                pct_for_admin, e
                            );
                        }
                    }
                }
            } else {
                debug!(
                    "Smart Flexible Search: Skipping proactive admin search for '{}': no subsequent admin levels available (start_level: {}).",
                    pct_for_admin, additional_admin_start_level
                );
            }
        }
    } else if place_candidate_term.is_some()
        && config.attempt_place_candidate_as_admin_before_place_search
        && is_place_candidate_also_last_admin_term_in_sequence
    {
        debug!(
            "Smart Flexible Search: Skipping proactive admin search for '{}': it was already processed as the last admin term in the main sequence.",
            place_candidate_term.as_ref().unwrap_or(&String::new())
        );
    }

    // --- 4. Final Place Search (if a place_candidate_term exists) ---
    if let Some(ref final_pct) = place_candidate_term {
        debug!(
            "Smart Flexible Search: Searching for place candidate '{}' as PLACE entity using final context.",
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

        match place_search(
            final_pct,
            places_fts_index,
            places_data_lf.clone(),
            last_successful_context.clone(),
            &place_params,
        ) {
            Ok(Some(df)) if !df.is_empty() => {
                info!(
                    "Smart Flexible Search: Found {} results for place candidate '{}' as PLACE.",
                    df.height(),
                    final_pct
                );
                all_found_results_list.push(df);
            }
            Ok(_) => {
                debug!(
                    "Smart Flexible Search: No results for place candidate '{}' as PLACE.",
                    final_pct
                );
            }
            Err(e) => {
                warn!(
                    "Smart Flexible Search: Error searching for place '{}': {:?}",
                    final_pct, e
                );
            }
        }
    }

    info!(
        "Smart Flexible Search finished. Returning {} DataFrame(s).",
        all_found_results_list.len()
    );
    Ok(all_found_results_list)
}
