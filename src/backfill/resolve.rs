//! Location context resolution and administrative hierarchy backfilling.
//!
//! This module takes raw search results and enriches them with full administrative
//! context by looking up parent administrative entities. It resolves partial location
//! information into complete location hierarchies with country, state, county, etc.

use itertools::izip;
use polars::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tracing::{debug, debug_span, instrument, trace, warn};

use super::{LocationContext, ResolvedSearchResult, Result, entry::LocationEntry};
use crate::search::SearchResult;

pub type LocationResults<E> = Vec<ResolvedSearchResult<E>>;

/// Configuration for the location resolution process.
#[derive(Debug, Clone)]
pub struct ResolveConfig {
    /// Maximum number of candidates to resolve per query
    pub limit_per_query: usize,
}
impl Default for ResolveConfig {
    fn default() -> Self {
        Self {
            limit_per_query: 10,
        }
    }
}

/// Administrative codes extracted from a location for context building.
///
/// Contains all the administrative hierarchy codes (admin0 through admin4)
/// that will be used to look up parent administrative entities and build
/// the complete location context.
#[derive(Debug, Clone, Default)]
pub struct TargetLocationAdminCodes {
    _geoname_id: u32,
    pub admin0_code: Option<String>,
    pub admin1_code: Option<String>,
    pub admin2_code: Option<String>,
    pub admin3_code: Option<String>,
    pub admin4_code: Option<String>,
}

impl TargetLocationAdminCodes {
    pub fn from_df(df: &DataFrame) -> Result<Vec<Self>> {
        let cols = df
            .clone()
            .select([
                "geonameId",
                "admin0_code",
                "admin1_code",
                "admin2_code",
                "admin3_code",
                "admin4_code",
            ])?
            .take_columns();

        Ok(izip!(
            cols[0].u32()?,
            cols[1].str()?,
            cols[2].str()?,
            cols[3].str()?,
            cols[4].str()?,
            cols[5].str()?,
        )
        .map(
            |(geoname_id, admin0_code, admin1_code, admin2_code, admin3_code, admin4_code)| Self {
                _geoname_id: geoname_id.expect("geonameId should never be None"),
                admin0_code: admin0_code.map(|s| s.to_owned()),
                admin1_code: admin1_code.map(|s| s.to_owned()),
                admin2_code: admin2_code.map(|s| s.to_owned()),
                admin3_code: admin3_code.map(|s| s.to_owned()),
                admin4_code: admin4_code.map(|s| s.to_owned()),
            },
        )
        .collect::<Vec<_>>())
    }

    pub fn codes_hierarchy(&self) -> [(Option<&String>, &'static str); 5] {
        [
            (self.admin0_code.as_ref(), "admin0_code"),
            (self.admin1_code.as_ref(), "admin1_code"),
            (self.admin2_code.as_ref(), "admin2_code"),
            (self.admin3_code.as_ref(), "admin3_code"),
            (self.admin4_code.as_ref(), "admin4_code"),
        ]
    }
}

/// Query administrative data to find entities at specific hierarchy levels.
///
/// Uses the administrative codes to build precise filters that find the
/// administrative entity at each level (country, state, county, etc.) that
/// corresponds to the target location.
#[instrument(level = "trace", skip(admin_data_lf, filter_expr), fields(filter = format!("{:?}", filter_expr)))]
fn query_admin_level_entity_lazy<E: LocationEntry>(
    admin_data_lf: &LazyFrame,
    filter_expr: Expr,
) -> LazyFrame {
    admin_data_lf
        .clone()
        .filter(filter_expr)
        .select(E::field_names().iter().map(|s| col(*s)).collect::<Vec<_>>())
        //.sort(["admin_level"], Default::default())
        .limit(1) // Expecting only one match for a full code path at a specific level.
}

/// Build complete administrative context by querying for entities at each level.
///
/// Takes the administrative codes from target locations and builds complete
/// LocationContext objects by looking up the actual administrative entities
/// (countries, states, etc.) that correspond to those codes.
#[instrument(level = "trace", skip(admin_data_lf), fields(num_target_codes = ?target_codes.len()))]
fn backfill_administrative_context<E: LocationEntry>(
    target_codes: &[TargetLocationAdminCodes],
    admin_data_lf: &LazyFrame,
) -> Result<Vec<LocationContext<E>>> {
    let codes_hierarchys = target_codes
        .iter()
        .map(|tc| tc.codes_hierarchy())
        .collect::<Vec<_>>();

    let mut final_lf_for_all_levels: Vec<Vec<(usize, LazyFrame)>> = Vec::new();

    let mut cumulative_filter_parts: Vec<Expr> = Vec::new();
    for code_hierachys in codes_hierarchys.into_iter() {
        cumulative_filter_parts.clear();
        let mut final_lf_for_level: Vec<(usize, LazyFrame)> = Vec::new();
        for (admin_level_idx, (level_code_opt, level_code_name)) in
            code_hierachys.into_iter().enumerate()
        {
            let Some(current_level_code) = level_code_opt else {
                trace!(
                    level = admin_level_idx,
                    "No code provided in TargetLocationAdminCodes for this admin level."
                );
                continue;
            };
            cumulative_filter_parts.push(col(level_code_name).eq(lit(current_level_code.clone())));
            let mut level_specific_filter_parts =
                vec![col("admin_level").eq(lit(admin_level_idx as u8))]; // Use admin_level_idx
            level_specific_filter_parts.extend_from_slice(&cumulative_filter_parts);

            let final_filter_for_level = level_specific_filter_parts
                .into_iter()
                .reduce(|acc, expr| acc.and(expr))
                .expect("Filter parts should not be empty if code is present");
            let lf = query_admin_level_entity_lazy::<E>(admin_data_lf, final_filter_for_level);
            final_lf_for_level.push((admin_level_idx, lf));
        }
        final_lf_for_all_levels.push(final_lf_for_level);
    }
    let final_lf_len = final_lf_for_all_levels.len();

    // Track the positions of each LazyFrame
    let (position_map, flat_lazy_frames): (Vec<_>, Vec<_>) = final_lf_for_all_levels
        .into_iter()
        .enumerate()
        .flat_map(|(batch_idx, batch)| {
            batch
                .into_iter()
                .map(|(admin_level, lf)| ((batch_idx, admin_level), lf))
                .collect::<Vec<_>>()
        })
        .unzip();

    // Use collect_all on the flattened vector

    let collected_frames = {
        let _span = debug_span!("Collecting LazyFrames", num = flat_lazy_frames.len(),).entered();
        collect_all(flat_lazy_frames)?
    };

    // Rebuild the original structure with the collected frames
    let mut result: Vec<LocationContext<E>> = vec![LocationContext::<E>::default(); final_lf_len];
    for ((batch_idx, admin_level), df) in izip!(position_map, collected_frames) {
        if df.is_empty() {
            debug!(
                "No data found for admin level {} in batch {}. Skipping.",
                admin_level, batch_idx
            );
            continue;
        }
        let context = &mut result[batch_idx];
        assert_eq!(df.shape().0, 1);
        let mut entity_list = E::from_df(&df)?;
        let entity = entity_list.pop();
        match admin_level {
            0 => context.admin0 = entity,
            1 => context.admin1 = entity,
            2 => context.admin2 = entity,
            3 => context.admin3 = entity,
            4 => context.admin4 = entity,
            _ => unreachable!(),
        }
    }
    Ok(result)
}

/// Main entry point for resolving search candidates into complete location information.
///
/// Takes raw search results and enriches them with administrative hierarchy context.
/// This is where search results get transformed from simple matches into complete
/// location records with country, state, city context.
#[instrument(
    name = "Resolve Search Candidate",
    level = "info",
    skip(search_results, admin_data_lf),
    fields(num_batches = 0, limit_per_query = config.limit_per_query)
)]
pub fn resolve_search_candidate<E: LocationEntry>(
    search_results: Vec<SearchResult>,
    admin_data_lf: &LazyFrame,
    config: &ResolveConfig,
) -> Result<LocationResults<E>> {
    if search_results.is_empty() {
        debug!("No search results found.");
        return Ok(Vec::new());
    }
    let primary_candidates_df = match search_results.last() {
        Some(df) if !df.is_empty() => df.clone(),
        _ => {
            debug!("No suitable primary candidate DataFrame or it's empty.");
            return Ok(Vec::new());
        }
    };
    tracing::Span::current().record("num_batches", primary_candidates_df.height());

    let mut primary_candidates_df =
        primary_candidates_df.map(|df| df.head(Some(config.limit_per_query)));

    let target_codes = TargetLocationAdminCodes::from_df(&primary_candidates_df)?;
    let final_contexts = backfill_administrative_context::<E>(&target_codes, admin_data_lf)?;
    // Used to determine if the primary candidate is a place or not?
    let candidate_entities: Vec<E> = E::from_df(&primary_candidates_df)?;
    let scores = primary_candidates_df
        .pop()
        .expect("Last column should be the score column");
    let scores = scores.f64()?;

    let mut resolved = izip!(final_contexts, candidate_entities, scores)
        .map(|(mut final_context, original_candidate, score)| {
            if !final_context.candidate_already_in_context(&original_candidate) {
                final_context.place = Some(original_candidate)
            }
            ResolvedSearchResult {
                context: final_context,
                score: score.unwrap_or(0.0),
            }
        })
        .collect::<Vec<_>>();
    resolved.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(resolved)
}

/// Batch version of search candidate resolution.
///
/// Efficiently processes multiple resolution requests in parallel,
/// making it ideal for bulk data processing scenarios.
#[instrument(name="Resolve Search Candidate Batches",
    level = "info", skip(search_results_batches, admin_data_lf), fields(num_batches = search_results_batches.len(), limit_per_query = config.limit_per_query))]
pub fn resolve_search_candidate_batches<E: LocationEntry>(
    search_results_batches: Vec<Vec<SearchResult>>,
    admin_data_lf: &LazyFrame,
    config: &ResolveConfig,
) -> Result<Vec<LocationResults<E>>> {
    search_results_batches
        .into_par_iter()
        .enumerate()
        .map(|(i, search_results)| {
            let _search_span = debug_span!("Resolve Search Candidate Batches", batch = i).entered();
            resolve_search_candidate::<E>(search_results, admin_data_lf, config)
        })
        .collect::<Result<Vec<_>>>()
}
