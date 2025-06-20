//! Administrative entity search for countries, states, provinces, and local divisions.
//!
//! This module handles searching through administrative hierarchies from country level
//! (admin level 0) down to local administrative divisions (admin level 4). It provides
//! sophisticated scoring that considers text relevance, population, feature importance,
//! and hierarchical relationships.

use std::ops::{Deref, DerefMut, Mul};

use ahash::AHashMap as HashMap;
use polars::prelude::*;
use tracing::{debug, info_span, instrument, trace, trace_span, warn};

use super::Result;
use crate::index::{AdminIndexDef, FTSIndex, FTSIndexSearchParams};

/// Wrapper around `DataFrame` for administrative search results.
///
/// Provides type safety and admin-specific operations on search result `DataFrames`.
/// Ensures the `DataFrame` contains expected columns for administrative entity data.
#[derive(Debug, Clone)]
pub struct AdminFrame(DataFrame);

impl AdminFrame {
    /// Creates a new `AdminFrame` from a `DataFrame`.
    /// This will panic if the `DataFrame` does not have the expected columns.
    pub fn new(df: DataFrame) -> Self {
        Self::from(df)
    }

    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(DataFrame) -> DataFrame,
    {
        Self(f(self.0))
    }

    /// Returns the underlying `DataFrame`.
    pub fn into_inner(self) -> DataFrame {
        self.0
    }
}
impl IntoLazy for AdminFrame {
    fn lazy(self) -> LazyFrame {
        self.0.lazy()
    }
}

impl Deref for AdminFrame {
    type Target = DataFrame;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for AdminFrame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl From<DataFrame> for AdminFrame {
    fn from(df: DataFrame) -> Self {
        // Ensure the DataFrame has the expected columns for admin search
        let expected_columns = [
            "admin_level",
            "geonameId",
            "name",
            "asciiname",
            "admin_level",
            "admin0_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "feature_class",
            "feature_code",
            "population",
            "latitude",
            "longitude",
        ];
        let df_cols = df
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();

        for col in expected_columns {
            assert!(
                df_cols.contains(&col),
                "DataFrame is missing expected column: {col}"
            );
        }
        Self(df)
    }
}
impl From<AdminFrame> for DataFrame {
    fn from(admin_frame: AdminFrame) -> Self {
        admin_frame.0
    }
}

/// Parameters for scoring administrative entities.
///
/// Controls how administrative entities are ranked by balancing text similarity,
/// population size, hierarchical relationships, and feature type importance.
/// The weights should sum to 1.0 for balanced scoring.
#[derive(Debug, Clone, Copy)]
pub struct SearchScoreAdminParams {
    /// Weight for text relevance score (default: 0.4)
    pub text_weight: f32,
    /// Weight for population importance (default: 0.25)
    pub pop_weight: f32,
    /// Weight for parent score influence (default: 0.20)
    pub parent_weight: f32,
    /// Weight for feature type importance (default: 0.15)
    pub feature_weight: f32,
}

impl Default for SearchScoreAdminParams {
    fn default() -> Self {
        Self {
            text_weight: 0.40,
            pop_weight: 0.25,
            parent_weight: 0.20,
            feature_weight: 0.15,
        }
    }
}

/// Calculate comprehensive scoring for administrative entity search results.
///
/// Combines multiple scoring factors:
/// - Text relevance from full-text search matching
/// - Population-based importance (larger = more important)
/// - Feature type relevance (countries > states > counties, etc.)
/// - Parent administrative context boost
/// - Geographic/political prominence factors
#[instrument(
    name = "Admin Search Score",
    level = "trace",
    skip_all,
    fields(search_term, score_col_name)
)]
fn search_score_admin(
    search_term: &str,
    lf: LazyFrame,
    score_col_name: &str,
    params: &SearchScoreAdminParams,
) -> Result<LazyFrame> {
    // ===== 1. Text relevance score =====
    let lf = super::common::text_relevance_score(lf, search_term).with_columns([
        // ===== 2. Population importance =====
        when(col("population").gt(0))
            .then(lit(1.0) - lit(1.0) / (lit(1.0) + (col("population").log(10.0) / lit(3.0))))
            .otherwise(lit(0.1))
            .alias("pop_score"),
        // ===== 3. Feature type importance =====
        when(col("feature_code").eq(lit("PCLI")))
            .then(lit(1.0))
            .when(col("feature_code").str().starts_with(lit("PCL")))
            .then(lit(0.9))
            .when(col("feature_code").eq(lit("PPLC")))
            .then(lit(0.95))
            .when(col("feature_code").str().starts_with(lit("PPL")))
            .then(lit(0.8))
            .when(col("feature_code").str().starts_with(lit("ADM1")))
            .then(lit(0.85))
            .when(col("feature_code").str().starts_with(lit("ADM2")))
            .then(lit(0.75))
            .when(col("feature_code").str().starts_with(lit("ADM3")))
            .then(lit(0.65))
            .when(col("feature_code").str().starts_with(lit("ADM")))
            .then(lit(0.55))
            .otherwise(lit(0.5))
            .alias("feature_score"),
        // ===== 4. Country/region prominence =====
        when(
            col("admin0_code").is_in(
                lit(Series::new(
                    "major_countries".into(),
                    &[
                        "US", "GB", "DE", "FR", "JP", "CN", "IN", "BR", "RU", "CA", "AU",
                    ],
                ))
                .implode(),
                false,
            ),
        )
        .then(lit(0.8))
        .otherwise(lit(0.5))
        .alias("country_score"),
    ]);

    // ===== 5. Parent score influence =====
    let lf = super::common::parent_factor(lf)?
        // ====== 6. Final score calculation =====
        .with_column(
            (col("text_score").mul(lit(params.text_weight))
                + col("pop_score").mul(lit(params.pop_weight))
                + col("feature_score").mul(lit(params.feature_weight))
                + col("parent_factor").mul(lit(params.parent_weight)))
            .alias("base_score"),
        )
        // Apply country prominence boost to the final score
        .with_column(
            (col("base_score") * (lit(0.7) + (lit(0.3) * col("country_score"))))
                .alias(score_col_name),
        )
        .sort(
            [score_col_name],
            SortMultipleOptions::new().with_order_descending(true),
        );
    Ok(lf)
}

/// Configuration parameters for administrative search operations.
#[derive(Debug, Clone, Copy)]
pub struct AdminSearchParams {
    /// The maximum number of results to return.
    pub limit: usize,
    /// If true, all columns from the input data will be included in the output.
    pub all_cols: bool,
    /// Parameters for the FTS search.
    pub fts_search_params: FTSIndexSearchParams,
    /// Parameters for scoring the search results.
    pub search_score_params: SearchScoreAdminParams,
}

impl Default for AdminSearchParams {
    fn default() -> Self {
        let limit = 10;
        Self {
            limit,
            all_cols: false,
            fts_search_params: FTSIndexSearchParams {
                limit: limit * 3,
                ..Default::default()
            },
            search_score_params: SearchScoreAdminParams::default(),
        }
    }
}

/// Main administrative entity search implementation.
///
/// Performs hierarchical search through administrative levels, applying context from
/// previous search results to progressively narrow down location scope. Handles
/// complex scenarios like "California, USA" by using USA context to filter California results.
///
/// The function coordinates between full-text search, hierarchical filtering,
/// parent-child relationship scoring, and final result ranking.
#[inline]
#[instrument(name = "Admin Level Search", level="debug", skip_all, fields(term=term, levels = ?levels, limit = params.limit, has_previous_result = previous_result.is_some()))]
pub fn admin_search_inner(
    term: &str,
    levels: &[u8],
    index: &FTSIndex<AdminIndexDef>,
    data: impl IntoLazy,
    previous_result: Option<AdminFrame>,
    params: &AdminSearchParams,
) -> Result<Option<AdminFrame>> {
    let data = data.lazy();
    let previous_result = previous_result.map(IntoLazy::lazy);

    let levels_series = Series::new("levels_filter".into(), levels);

    let (filtered_data_lf, join_cols_expr) = {
        let join_cols_expr =
            super::common::get_join_expr_from_previous_result(previous_result.as_ref())?;
        let filtered_data = match &previous_result {
            Some(prev_lf) if !join_cols_expr.is_empty() => {
                let prev_lf = prev_lf.clone().collect()?.lazy();
                super::common::filter_data_from_previous_results(
                    data,
                    prev_lf.clone(),
                    &join_cols_expr,
                )?
            }
            _ => data,
        };
        (
            filtered_data.filter(col("admin_level").is_in(lit(levels_series).implode(), false)),
            join_cols_expr,
        )
    };

    let gids_vec = {
        let _gid_span = trace_span!("collect_gids_for_fts_subset_search").entered();
        let gids_df = filtered_data_lf
            .clone()
            .select([col("geonameId")])
            .collect()?;
        let gid_series = gids_df.column("geonameId")?;

        if gid_series.is_empty() {
            warn!(
                "gid_series is empty after filtering for levels and previous results. Term: '{}', Levels: {:?}",
                term, levels
            );
            return Ok(None);
        }

        gid_series
            .cast(&DataType::UInt64)?
            .u64()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    };
    debug_assert!(!gids_vec.is_empty(), "gids should not be empty");
    trace!(num_gids_for_fts = gids_vec.len(), "GIDs collected for FTS");
    if gids_vec.is_empty() {
        warn!(term = term, "gids_vec for FTS is empty.");
        return Ok(None);
    }

    let (fts_gids, fts_scores): (Vec<_>, Vec<_>) = index
        .search_in_subset(term, &gids_vec, &params.fts_search_params)?
        .into_iter()
        .map(|(gid, score)| (gid as u32, score))
        .unzip();

    if fts_gids.is_empty() {
        warn!(term = term, "FTS returned no results");
        return Ok(None);
    }

    let base_fts_results_lf = df!(
        "geonameId" => fts_gids,
        "fts_score" => fts_scores,
    )?
    .lazy()
    .join(
        filtered_data_lf,
        [col("geonameId")],
        [col("geonameId")],
        JoinArgs::new(JoinType::Inner),
    )
    .sort(
        ["fts_score"],
        SortMultipleOptions::default().with_order_descending(true),
    )
    .select([
        // Select all columns except fts_score
        col("*").exclude(["fts_score"]),
        // Then select fts_score to place it at the end
        col("fts_score"),
    ]);
    // Join the FTS results with the original data
    let fts_results_lf_with_potential_parents = match previous_result {
        Some(prev_lf_original) if !join_cols_expr.is_empty() => {
            debug!("Processing previous_results for parent scores.");
            let mut prev_lf_processed = prev_lf_original.clone();
            let prev_schema = prev_lf_processed.clone().collect_schema()?;

            let mut renames_map: HashMap<String, String> = HashMap::new();
            let mut parent_score_col_exprs: Vec<Expr> = Vec::new();
            let score_pattern = regex::Regex::new(r"^score_admin_[0-4]$").unwrap();

            // Identify and rename parent score columns to avoid conflicts
            for name_idx in prev_schema.iter_names() {
                let name = name_idx.as_ref();
                if score_pattern.is_match(name) {
                    let new_name = format!("parent_{name}");
                    debug!(
                        "Will rename previous score column: {} -> {}",
                        name, new_name
                    );
                    renames_map.insert(name.to_string(), new_name.clone());
                    parent_score_col_exprs.push(col(&new_name));
                }
            }

            if !renames_map.is_empty() {
                let old_names: Vec<String> = renames_map.keys().cloned().collect();
                let new_names: Vec<String> = renames_map.values().cloned().collect();
                prev_lf_processed = prev_lf_processed.rename(&old_names, &new_names, true);
            }

            let mut lfs_to_concat: Vec<LazyFrame> = Vec::new();

            // Create joins at different hierarchy levels for comprehensive parent scoring
            for i in 1..=join_cols_expr.len() {
                let current_join_key_exprs: &[Expr] = &join_cols_expr[0..i];

                let mut cols_to_select_from_prev = current_join_key_exprs.to_vec();
                if !parent_score_col_exprs.is_empty() {
                    cols_to_select_from_prev.extend_from_slice(&parent_score_col_exprs);
                }

                let selected_previous_lf =
                    prev_lf_processed.clone().select(&cols_to_select_from_prev);

                let joined_lf = base_fts_results_lf.clone().join(
                    selected_previous_lf,
                    current_join_key_exprs,
                    current_join_key_exprs,
                    // TODO: Use JoinType::Semi and remove any code that becomes redundant because of that. (the select above and therefore `cols_to_select_from_prev` ect)
                    JoinArgs::new(JoinType::Semi),
                );
                lfs_to_concat.push(joined_lf);
            }

            if lfs_to_concat.is_empty() {
                debug!(
                    "No parent join paths generated, using base FTS results without parent scores."
                );
                base_fts_results_lf
            } else {
                debug!(
                    "Concatenating {} LazyFrames from different parent join paths.",
                    lfs_to_concat.len()
                );
                concat(&lfs_to_concat, UnionArgs::default())?
            }
        }
        _ => {
            debug!("No previous_result with join columns, using base FTS results.");
            base_fts_results_lf
        }
    };
    let fts_results_for_scoring = fts_results_lf_with_potential_parents;

    let min_level = levels.iter().min().copied().unwrap_or(0);
    debug_assert!(min_level < 5, "Level must be between 0 and 4");
    let score_col_name = format!("score_admin_{min_level}");

    let select_exprs = if params.all_cols {
        vec![col("*")]
    } else {
        vec![
            col("geonameId"),
            col("name"),
            col("asciiname"),
            col("admin0_code"),
            col("admin1_code"),
            col("admin2_code"),
            col("admin3_code"),
            col("admin4_code"),
            col("feature_class"),
            col("feature_code"),
            col("population"),
            col("latitude"),
            col("longitude"),
            col("admin_level"),
            col("fts_score"),
            col("fuzzy_score"),
            col("text_score"),
            col("^score_admin_[0-4]$"),
        ]
    };

    let scored_lf = search_score_admin(
        term,
        fts_results_for_scoring,
        &score_col_name,
        &params.search_score_params,
    )?;

    let output_lf = scored_lf
        .unique_stable(Some(vec!["geonameId".into()]), UniqueKeepStrategy::First)
        .limit(params.limit as u32)
        .select(&select_exprs);

    let output_df = {
        let _collect_span = info_span!("collect_final_admin_search_df").entered();
        let t_collect = std::time::Instant::now();
        let df = output_lf.collect()?;
        debug!(
            collection_time_seconds = t_collect.elapsed().as_secs_f32(),
            num_results = df.height(),
            "Final admin search DataFrame collected."
        );
        df
    };

    if output_df.is_empty() {
        debug!(
            "Admin search for term '{}' yielded no results after all processing.",
            term
        );
        Ok(None)
    } else {
        Ok(Some(AdminFrame(output_df)))
    }
}
