use crate::search::{fts_search::FTSIndexSearchParams, FTSIndex, PlacesIndexDef};
use ahash::AHashMap as HashMap;
use anyhow::{Context, Result};
use itertools::Itertools;
use once_cell::sync::OnceCell;
use polars::prelude::*;
use std::ops::Mul;
use tracing::{debug, debug_span, info, info_span, instrument, trace, trace_span, warn};

static PLACES_DF_CACHE: OnceCell<LazyFrame> = OnceCell::new();

pub fn get_places_df() -> Result<&'static LazyFrame> {
    PLACES_DF_CACHE.get_or_try_init(|| {
        info!("Loading and collecting 'places_search.parquet' into memory for the first time...");
        let t_load = std::time::Instant::now();
        let df = LazyFrame::scan_parquet(
            "./data/processed/geonames/places_search.parquet",
            Default::default(),
        )?
        .collect()
        .map(|df| df.lazy())
        .map_err(anyhow::Error::from);
        info!(
            "'places_search.parquet' collected in {:.3} seconds",
            t_load.elapsed().as_secs_f32()
        );
        df
    })
}

const EARTH_RADIUS_KM: f64 = 6371.0;

/// Parameters for scoring places based on various factors.
/// The weights for each factor can be adjusted to change the scoring behaviors.
/// The weights should sum to 1.0 for a balanced score.
#[derive(Debug, Clone, Copy)]
pub struct SearchScorePlaceParams {
    /// Weight for the text relevance score. Default is `0.40`.
    pub text_weight: f32,
    /// Weight for the importance score. Default is `0.20`.
    pub importance_weight: f32,
    /// Weight for the feature type score. Default is `0.15`.
    pub feature_weight: f32,
    /// Weight for the parent admin score. Default is `0.15`.
    pub parent_admin_score_weight: f32,
    /// Weight for the feature type score. Default is `0.05`.
    pub distance_weight: f32,
}

impl Default for SearchScorePlaceParams {
    fn default() -> Self {
        Self {
            text_weight: 0.4,
            importance_weight: 0.20,
            feature_weight: 0.15,
            parent_admin_score_weight: 0.15,
            distance_weight: 0.05,
        }
    }
}

#[instrument(
    name = "Place Search Score",
    level = "trace",
    skip_all,
    fields(search_term, score_col_name)
)]
fn search_score_place(
    search_term: &str,
    lf: LazyFrame,
    center_lat: Option<f32>,
    center_lon: Option<f32>,
    score_col_name: &str,
    params: &SearchScorePlaceParams,
) -> Result<LazyFrame> {
    // ===== 1. Text relevance score (FTS) =====
    let lf = super::text_relevance_score(lf, search_term)
        // ===== 2. Importance score (pre-calculated) =====
        .with_column(
            col("importance_score")
                .clip(lit(0.0_f32), lit(1.0_f32))
                .alias("importance_norm"),
        )
        // ===== 3. Feature type scoring for places =====
        .with_column(
            when(col("feature_code").is_in(lit(Series::new(
                "major_capitals".into(),
                &["PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4"],
            ))))
            .then(lit(1.0_f32))
            .when(col("feature_code").is_in(lit(Series::new(
                "landmarks".into(),
                &["CSTL", "MNMT", "RUIN", "TOWR"],
            ))))
            .then(lit(0.95_f32))
            .when(col("feature_code").is_in(lit(Series::new(
                "cultural".into(),
                &["MUS", "THTR", "AMTH", "LIBR", "OPRA"],
            ))))
            .then(lit(0.9_f32))
            .when(col("feature_code").is_in(lit(Series::new(
                "populated".into(),
                &["PPL", "PPLF", "PPLS", "PPLX"],
            ))))
            .then(lit(0.85_f32))
            .when(col("feature_code").is_in(lit(Series::new(
                "transport".into(),
                &["AIRP", "RSTN", "PRT", "MAR"],
            ))))
            .then(lit(0.8_f32))
            .when(col("feature_code").is_in(lit(Series::new(
                "facilities_edu_med".into(),
                &["UNIV", "SCH", "HSP", "HTL", "RSRT"],
            ))))
            .then(lit(0.75_f32))
            .when(
                col("feature_code").is_in(lit(Series::new("commercial".into(), &["MALL", "MKT"]))),
            )
            .then(lit(0.7_f32))
            .when(col("feature_code").is_in(lit(Series::new(
                "religious".into(),
                &["CH", "MSQE", "TMPL", "SHRN"],
            ))))
            .then(lit(0.65_f32))
            .when(col("feature_code").is_in(lit(Series::new(
                "natural".into(),
                &["MT", "PK", "VLC", "ISL", "BCH", "LK", "BAY"],
            ))))
            .then(lit(0.6_f32))
            .otherwise(lit(0.3_f32))
            .alias("feature_score"),
        );

    // ===== 4. Distance score =====
    let lf = if let (Some(clat), Some(clon)) = (center_lat, center_lon) {
        let lat1 = col("latitude").cast(DataType::Float64).radians();
        let lon1 = col("longitude").cast(DataType::Float64).radians();
        let lat2 = lit(clat as f64).radians();
        let lon2 = lit(clon as f64).radians();

        let d_lat = (lat2 - lat1) / lit(2.0);
        let d_lon = (lon2 - lon1) / lit(2.0);

        let a = d_lat.sin().pow(2)
            + col("latitude").cast(DataType::Float64).radians().cos()
                * lit(clat as f64).radians().cos()
                * d_lon.sin().pow(2);

        let distance_km = lit(2.0_f64) * a.sqrt().arcsin() * lit(EARTH_RADIUS_KM);

        lf.with_column(distance_km.alias("distance_km"))
            .with_column(
                (lit(-1.0_f32) * col("distance_km").cast(DataType::Float32) / lit(50.0_f32))
                    .exp()
                    .alias("distance_score"),
            )
    } else {
        lf.with_column(lit(0.5_f32).alias("distance_score"))
    };

    // ===== 5. Parent Admin Score Factor =====
    let lf = super::parent_factor(lf)?
        // ===== 6. Final score calculation =====
        .with_column(
            (col("text_score").mul(lit(params.text_weight))
                + col("importance_norm").mul(lit(params.importance_weight))
                + col("feature_score").mul(lit(params.feature_weight))
                + col("distance_score").mul(lit(params.distance_weight))
                + col("parent_factor").mul(lit(params.parent_admin_score_weight)))
            .alias(score_col_name),
        )
        // ===== 7. Apply tier boost =====
        .with_column(
            col(score_col_name)
                * when(col("importance_tier").eq(lit(1_u8)))
                    .then(lit(1.2_f32))
                    .when(col("importance_tier").eq(lit(2_u8)))
                    .then(lit(1.1_f32))
                    .when(col("importance_tier").eq(lit(3_u8)))
                    .then(lit(1.0_f32))
                    .when(col("importance_tier").eq(lit(4_u8)))
                    .then(lit(0.9_f32))
                    .otherwise(lit(0.8_f32)) // Tier 5 and others
                    .alias(score_col_name),
        )
        .sort(
            [score_col_name],
            SortMultipleOptions::new().with_order_descending(true),
        );
    Ok(lf)
}

#[derive(Debug, Clone, Copy)]
pub struct PlaceSearchParams {
    /// Maximum number of results to return.
    pub limit: usize,
    /// If true, all columns from the input data will be included in the output.
    pub all_cols: bool,
    /// Minimum importance tier to include in the search.
    ///
    /// From 1 to 5 where 1 = most important, 5 = least important
    pub min_importance_tier: u8,
    /// Latitude of the center point for distance scoring.
    pub center_lat: Option<f32>,
    /// Longitude of the center point for distance scoring.
    pub center_lon: Option<f32>,
    /// Parameters for the FTS search.
    pub fts_search_params: FTSIndexSearchParams,
    /// Parameters for scoring the search results.
    pub search_score_params: SearchScorePlaceParams,
}
impl Default for PlaceSearchParams {
    fn default() -> Self {
        let limit = 20;
        Self {
            limit,
            all_cols: false,
            min_importance_tier: 4,
            center_lat: None,
            center_lon: None,
            fts_search_params: FTSIndexSearchParams {
                limit: limit * 3,
                fuzzy_search: true,
            },
            search_score_params: SearchScorePlaceParams::default(),
        }
    }
}
#[inline]
#[instrument(name = "Place Search", level="debug", skip_all, fields(term=term, limit = params.limit, has_previous_result = previous_result.is_some()))]
pub fn place_search_inner(
    term: &str,
    index: &FTSIndex<PlacesIndexDef>,
    data: impl IntoLazy,
    previous_result: Option<impl IntoLazy>,
    params: &PlaceSearchParams,
) -> Result<Option<DataFrame>> {
    let data = data.lazy();
    let previous_result = previous_result.map(|df| df.lazy());

    // --- Filter by importance tier first ---
    let data_filtered_by_tier =
        data.filter(col("importance_tier").lt_eq(lit(params.min_importance_tier.clamp(1, 5))));

    // --- Determine join columns and filter by previous admin results ---
    let (filtered_data_lf, join_cols_expr) = {
        let _filter_span = debug_span!("prepare_filtered_data_for_search").entered();

        let join_cols_expr = super::get_join_expr_from_previous_result(previous_result.as_ref())
            .context("Failed to get join columns from previous result")?;

        let filtered_data_for_fts = match &previous_result {
            Some(prev_lf) if !join_cols_expr.is_empty() => {
                debug!("Filtering place data based on previous admin results.");
                super::filter_data_from_previous_results(
                    data_filtered_by_tier,
                    prev_lf.clone(),
                    &join_cols_expr,
                )
                .context("Failed to filter place data from previous admin results")?
            }
            _ => {
                debug!("No previous admin results to filter by, or no join columns applicable.");
                data_filtered_by_tier
            }
        };
        (filtered_data_for_fts, join_cols_expr)
    };

    // --- Perform FTS search on the filtered data ---
    let gids_vec = {
        let _gid_span = trace_span!("collect_gids_for_fts_subset_search").entered();

        let gids_df = filtered_data_lf
            .clone()
            .select([col("geonameId")])
            .collect()
            .context("Failed to collect geonameIds for FTS subset search in places")?;
        let gid_series = gids_df.column("geonameId")?;

        if gid_series.is_empty() {
            warn!(
                "Place search: gid_series is empty after filtering. Term: '{}'",
                term
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

    // --- Join with previous_result to get parent admin scores ---
    let fts_results_lf_with_potential_parents = match previous_result {
        Some(ref prev_lf_original) if !join_cols_expr.is_empty() => {
            debug!("Place search: Processing previous_results for parent admin scores.");
            let mut prev_lf_processed = prev_lf_original.clone();
            let prev_schema = prev_lf_processed.clone().collect_schema().context(
                "Failed to collect schema from previous_result for renaming parent scores",
            )?;

            let mut renames_map: HashMap<String, String> = HashMap::new();
            let score_pattern = regex::Regex::new(r"^score_admin_[0-4]$").unwrap();

            for name_idx in prev_schema.iter_names() {
                let name = name_idx.as_ref();
                if score_pattern.is_match(name) {
                    let new_name = format!("parent_{}", name);
                    renames_map.insert(name.to_string(), new_name.clone());
                }
            }

            if !renames_map.is_empty() {
                let old_names: Vec<String> = renames_map.keys().cloned().collect();
                let new_names: Vec<String> = renames_map.values().cloned().collect();
                prev_lf_processed = prev_lf_processed.rename(&old_names, &new_names, true);
            }

            // Select only the join keys and the renamed parent_score_admin_ columns
            let mut cols_to_select_from_prev = join_cols_expr.clone();
            for new_parent_score_name in renames_map.values() {
                cols_to_select_from_prev.push(col(new_parent_score_name));
            }
            let cols_to_select_from_prev_str = cols_to_select_from_prev
                .iter()
                .map(|expr| match expr {
                    Expr::Column(name) => name.to_string(),
                    _ => panic!("Expected Expr::Column, got {:?}", expr),
                })
                .collect_vec();
            let prev_lf_for_join = prev_lf_processed.select(&cols_to_select_from_prev).unique(
                Some(cols_to_select_from_prev_str),
                UniqueKeepStrategy::First,
            );

            base_fts_results_lf.join(
                prev_lf_for_join,
                &join_cols_expr, // join keys
                &join_cols_expr, // join keys
                JoinArgs::new(JoinType::Left),
            )
        }
        _ => {
            debug!("Place search: No previous_result with join columns for parent scores.");
            base_fts_results_lf
        }
    };

    let (center_lat, center_lon) = (params.center_lat, params.center_lon);

    // --- Determine center_lat and center_lon for distance scoring ---
    let (final_center_lat, final_center_lon) = if center_lat.is_some() && center_lon.is_some() {
        (center_lat, center_lon)
    } else if let Some(prev_res_for_center) = &previous_result {
        // Attempt to get center from previous_result (e.g., mean of top result)
        let center_df_res = prev_res_for_center
            .clone()
            .select([
                col("latitude").mean().alias("center_lat"),
                col("longitude").mean().alias("center_lon"),
            ])
            .collect();
        match center_df_res {
            Ok(center_df) if !center_df.is_empty() => {
                let lat = center_df
                    .column("center_lat")
                    .ok()
                    .and_then(|s| s.f32().ok())
                    .and_then(|ca| ca.get(0));
                let lon = center_df
                    .column("center_lon")
                    .ok()
                    .and_then(|s| s.f32().ok())
                    .and_then(|ca| ca.get(0));
                if lat.is_some() && lon.is_some() {
                    debug!(
                        "Using center point from previous admin results: ({:?}, {:?})",
                        lat, lon
                    );
                }
                (lat, lon)
            }
            _ => (None, None),
        }
    } else {
        (None, None)
    };

    // --- Score the results ---
    let score_col_name = "score_place";
    let scored_lf = search_score_place(
        term,
        fts_results_lf_with_potential_parents,
        final_center_lat,
        final_center_lon,
        score_col_name,
        &params.search_score_params,
    )
    .context("Failed to score places")?;

    // --- Select final columns ---
    let final_select_exprs = if params.all_cols {
        vec![col("*")]
    } else {
        // Define your standard set of columns for place search results
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
            col("latitude"),
            col("longitude"),
            col("population"),
            col("importance_score"), // Original importance
            col("importance_tier"),
            col("fts_score"),
            col("text_score"),
            col("^parent_score_admin_[0-4]$"), // Include parent scores for inspection
            col(score_col_name),               // The final calculated place_score
        ]
    };

    let output_lf = scored_lf // Already sorted by search_score_place
        .unique_stable(Some(vec!["geonameId".into()]), UniqueKeepStrategy::First) // Keep best score for each place
        .limit(params.limit as u32)
        .select(&final_select_exprs);

    let output_df = {
        let _collect_span = info_span!("collect_final_admin_search_df").entered();
        let t_collect = std::time::Instant::now();
        let df = output_lf
            .collect()
            .context("Failed to collect final output DataFrame")?;
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
        Ok(Some(output_df))
    }
}
