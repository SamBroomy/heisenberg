use crate::{search::AdminIndexDef, FTSIndex};
use anyhow::{Context, Result};
use once_cell::sync::OnceCell;

use polars::prelude::*;
use std::{collections::HashMap, ops::Mul};
use tracing::{debug, info, instrument, warn};
static ADMIN_DF_CACHE: OnceCell<LazyFrame> = OnceCell::new();

pub fn get_admin_df() -> Result<&'static LazyFrame> {
    ADMIN_DF_CACHE.get_or_try_init(|| {
        info!("Loading and collecting 'admin_search.parquet' into memory for the first time...");
        let t_load = std::time::Instant::now();
        let df = LazyFrame::scan_parquet(
            "./data/processed/geonames/admin_search.parquet",
            Default::default(),
        )?
        .collect()
        .map(|df| df.lazy())
        .map_err(anyhow::Error::from);
        info!(
            "'admin_search.parquet' collected in {:.3} seconds",
            t_load.elapsed().as_secs_f32()
        );
        df
    })
}

fn search_score_admin(
    lf: LazyFrame,
    score_col_name: &str,
    text_weight: f32,
    pop_weight: f32,
    feature_weight: f32,
    parent_weight: f32,
    search_term: &str,
) -> Result<LazyFrame> {
    // ===== 1. Text relevance score =====
    let lf = super::text_relevance_score(lf, search_term)
        // ===== 2. Population importance =====
        .with_column(
            when(col("population").gt(0))
                .then(lit(1.0) - lit(1.0) / (lit(1.0) + (col("population").log(10.0) / lit(3))))
                .otherwise(lit(0.1))
                .alias("pop_score"),
        )
        // ===== 3. Feature type importance =====
        .with_column(
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
        )
        // ===== 4. Country/region prominence =====
        .with_column(
            when(col("admin0_code").is_in(lit(Series::new(
                "major_countries".into(),
                &[
                    "US", "GB", "DE", "FR", "JP", "CN", "IN", "BR", "RU", "CA", "AU",
                ],
            ))))
            .then(lit(0.8))
            .otherwise(lit(0.5))
            .alias("country_score"),
        );

    // ===== 5. Parent score influence =====
    let lf = super::parent_factor(lf)?
        // ====== 6. Final score calculation =====
        .with_column(
            (col("text_score").mul(lit(text_weight))
                + col("pop_score").mul(lit(pop_weight))
                + col("feature_score").mul(lit(feature_weight))
                + col("parent_factor").mul(lit(parent_weight)))
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

#[instrument(skip(data, index, previous_result, limit))]
pub fn admin_search(
    term: &str,
    levels: &[u8],
    data: LazyFrame,
    index: &FTSIndex<AdminIndexDef>,
    previous_result: Option<LazyFrame>,
    limit: Option<usize>,
    all_cols: bool,
) -> Result<Option<LazyFrame>> {
    let levels_series = Series::new("levels".into(), levels);
    let limit = limit.unwrap_or(20);

    let join_cols_expr = super::get_join_expr_from_previous_result(previous_result.as_ref())
        .context("Failed to get join columns from previous result")?;
    dbg!(&join_cols_expr);
    //dbg!(&levels);
    //let data = dbg!(data.collect()?).lazy();
    let filtered_data = match &previous_result {
        Some(prev_lf) if !join_cols_expr.is_empty() => {
            //let prev_lf = dbg!(prev_lf.collect()?).lazy();
            super::filter_data_from_previous_results(data, prev_lf.clone(), &join_cols_expr)
                .context("Failed to filter data from previous results")?
        }
        _ => data,
    };
    let data_for_level_filter = filtered_data.filter(col("admin_level").is_in(lit(levels_series)));
    dbg!(&data_for_level_filter.clone().collect()?);

    let gids_df = data_for_level_filter
        .clone()
        .select([col("geonameId")])
        .collect()?;
    let gid_series = gids_df.column("geonameId")?;

    if gid_series.is_empty() {
        warn!("gid_series is empty after filtering for levels and previous results. Term: '{}', Levels: {:?}", term, levels);
        return Ok(None);
    }

    let gids_vec: Vec<u64> = gid_series
        .cast(&DataType::UInt64)?
        .u64()?
        .into_iter()
        .flatten()
        .collect();
    debug_assert!(!gids_vec.is_empty(), "gids should not be empty");
    // Early return if gids is empty

    // Unzip results into separate vectors
    let (fts_gids, fts_scores): (Vec<_>, Vec<_>) = index
        .search_in_subset(term, &gids_vec, limit * 5, true)?
        .into_iter()
        .unzip();

    if fts_gids.is_empty() {
        warn!("FTS search returned no results for term: '{}'", term);
        return Ok(None);
    }

    let base_fts_results_lf = df!(
        "geonameId" => fts_gids,
        "fts_score" => fts_scores,
    )?
    .lazy()
    .join(
        data_for_level_filter, // Join with the already filtered data
        [col("geonameId")],
        [col("geonameId")],
        JoinArgs::new(JoinType::Inner), // Inner join to keep only FTS matches
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
            let prev_schema = prev_lf_processed
                .clone()
                .collect_schema()
                .context("Failed to collect schema from previous_result for renaming")?;

            let mut renames_map: HashMap<String, String> = HashMap::new();
            let mut parent_score_col_exprs: Vec<Expr> = Vec::new();
            let score_pattern = regex::Regex::new(r"^adjusted_score_[0-4]$").unwrap();

            for name_idx in prev_schema.iter_names() {
                let name = name_idx.as_ref();
                if score_pattern.is_match(name) {
                    let new_name = format!("parent_{}", name);
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
                    JoinArgs::new(JoinType::Left),
                );
                lfs_to_concat.push(joined_lf);
            }

            if !lfs_to_concat.is_empty() {
                debug!(
                    "Concatenating {} LazyFrames from different parent join paths.",
                    lfs_to_concat.len()
                );
                concat(&lfs_to_concat, UnionArgs::default())
                    .context("Failed to concat LazyFrames for parent scores")?
            } else {
                debug!(
                    "No parent join paths generated, using base FTS results without parent scores."
                );
                base_fts_results_lf // Fallback to base if no concat paths
            }
        }
        _ => {
            debug!("No previous_result with join columns, using base FTS results.");
            base_fts_results_lf
        }
    };
    // It's often good to let the lazy engine optimize fully, so collecting here might be optional
    // unless you observe specific query plan issues or very large intermediate results.
    // For now, let's proceed without the explicit collect().lazy() here.
    let fts_results_for_scoring = fts_results_lf_with_potential_parents;

    let min_level = levels.iter().min().copied().unwrap_or(0); // .copied() and fixed unwrap
    debug_assert!(min_level < 5, "Level must be between 0 and 4");
    let score_col_name = format!("adjusted_score_{}", min_level);

    let select_exprs = if all_cols {
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
            col("fts_score"),
            col("^adjusted_score_[0-4]$"),
        ]
    };

    let scored_lf = search_score_admin(
        fts_results_for_scoring,
        &score_col_name,
        0.35, // text_weight
        0.35, // pop_weight
        0.15, // feature_weight
        0.15, // parent_weight
        term,
    )?;

    let output_lf =
        scored_lf // Already sorted by search_score_admin
            .unique_stable(Some(vec!["geonameId".into()]), UniqueKeepStrategy::First)
            .limit(limit as u32) // limit is u32
            .select(&select_exprs); // Pass as slice

    Ok(Some(output_lf))
}
