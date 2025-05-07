use anyhow::{Context, Result};

use crate::FTSIndex;
use polars::prelude::*;
use std::{ops::Mul, rc::Rc};
use tracing::{debug, instrument, warn};

fn search_score_admin(
    lf: LazyFrame,
    score_col: &str,
    text_weight: f32,
    pop_weight: f32,
    feature_weight: f32,
    parent_weight: f32,
    search_term: &str,
) -> Result<LazyFrame> {
    // ===== 1. Text relevance score =====
    let lf = lf
        .with_column(
            ((col("fts_score") - col("fts_score").mean())
                / when(col("fts_score").std(0).gt(0.0))
                    .then(col("fts_score").std(0))
                    .otherwise(1.0))
            .alias("z_score"),
        )
        .with_column(
            (lit(1.0) / (lit(1.0) + col("z_score").mul(lit(-1.5)).exp())).alias("text_score"),
        )
        .with_column(
            when(
                col("name")
                    .str()
                    .to_lowercase()
                    .eq(lit(search_term.to_lowercase())),
            )
            .then(lit(1.0))
            .otherwise(col("text_score"))
            .clip(lit(0.0), lit(1.0))
            .alias("text_score"),
        )
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
    let parent_score_re = regex::Regex::new(r"^parent_adjusted_score_[0-4]$").unwrap();

    let input_schema = lf
        .clone()
        .collect_schema()
        .context("Failed to get schema for parent score check")?;
    let parent_score_exprs_for_mean = input_schema
        .iter_names()
        .filter_map(|name| {
            if parent_score_re.is_match(name) {
                debug!("Found adjusted_score_ column: {}", name);
                Some(col(name.clone()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let lf = if !parent_score_exprs_for_mean.is_empty() {
        lf.with_column(
            polars::lazy::dsl::mean_horizontal(parent_score_exprs_for_mean, true)?
                .fill_null(0.0_f32)
                .alias("average_parent_score"),
        )
        .with_column(
            when(col("average_parent_score").max().gt(lit(1e-9)))
                .then(col("average_parent_score") / (col("average_parent_score").max() + lit(1e-9)))
                .otherwise(lit(0.5))
                .clip(lit(0.0), lit(1.0))
                .alias("parent_factor"),
        )
    } else {
        debug!("No parent_adjusted_score_ columns found. Setting parent_factor to 0.5.");
        lf.with_column(lit(0.5).alias("parent_factor"))
    };

    // ====== 6. Final score calculation =====
    let lf = lf
        .with_column(
            (col("text_score").mul(lit(text_weight))
                + col("pop_score").mul(lit(pop_weight))
                + col("feature_score").mul(lit(feature_weight))
                + col("parent_factor").mul(lit(parent_weight)))
            .alias("base_score"),
        )
        // Apply country prominence boost to the final score
        .with_column(
            (col("base_score") * (lit(0.7) + (lit(0.3) * col("country_score")))).alias(score_col),
        )
        .sort(
            [score_col],
            SortMultipleOptions::new().with_order_descending(true),
        );
    Ok(lf)
}

#[instrument(skip(previous_result))]
fn get_join_keys(previous_result: &LazyFrame) -> Result<Vec<Expr>> {
    let mut join_on_cols_str = vec![];
    let join_keys = previous_result
        .clone()
        .select([col("^admin[0-4]_code$")])
        .unique(None, Default::default())
        .collect()
        .context("Failed to collect join keys")?;

    let column_names = join_keys
        .get_column_names()
        .iter()
        .map(|s| col(s.as_str()))
        .collect::<Vec<_>>();

    let jkl = join_keys.lazy();

    for col_name in column_names {
        let has_not_null = jkl
            .clone()
            .select([col_name.clone().is_not_null().any(true).alias("has_any")])
            .collect()
            .context("Failed to collect join keys not null")?
            .column("has_any")
            .context("Failed to get has_any column")?
            .bool()
            .context("Failed to get column as bool column")?
            .get(0)
            .unwrap_or(false);
        if has_not_null {
            debug!("{} is NOT null", col_name);
            join_on_cols_str.push(col_name);
        } else {
            debug!("{} is null", col_name);
        }
    }
    Ok(join_on_cols_str)
}

fn get_col_name_from_expr(expr: &Expr) -> Result<Rc<str>> {
    match expr {
        Expr::Column(name) => Ok(name.as_str().into()),
        _ => Err(anyhow::anyhow!("Expected Expr::Column, got {:?}", expr)),
    }
}

#[instrument(skip(data, previous_result_df))]
fn filter_data_from_previous_results(
    data: LazyFrame,
    previous_result_df: LazyFrame,
    join_key_exprs: &[Expr],
) -> Result<LazyFrame> {
    if join_key_exprs.is_empty() {
        debug!("No join_key_exprs provided, returning original data.");
        return Ok(data);
    }
    let join_key_names_arc: Vec<Rc<str>> = join_key_exprs
        .iter()
        .map(get_col_name_from_expr)
        .collect::<Result<_>>()
        .context("Failed to extract column names from join_key_exprs")?;

    // Select only the necessary columns for unique paths
    let unique_previous_paths_df = previous_result_df
        .select(join_key_exprs) // Select using the original expressions
        .unique(None, UniqueKeepStrategy::Any) // Get unique admin paths
        .collect()
        .context("Failed to collect unique paths from previous_result_df")?;

    if unique_previous_paths_df.is_empty() {
        debug!("No unique paths found in previous_result_df to filter by. Returning empty frame.");
        return Ok(data.limit(0)); // No paths mean no data matches criteria
    }
    let mut lfs_to_concat: Vec<LazyFrame> = Vec::new();

    for row_idx in 0..unique_previous_paths_df.height() {
        let mut current_path_filter: Option<Expr> = None;

        for key_name_arc in &join_key_names_arc {
            let key_name_str = key_name_arc.as_ref();
            let key_value_series =
                unique_previous_paths_df
                    .column(key_name_str)
                    .with_context(|| {
                        format!(
                            "Failed to get column '{}' from unique_previous_paths_df",
                            key_name_str
                        )
                    })?;

            let val_anyvalue = key_value_series.get(row_idx).with_context(|| {
                format!(
                    "Failed to get value at index {} for column '{}'",
                    row_idx, key_name_str
                )
            })?;

            if val_anyvalue.is_null() {
                // If a part of the path in previous_result is null (e.g., admin1_code for an ADM0),
                // we stop adding further constraints for this specific path.
                // The filter will be based on the non-null parts encountered so far.
                debug!(
                    "Path component {} is null for row {}, stopping constraint for this path.",
                    key_name_str, row_idx
                );
                break;
            } else {
                // Create a literal from AnyValue. This needs to handle various types if admin codes aren't always strings.
                // Assuming admin codes are strings for this example.
                let lit_val = match val_anyvalue {
                    AnyValue::String(s) => lit(s),
                    AnyValue::StringOwned(s_owned) => lit(s_owned),
                    ref av => {
                        // Attempt to create a series from AnyValue, then literal
                        // This is a robust way if types are mixed but known to Polars.
                        // However, direct lit construction is usually for concrete types.
                        // For simplicity, error out if not string, or handle more types.
                        warn!(
                            "Unsupported AnyValue type for admin code literal: {:?} for column {}",
                            av, key_name_str
                        );
                        // Create a series to then create a literal
                        // This is a bit of a workaround for direct AnyValue to lit.
                        let temp_series =
                            Series::new_null(key_name_str.into(), 0).cast(&av.dtype())?; // Get dtype
                        let temp_series = temp_series.extend_constant(av.clone(), 1)?;

                        lit(temp_series)
                        // return Err(anyhow::anyhow!(
                        //     "Unsupported AnyValue type {:?} for admin code in column {}",
                        //     av, key_name_str
                        // ));
                    }
                };

                let segment_filter = col(key_name_str).eq(lit_val);
                // debug!(
                //     "Adding segment filter for path row {}: {} == {:?}",
                //     row_idx, key_name_str, val_anyvalue
                // );

                if let Some(existing_filter) = current_path_filter.take() {
                    current_path_filter = Some(existing_filter.and(segment_filter));
                } else {
                    current_path_filter = Some(segment_filter);
                }
            }
        }

        if let Some(filter) = current_path_filter {
            lfs_to_concat.push(data.clone().filter(filter));
        } else {
            // This case might happen if all join_key_names resulted in null values for a path,
            // which implies no constraint from that path.
            // Depending on desired behavior, could add `data.clone()` (no filter) or skip.
            // Given the `break` logic, a `current_path_filter` should exist if any key was non-null.
            // If all keys were null for a path, it means that path is unconstrained.
            // However, get_join_keys ensures keys have *some* non-nulls in previous_result_df.
            // This path should likely not occur if unique_previous_paths_df is not empty.
            warn!("No filter constructed for path row {}, possibly all keys were null (should not happen if get_join_keys is effective).", row_idx);
        }
    }

    if lfs_to_concat.is_empty() {
        debug!("No data matched any constructed filter paths from previous results. Returning empty frame.");
        Ok(data.limit(0))
    } else {
        debug!(
            "Concatenating {} LazyFrames from filtered data chunks.",
            lfs_to_concat.len()
        );
        concat(&lfs_to_concat, UnionArgs::default())
            .context("Failed to concat LazyFrames from filtered data chunks")
            .map(|lf| lf.unique_stable(Some(vec!["geonameId".into()]), UniqueKeepStrategy::First))
    }
}

#[instrument(skip(data, index, previous_result, limit))]
pub fn admin_search(
    term: &str,
    levels: &[u8],
    data: LazyFrame,
    index: &FTSIndex,
    previous_result: Option<LazyFrame>,
    limit: Option<usize>,
    all_cols: bool,
) -> Result<Option<LazyFrame>> {
    let levels_series = Series::new("levels".into(), levels);
    let limit = limit.unwrap_or(20);

    let join_cols_expr = match &previous_result {
        Some(prev_lf) => {
            //let prev_lf = dbg!(prev_lf.collect()?).lazy();
            get_join_keys(prev_lf).context("Failed to get join keys")?
        }
        None => {
            vec![]
        }
    };
    dbg!(&join_cols_expr);
    //dbg!(&levels);
    //let data = dbg!(data.collect()?).lazy();
    let filtered_data = match &previous_result {
        Some(prev_lf) if !join_cols_expr.is_empty() => {
            //let prev_lf = dbg!(prev_lf.collect()?).lazy();
            filter_data_from_previous_results(data, prev_lf.clone(), &join_cols_expr)
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
        .search_in_subset(term, &gids_vec, limit * 5, false)? // Fetch more for ranking flexibility
        .iter()
        .map(|(_, id, score)| (*id, *score))
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

            let mut renames_map: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();
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
