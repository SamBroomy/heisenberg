mod admin_search;
mod place_search;

use anyhow::{Context, Result};
use polars::prelude::*;
use std::ops::Mul;
use std::rc::Rc;
use tracing::{debug, instrument, warn};

pub use admin_search::{admin_search, get_admin_df};
pub use place_search::{get_places_df, place_search};

fn text_relevance_score(lf: LazyFrame, search_term: &str) -> LazyFrame {
    lf.with_column(
        ((col("fts_score") - col("fts_score").mean())
            / when(col("fts_score").std(0).gt(0.0))
                .then(col("fts_score").std(0))
                .otherwise(1.0))
        .alias("z_score"),
    )
    .with_column((lit(1.0) / (lit(1.0) + col("z_score").mul(lit(-1.5)).exp())).alias("text_score"))
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
}

fn parent_factor(lf: LazyFrame) -> Result<LazyFrame> {
    let parent_score_re = regex::Regex::new(r"^parent_adjusted_score_[0-4]$").unwrap();
    let parent_score_exprs_for_mean = lf
        .clone()
        .collect_schema()
        .context("Failed to collect schema for parent score expressions")?
        .iter_names()
        .filter_map(|name| {
            if parent_score_re.is_match(name) {
                Some(col(name.clone()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let lf = if !parent_score_exprs_for_mean.is_empty() {
        lf.with_column(
            polars::lazy::dsl::mean_horizontal(parent_score_exprs_for_mean, true)
                .context("Failed to calculate mean for parent score expressions")?
                .fill_null(0.5_f32)
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
        lf.with_column(lit(0.5_f32).alias("parent_factor"))
    };
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

fn get_join_expr_from_previous_result(previous_result: Option<&LazyFrame>) -> Result<Vec<Expr>> {
    match previous_result {
        Some(prev_lf) => {
            //let prev_lf = dbg!(prev_lf.collect()?).lazy();
            get_join_keys(prev_lf)
        }
        None => Ok(vec![]),
    }
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
