use std::{ops::Mul, rc::Rc};

use itertools::izip;
use polars::prelude::*;
use rapidfuzz::fuzz::RatioBatchComparator;
use tracing::{debug, trace, warn};

use super::Result;

pub(super) fn text_relevance_score(lf: LazyFrame, search_term: &str) -> LazyFrame {
    let search_term_capture = search_term.to_string();
    lf.with_column(
        ((col("fts_score") - col("fts_score").mean())
            / when(col("fts_score").std(0).gt(0.0))
                .then(col("fts_score").std(0))
                .otherwise(1.0))
        .alias("z_score"),
    )
    .with_column(
        as_struct(vec![col("name"), col("alternatenames")])
            .map(
                move |s| {
                    let scorer = RatioBatchComparator::new(search_term_capture.chars());
                    let s = s.struct_().unwrap();
                    let name = s.field_by_name("name").unwrap();
                    let name = name.str().unwrap();
                    let altname = s.field_by_name("alternatenames").unwrap();
                    let altname = altname
                        .list()
                        .unwrap()
                        .into_iter()
                        .map(|series| {
                            series.map(|s| {
                                s.str()
                                    .unwrap()
                                    .into_iter()
                                    .map(|s| s.unwrap().chars().collect::<Vec<_>>())
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect::<Vec<_>>();

                    // let altname = altname.str().unwrap();
                    // let altname: Vec<Option<Vec<std::str::Chars<'_>>>> = altname
                    //     .iter()
                    //     .map(|s| s.map(|s| s.split(',').map(|s| s.chars()).collect::<Vec<_>>()))
                    //     .collect::<Vec<_>>();

                    let matches = izip!(name, altname)
                        .map(|(name, alternatenames)| {
                            let mut name = scorer
                                .similarity(name.expect("Should never be None").chars())
                                as f32;

                            let alternatenames = alternatenames.and_then(|s| {
                                s.into_iter()
                                    .map(|s| scorer.similarity(s) as f32)
                                    .max_by(|x, y| {
                                        x.abs().partial_cmp(&y.abs()).expect("Should never be NAN")
                                    })
                            });
                            if let Some(alternatenames) = alternatenames {
                                name = (name * 0.75) + (alternatenames * 0.25);
                            }
                            name
                        })
                        .collect::<Vec<_>>();
                    let out = Column::new("fuzzy_score".into(), matches);
                    Ok(Some(out))
                },
                GetOutput::from_type(DataType::Float32),
            )
            .alias("fuzzy_score"),
    )
    // Calculate the FTS contribution (0-1 score)
    .with_column(
        (lit(1.0) / (lit(1.0) + col("z_score").mul(lit(-1.5)).exp()))
            .alias("fts_contribution_score"),
    )
    // Normalize fuzzy_score to 0-1 range
    .with_column(
        when(col("fuzzy_score").max().neq(col("fuzzy_score").min()))
            .then(
                (col("fuzzy_score") - col("fuzzy_score").min())
                    / (col("fuzzy_score").max() - col("fuzzy_score").min()),
            )
            .otherwise(lit(0.75))
            .fill_null(0.0)
            .clip(lit(0.0), lit(1.0))
            .alias("fuzzy_contribution_score"),
    )
    .with_column(
        (lit(0.3) * col("fts_contribution_score") + lit(0.7) * col("fuzzy_contribution_score"))
            .alias("text_score"),
    )
}

pub(super) fn parent_factor(lf: LazyFrame) -> Result<LazyFrame> {
    let parent_score_re = regex::Regex::new(r"^parent_score_admin_[0-4]$").unwrap();
    let parent_score_exprs_for_mean = lf
        .clone()
        .collect_schema()?
        .iter_names()
        .filter_map(|name| {
            if parent_score_re.is_match(name) {
                Some(col(name.clone()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let lf = if parent_score_exprs_for_mean.is_empty() {
        lf.with_column(lit(0.5_f32).alias("parent_factor"))
    } else {
        lf.with_column(
            polars::lazy::dsl::mean_horizontal(parent_score_exprs_for_mean, true)?
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
    };
    Ok(lf)
}

fn get_join_keys(previous_result: &LazyFrame) -> Result<Vec<Expr>> {
    let mut join_on_cols_str = vec![];
    let join_keys = previous_result
        .clone()
        .select([col("^admin[0-4]_code$")])
        .unique(None, UniqueKeepStrategy::default())
        .collect()?;

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
            .collect()?
            .column("has_any")?
            .bool()?
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

pub(super) fn get_join_expr_from_previous_result(
    previous_result: Option<&LazyFrame>,
) -> Result<Vec<Expr>> {
    previous_result.map_or_else(|| Ok(vec![]), get_join_keys)
}

pub(super) fn get_col_name_from_expr(expr: &Expr) -> Result<Rc<str>> {
    match expr {
        Expr::Column(name) => Ok(name.as_str().into()),
        _ => Err(anyhow::anyhow!("Expected Expr::Column, got {:?}", expr).into()),
    }
}

pub(super) fn filter_data_from_previous_results(
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
        .collect::<Result<_>>()?;
    trace!(join_key_names = ?join_key_names_arc);

    // Select only the necessary columns for unique paths
    let unique_previous_paths_df = previous_result_df
        .select(join_key_exprs) // Select using the original expressions
        .unique(None, UniqueKeepStrategy::Any) // Get unique admin paths
        .collect()?;

    if unique_previous_paths_df.is_empty() {
        debug!("No unique paths found in previous_result_df to filter by. Returning empty frame.");
        return Ok(data.limit(0)); // No paths mean no data matches criteria
    }
    let mut lfs_to_concat: Vec<LazyFrame> = Vec::new();

    for row_idx in 0..unique_previous_paths_df.height() {
        let mut current_path_filter: Option<Expr> = None;

        for key_name_arc in &join_key_names_arc {
            let key_name_str = key_name_arc.as_ref();
            let key_value_series = unique_previous_paths_df.column(key_name_str)?;

            let val_anyvalue = key_value_series.get(row_idx)?;

            if val_anyvalue.is_null() {
                // If a part of the path in previous_result is null (e.g., admin1_code for an ADM0),
                // we stop adding further constraints for this specific path.
                // The filter will be based on the non-null parts encountered so far.
                debug!(
                    "Path component {} is null for row {}, stopping constraint for this path.",
                    key_name_str, row_idx
                );
                break;
            }
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
                    let temp_series = Series::new_null(key_name_str.into(), 0).cast(&av.dtype())?; // Get dtype
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

        if let Some(filter) = current_path_filter {
            trace!(filter = ?filter);
            lfs_to_concat.push(data.clone().filter(filter));
        } else {
            // This case might happen if all join_key_names resulted in null values for a path,
            // which implies no constraint from that path.
            // Depending on desired behavior, could add `data.clone()` (no filter) or skip.
            // Given the `break` logic, a `current_path_filter` should exist if any key was non-null.
            // If all keys were null for a path, it means that path is unconstrained.
            // However, get_join_keys ensures keys have *some* non-nulls in previous_result_df.
            // This path should likely not occur if unique_previous_paths_df is not empty.
            warn!(
                "No filter constructed for path row {}, possibly all keys were null (should not happen if get_join_keys is effective).",
                row_idx
            );
        }
    }

    if lfs_to_concat.is_empty() {
        trace!(
            "No data matched any constructed filter paths from previous results. Returning empty frame."
        );
        Ok(data.limit(0))
    } else {
        trace!(
            "Concatenating {} LazyFrames from filtered data chunks.",
            lfs_to_concat.len()
        );
        concat(&lfs_to_concat, UnionArgs::default())
            .map(|lf| lf.unique_stable(Some(vec!["geonameId".into()]), UniqueKeepStrategy::First))
            .map_err(From::from)
    }
}
