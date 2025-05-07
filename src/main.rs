pub mod tantivy_index;
use anyhow::{Context, Result};

use polars::prelude::*;
use std::ops::Div;
use std::ops::Mul;
use tantivy_index::{FTSIndex, FTSIndexes};
use tracing::{debug, info, instrument, warn, Level};
use tracing_subscriber::fmt::format::FmtSpan;

fn main() -> Result<()> {
    tracing_subscriber::fmt::fmt()
        // filter spans/events with level TRACE or higher.
        .with_max_level(Level::INFO)
        .with_span_events(FmtSpan::CLOSE)
        // build but do not install the subscriber.
        .init();

    let admin_index = FTSIndex::new(FTSIndexes::AdminSearch)?;

    let lf = LazyFrame::scan_parquet(
        "./data/processed/geonames/admin_search.parquet",
        Default::default(),
    )?;

    let admins = admin_search(
        "The united states of america",
        &[0],
        lf.clone(),
        &admin_index,
        None,
        Some(20),
        true,
    )?
    .unwrap()
    .collect()?;

    info!("Admin search results: {:?}", admins);

    let admins1 = admin_search(
        "California",
        &[1],
        lf.clone(),
        &admin_index,
        Some(admins.lazy()),
        Some(20),
        true,
    )?
    .unwrap()
    .collect()?;
    info!("Admin1 search results: {:?}", admins1);

    let admins2 = admin_search(
        "Los Angeles County",
        &[2],
        lf.clone(),
        &admin_index,
        Some(admins1.lazy()),
        Some(20),
        false,
    )?
    .unwrap()
    .collect()?;
    info!("Admin2 search results: {:?}", admins2);

    Ok(())
}

/*
```.py

def get_latest_adjusted_score_level(columns: list[str]) -> int | None:
    adjusted_score_columns = [
        col for col in columns if col.startswith("adjusted_score_")
    ]
    if not adjusted_score_columns:
        return None
    # Extract the level from the column name and find the maximum level
    levels = [int(col.rsplit("_", maxsplit=1)[-1]) for col in adjusted_score_columns]
    max_level = max(levels)
    return max_level

def search_score_admin(
    df: pl.LazyFrame,
    level: int,
    text_weight: float = 0.35,
    pop_weight: float = 0.35,
    feature_weight: float = 0.15,
    parent_weight: float = 0.15,
    search_term: str | None = None,
) -> pl.LazyFrame:
    """
    A scoring function for geographic entities that better prioritizes
    significant locations.

    Parameters:
    - df: DataFrame with search results
    - level: Admin level (0=country, 1=admin1, etc.)
    - text_weight: Weight for text matching score
    - pop_weight: Weight for population-based importance
    - feature_weight: Weight for feature type significance
    - parent_weight: Weight for parent entity scores
    - search_term: Original search term (for exact match detection)

    Returns:
    - DataFrame with adjusted scores
    """
    assert level in range(5), "Level must be between 0 and 4"
    score_col = f"adjusted_score_{level}"
    columns = df.collect_schema().names()

    # ===== 1. Text relevance score =====
    fts_column = f"fts_score_{level}"
    if "fts_score" in columns:
        df = df.rename({"fts_score": fts_column})

        df = df.with_columns(
            # Calculate z-score
            z_score=(
                (pl.col(fts_column) - pl.col(fts_column).mean())
                / pl.when(pl.col(fts_column).std() > 0)
                .then(pl.col(fts_column).std())
                .otherwise(1.0)
            ),
        ).with_columns(
            # Apply sigmoid transformation: 1/(1+e^(-z))
            text_score=(1 / (1 + pl.col.z_score.mul(-1.5).exp()))
        )
        if search_term:
            df = df.with_columns(
                text_score=pl.when(
                    pl.col.name.str.to_lowercase() == search_term.lower()
                )
                .then(1)
                .otherwise(pl.col.text_score)
                .clip(0, 1)
            )
    else:
        logger.warning(
            f"Column '{fts_column}' not found in DataFrame. Skipping Z-score normalization."
        )
        df = df.with_columns(text_score=pl.lit(0.5))

    # ===== 2. Population importance - stronger scaling =====
    pop_col = "population"
    if pop_col in columns:
        df = df.with_columns(
            # Sigmoid normalized population factor
            pop_score=pl.when(pl.col(pop_col) > 0)
            .then(
                (
                    # Stronger population scaling using logarithmic curve
                    1 - 1 / (1 + (pl.col(pop_col).log10() / 3))
                )
            )
            .otherwise(0.1)
        )
    else:
        logger.warning(
            f"Column '{pop_col}' not found in DataFrame. Skipping population factor."
        )
        df = df.with_columns(pop_score=pl.lit(0.3))

    # ===== 3. Feature type importance =====
    feature_col = "feature_code"
    if feature_col in columns:
        df = df.with_columns(
            # More nuanced feature type scoring based on importance
            feature_score=pl.when(pl.col(feature_col) == "PCLI")
            .then(1.0)  # Independent countries
            .when(pl.col(feature_col).str.starts_with("PCL"))
            .then(0.9)  # Other country-like entities
            .when(pl.col(feature_col) == "PPLC")
            .then(0.95)  # Capital cities
            .when(pl.col(feature_col).str.starts_with("PPL"))
            .then(0.8)  # Major populated places
            .when(pl.col(feature_col).str.starts_with("ADM1"))
            .then(0.85)  # First-level admin (provinces/states)
            .when(pl.col(feature_col).str.starts_with("ADM2"))
            .then(0.75)  # Second-level admin (counties)
            .when(pl.col(feature_col).str.starts_with("ADM3"))
            .then(0.65)  # Third-level admin (districts)
            .when(pl.col(feature_col).str.starts_with("ADM"))
            .then(0.55)  # Other admin units
            .otherwise(0.5)
        )
    else:
        logger.warning(
            f"Column '{feature_col}' not found in DataFrame. Skipping feature factor."
        )
        df = df.with_columns(feature_score=pl.lit(0.5))

    # ===== 4. Country/region prominence - prioritize major countries =====
    country_col = "admin0_code"
    if country_col in columns:
        # List of major countries to prioritize
        major_countries = [
            "US",
            "GB",
            "DE",
            "FR",
            "JP",
            "CN",
            "IN",
            "BR",
            "RU",
            "CA",
            "AU",
        ]
        df = df.with_columns(
            country_score=pl.when(pl.col(country_col).is_in(major_countries))
            .then(0.8)  # Major countries
            .otherwise(0.5)  # Other countries
        )
    else:
        df = df.with_columns(country_score=pl.lit(0.5))

    # ===== 5. Parent score influence =====
    # TODO: No longer need for the function
    if get_latest_adjusted_score_level(columns) is not None:
        df = df.with_columns(
            average_parent_score=pl.mean_horizontal(cs.starts_with("adjusted_score_"))
        ).with_columns(
            parent_factor=pl.when(pl.col.average_parent_score > 0)
            .then(pl.col.average_parent_score / pl.col.average_parent_score.max())
            .otherwise(0.5)
        )

    else:
        logger.warning("No parent score column found. Skipping parent factor.")
        df = df.with_columns(parent_factor=pl.lit(0.5))

    # ===== 6. Final score calculation =====
    # Base score calculation
    df = df.with_columns(
        (
            pl.col("text_score").mul(text_weight)
            + pl.col("pop_score").mul(pop_weight)
            + pl.col("feature_score").mul(feature_weight)
            + pl.col("parent_factor").mul(parent_weight)
        ).alias("base_score")
    )

    # Apply country prominence boost to the final score
    df = df.with_columns(
        (pl.col("base_score") * (0.7 + (0.3 * pl.col("country_score")))).alias(
            score_col
        )
    )

    # For debugging, keep all intermediate scores
    return df.sort(score_col, descending=True)


def build_path_conditions(df: pl.DataFrame, admin_cols: list[str]) -> str:
    """
    Build SQL conditions by scanning backwards to find the last non-null value.
    """
    if not admin_cols or df.is_empty():
        return ""

    # Extract relevant columns and filter out all-null rows
    paths_df = (
        df.select(admin_cols).filter(~pl.all_horizontal(pl.all().is_null())).unique()
    )

    path_conditions = []
    for row in paths_df.iter_rows(named=True):
        # Scan backward to find the last non-null column
        last_non_null_idx = -1
        for idx in range(len(admin_cols) - 1, -1, -1):
            if row[admin_cols[idx]] is not None:
                last_non_null_idx = idx
                break

        if last_non_null_idx == -1:
            continue  # Skip rows with all nulls

        # Build conditions up through the last non-null column
        conditions = []
        for idx in range(last_non_null_idx + 1):
            col = admin_cols[idx]
            val = row[col]
            if val is None:
                conditions.append(f"{col} IS NULL")
            else:
                conditions.append(f"{col} = '{val}'")

        path_conditions.append(f"({' AND '.join(conditions)})")

    return " OR ".join(path_conditions)

def search_admin(
    term: str,
    levels: list[int] | int,
    con: DuckDBPyConnection,
    previous_results: pl.DataFrame | None = None,
    limit: int = 100,
    all_cols: bool = False,
) -> pl.DataFrame:
    """
    Search for admin entities across one or multiple admin levels with path-aware filtering.

    Parameters:
    - term: The search term to look for
    - levels: A list of admin levels to search over (0-4) or a single level
    - con: The DuckDB connection object
    - previous_results: Previous search results to filter against
    - limit: The maximum number of results to return

    Returns:
    - A DataFrame with the search results
    """
    # Normalize levels to a list
    if isinstance(levels, int):
        levels = [levels]
    elif not isinstance(levels, list):
        raise ValueError("Levels must be an integer or a list of integers")

    # Validate levels
    if not all(0 <= level <= 4 for level in levels):
        raise ValueError("All levels must be between 0 and 4")

    # Build level constraint
    level_conditions = " OR ".join([f"admin_level = {level}" for level in levels])
    where_clauses = [f"({level_conditions})"]

    # Special handling for country (admin_level = 0) exact matches
    has_country_level = 0 in levels
    country_exact_matches = None

    select: list[str] = (
        [
            "geonameId",
            "name",
            "asciiname",
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
        ]
        if not all_cols
        else ["*"]
    )

    if has_country_level and len(term) <= 3:
        # If the term is short (<= 3 characters), we can assume it's a country code
        # First try exact matches for country codes
        exact_match_query = f"""
        SELECT {", ".join(select)},
        -- High fixed score for exact matches
        CASE
            WHEN LOWER(ISO) = LOWER($term) THEN 10.0
            WHEN LOWER(ISO3) = LOWER($term) THEN 8.0
            WHEN LOWER(fips) = LOWER($term) THEN 4.0
        END AS fts_score
        FROM admin_search
        WHERE admin_level = 0 AND (
            LOWER(ISO) = LOWER($term) OR
            LOWER(ISO3) = LOWER($term) OR
            LOWER(fips) = LOWER($term)
        )
        """

        country_exact_matches = con.execute(exact_match_query, {"term": term}).pl()

        # If we found exact matches, exclude these from the FTS search
        if not country_exact_matches.is_empty():
            country_ids = country_exact_matches["geonameId"].to_list()
            where_clauses.append(
                f"(admin_level != 0 OR geonameId NOT IN ({','.join(map(str, country_ids))}))"
            )

    admin_cols = []
    # Build path filtering from previous results
    if previous_results is not None and not previous_results.is_empty():
        # Determine which admin code columns to use based on the previous results
        for i in range(5):
            col = f"admin{i}_code"
            if (
                col in previous_results.columns
                and previous_results[col].drop_nulls().shape[0] > 0
            ):
                admin_cols.append(col)

        # Build path conditions using the admin code columns
        if admin_cols:
            path_conditions = build_path_conditions(previous_results, admin_cols)
            if path_conditions:
                where_clauses.append(f"({path_conditions})")

    # Build the WHERE clause
    where_clause = " AND ".join(where_clauses)

    # Build and execute the FTS search
    fts_query = f"""

    WITH filtered_results AS (
        SELECT {",".join(select)}, fts_main_admin_search.match_bm25(geonameId, $term) AS fts_score
        FROM admin_search
        WHERE {where_clause}
    )
    -- Using a CTE to ensure we always filter before the FTS score is calculated. Because of the `WHERE fts_score IS NOT NULL` clause, the FTS score will be calculated for all rows, but we only want to keep those that match the search term, hence the subquery first in order to stop the filter push down.
    SELECT * FROM filtered_results
    WHERE fts_score IS NOT NULL
    ORDER BY fts_score DESC
    LIMIT $limit
    """
    logger.debug(f"Executing FTS query: {fts_query}")
    fts_results = con.execute(fts_query, {"term": term, "limit": limit * 2}).pl()

    # Combine exact matches with FTS results if we had exact matches
    if country_exact_matches is not None and not country_exact_matches.is_empty():
        # Ensure both have the same columns
        if not fts_results.is_empty():
            # Dont need this any more.
            # Make sure both have the same columns in the same order
            # all_columns = list(
            #     set(country_exact_matches.columns).union(set(fts_results.columns))
            # )

            # Add any missing columns with None values
            # for col in all_columns:
            #     if col not in country_exact_matches.columns:
            #         country_exact_matches = country_exact_matches.with_columns(
            #             pl.lit(None).alias(col)
            #         )
            #     if col not in fts_results.columns:
            #         fts_results = fts_results.with_columns(pl.lit(None).alias(col))

            # Combine and sort by score
            results = pl.concat(
                [country_exact_matches.lazy(), fts_results.lazy()],
                how="vertical_relaxed",
            )
            results = results.sort("fts_score", descending=True)
        else:
            # If no FTS results, just use exact matches
            results = country_exact_matches.lazy()
    else:
        # Just use FTS results
        results = fts_results.lazy()

    # Trying to get the adjusted scores from the previous results working with the flexible search. The issue is that we need to be able to join the previous results with the current results based on the admin codes. (Tracking the path back is much harder than when doing hierarchical search, as there are potentially multiple paths to the same entity. Unsure how to do this yet. )
    # logger.info(admin_cols)
    # if previous_results is not None and not previous_results.is_empty():
    #     for i in range(1, len(admin_cols)+1):
    #         logger.info(i)
    #         logger.warning(f"adjusted_score_{min(levels)-1}")
    #         tmp_cols = admin_cols[:i]
    #         logger.info(f"{tmp_cols=}")
    #         logger.info(f"{results.collect_schema().names()=}")
    #         logger.info(f"{previous_results.select(
    #                 cs.by_name(tmp_cols), cs.starts_with("adjusted_score_")
    #             ).collect_schema().names()=}")

    #         results = results.join(
    #             previous_results.select(
    #                 cs.by_name(tmp_cols), pl.col(f"adjusted_score_{min(levels)-1}")
    #             ),
    #             on=tmp_cols,
    #             how="left",
    #         )
    #         logger.info(results.collect_schema())
    #         if f"adjusted_score_{min(levels)-1}_right" in results.collect_schema().names():
    #             results = results.with_columns(pl.coalesce(cs.starts_with(f"adjusted_score_{min(levels)-1}"))).drop(cs.ends_with("right"))

    # Original way that works for hierarchical search but not flexible search. Want to try and get this working for flexible search as well.
    # For now we will just ignore any previous score when doing the flexible search as it complicates things too much.
    # A simple way to work out if we are doing a flexible search is to check the length of the admin_cols and the length of the levels.
    if (
        previous_results is not None and not previous_results.is_empty()
        # and 1 == len(levels)
    ):
        # Join with previous results to get adjusted scores
        results = results.join(
            previous_results.lazy().select(
                cs.by_name(admin_cols), cs.starts_with("adjusted_score_")
            ),
            on=admin_cols,
            how="left",
        )

    return (
        results.pipe(search_score_admin, min(levels), search_term=term)
        .sort(f"adjusted_score_{min(levels)}", descending=True)
        .unique("geonameId", keep="first", maintain_order=True)
        .head(limit)
        .select(
            (cs.by_name(select), cs.starts_with("adjusted_score_"))
            if not all_cols
            else "*"
        )
        .collect()
    )
*/

fn search_score_admin(
    lf: LazyFrame,
    score_col: &str,
    text_weight: f32,
    pop_weight: f32,
    feature_weight: f32,
    parent_weight: f32,
    search_term: &str,
) -> LazyFrame {
    let re = regex::Regex::new(r"^adjusted_score_[0-4]$").unwrap();
    let score_col_exists = dbg!(lf
        .clone()
        .collect_schema()
        .unwrap()
        .iter_names()
        .any(|s| { re.is_match(s) }));

    //.contains("^admin[0-4]_code$"));

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
                vec![
                    "US", "GB", "DE", "FR", "JP", "CN", "IN", "BR", "RU", "CA", "AU",
                ],
            ))))
            .then(lit(0.8))
            .otherwise(lit(0.5))
            .alias("country_score"),
        );

    // ===== 5. Parent score influence =====

    let lf = if score_col_exists {
        lf.with_column(
            polars::lazy::dsl::mean_horizontal([col("^adjusted_score_[0-4]$")], true)
                .context("Failed to calculate mean horizontal")
                .unwrap()
                .alias("average_parent_score"),
        )
        .with_column(
            when(col("average_parent_score").gt(0.0))
                .then(col("average_parent_score").div(col("average_parent_score").max()))
                .otherwise(lit(0.5))
                .alias("parent_factor"),
        )
    } else {
        lf.with_column(lit(0.5).alias("parent_factor"))
    };

    // ====== 6. Final score calculation =====
    lf.with_column(
        (col("text_score").mul(lit(text_weight))
            + col("pop_score").mul(lit(pop_weight))
            + col("feature_score").mul(lit(feature_weight))
            + col("parent_factor").mul(lit(parent_weight)))
        .alias("base_score"),
    )
    .collect()
    .unwrap()
    .lazy()
    // Apply country prominence boost to the final score
    .with_column(
        (col("base_score") * (lit(0.7) + (lit(0.3) * col("country_score")))).alias(score_col),
    )
    .collect()
    .unwrap()
    .lazy()
    .sort(
        [score_col],
        SortMultipleOptions::new().with_order_descending(true),
    )
}

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

#[instrument(skip(data, previous_result))]
fn filter_data_from_previous_results(
    data: LazyFrame,
    previous_result: LazyFrame,
    join_on_cols_str: &[Expr],
) -> Result<LazyFrame> {
    if !join_on_cols_str.is_empty() {
        // Select only the unique combinations of join columns from prev_lf
        let previous_result = previous_result
            .select(join_on_cols_str)
            .unique(None, Default::default());

        //let previous_result = dbg!(previous_result.collect()?).lazy();

        return Ok(data.join(
            previous_result,
            join_on_cols_str,
            join_on_cols_str,
            JoinArgs::new(JoinType::Inner),
        ));
    }
    Ok(data)
}

#[instrument(skip(data, index, previous_result, limit))]
fn admin_search(
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

    let join_cols_str = match previous_result {
        Some(ref prev_lf) => {
            //let prev_lf = dbg!(prev_lf.collect()?).lazy();
            get_join_keys(prev_lf).context("Failed to get join keys")?
        }
        None => {
            vec![]
        }
    };
    dbg!(&join_cols_str);
    //dbg!(&levels);
    //let data = dbg!(data.collect()?).lazy();
    let data = match previous_result {
        Some(ref prev_lf) => {
            //let prev_lf = dbg!(prev_lf.collect()?).lazy();
            filter_data_from_previous_results(data, prev_lf.clone(), &join_cols_str)
                .context("Failed to filter data from previous results")?
        }
        None => data,
    };
    let data = data.filter(col("admin_level").is_in(lit(levels_series)));

    // Helper to create an empty result with the correct schema
    // let create_empty_lf_with_schema = |schema_source_lf: &LazyFrame| {
    //     schema_source_lf
    //         .clone()
    //         .with_column(lit(0.0_f32).cast(DataType::Float32).alias("fts_score"))
    //         .select([col("*").exclude(["fts_score"]), col("fts_score")])
    //         .filter(lit(false)) // Ensure it's empty
    // };

    let gids = data.clone().select([col("geonameId")]).collect()?;
    let gid_series = gids.column("geonameId")?;
    // if gid_series.is_empty() {
    //     // If there are no geonameId values, return an empty DataFrame with the correct schema
    //     return Ok(create_empty_lf_with_schema(&data));
    // }
    if gid_series.is_empty() {
        // If there are no geonameId values, return an empty DataFrame with the correct schema
        // return Ok(create_empty_lf_with_schema(&data));
        warn!("gid_series is empty");
        return Ok(None);
    }

    let gids = gid_series
        .cast(&DataType::UInt64)
        .context("Failed to cast geonameId to UInt64")?
        .u64()
        .context("Failed to get geonameId as u64")?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    debug_assert!(!gids.is_empty(), "gids should not be empty");
    // Early return if gids is empty

    // Unzip results into separate vectors
    let (gids, scores): (Vec<_>, Vec<_>) = index
        .search_in_subset(term, gids.as_slice(), limit, false)?
        .iter()
        .map(|(_, id, score)| (*id, *score))
        .unzip();

    let fts_results_lf = df!("geonameId" => gids,
                "fts_score" => scores,
    )?
    .lazy()
    .join(
        data,
        [col("geonameId")],
        [col("geonameId")],
        JoinArgs::new(JoinType::Left),
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

    let fts_results_lf = if dbg!(!join_cols_str.is_empty()) & dbg!(previous_result.is_some()) {
        let fts_results_lf = dbg!(fts_results_lf
            .collect()
            .context("Failed to collect FTS results before join")?)
        .lazy();
        let mut selected_cols = join_cols_str.clone();
        selected_cols.push(col("^adjusted_score_[0-4]$"));

        fts_results_lf.join(
            previous_result.unwrap().select(&selected_cols),
            &join_cols_str,
            &join_cols_str,
            JoinArgs::new(JoinType::Left),
        )
    } else {
        fts_results_lf
    };

    let fts_results_lf = dbg!(fts_results_lf
        .collect()
        .context("Failed to collect FTS results")?)
    .lazy();

    let min_level = levels.iter().min().unwrap_or(&0);
    debug_assert!(min_level < &5, "Level must be between 0 and 4");
    let score_col = format!("adjusted_score_{}", min_level);

    let select = if all_cols {
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

    let output_lf = search_score_admin(
        fts_results_lf,
        &score_col,
        0.35, // text_weight
        0.35, // pop_weight
        0.15, // feature_weight
        0.15, // parent_weight
        term,
    )
    .unique_stable(Some(vec!["geonameId".into()]), UniqueKeepStrategy::First)
    .limit(limit.try_into().unwrap_or(100))
    .select(select);

    Ok(Some(output_lf))
}
