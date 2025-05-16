use anyhow::Result;
use polars::prelude::*;

pub fn get_admin_search_lf(all_countries: LazyFrame, country_info: LazyFrame) -> Result<LazyFrame> {
    Ok(all_countries
        .with_column(
            polars::lazy::dsl::sum_horizontal([col(r"^admin\d+_code$").is_not_null()], true)?
                .cast(DataType::UInt8)
                .alias("admin_level_tmp"),
        )
        .with_column(
            when(
                col("feature_code")
                    .str()
                    .contains(lit(r"^PCL[A-Z]*|TERR$"), true),
            )
            .then(lit(0))
            .when(col("feature_code").str().contains(lit(r"^ADM[1-5]$"), true))
            .then(col("admin_level_tmp") - lit(1))
            .when(
                col("feature_code")
                    .str()
                    .contains(lit(r"^PPLC|PPLA[2-5]*|PPLX$"), true),
            )
            .then(col("admin_level_tmp"))
            .otherwise(lit(NULL))
            .clip(lit(0), lit(5))
            .alias("admin_level"),
        )
        .filter(col("admin_level").lt(lit(5)))
        .with_column(
            when(col("admin_level").eq(lit(0)))
                .then(lit(NULL))
                .otherwise(col("admin1_code"))
                .alias("admin1_code"),
        )
        .join(
            country_info.clone(),
            [col("geonameId")],
            [col("geonameId")],
            JoinArgs {
                how: JoinType::Left,
                ..Default::default()
            },
        )
        .filter(col("admin_level").is_not_null())
        .select([
            col("geonameId"),
            col("name"),
            col("asciiname"),
            col("admin_level"),
            col("admin0_code"),
            col("admin1_code"),
            col("admin2_code"),
            col("admin3_code"),
            col("admin4_code"),
            col("feature_class"),
            col("feature_code"),
            col("ISO"),
            col("ISO3"),
            col("Country").alias("official_name"),
            col("fips"),
            col("latitude"),
            col("longitude"),
            col("population"),
            col("alternatenames"),
        ]))
}
