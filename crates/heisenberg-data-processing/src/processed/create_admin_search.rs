use polars::prelude::*;
use tracing::instrument;

use crate::raw::fetch::TempData;

/// Transform the `GeoNames` data into the admin search format
/// returns a `LazyFrame` with the admin search data
#[instrument(
    name = "Transform GeoNames data for admin search",
    skip_all,
    level = "debug"
)]
pub fn get_admin_search_lf(temp_data: &TempData) -> LazyFrame {
    let places_lf = temp_data.places.as_lazy_frame();
    let country_info_lf = temp_data.country_info.as_lazy_frame();
    let admin1_lf = temp_data.admin1.as_lazy_frame();
    let admin2_lf = temp_data.admin2.as_lazy_frame();

    // Strong admin data refers to the data coming from the places data. If the data being used is the allCountries data dump then
    // this will include all the admin regions and is the most comprehensive source of admin data.
    // If however the places data being used is one of the citiesNNNN data dumps then this will only include much data and only have places data,
    // not admin regions or countries.
    // However to combat this we can use the country info and admin1/admin2 data to fill in the gaps, and is what the weak_admin_data is.
    // This weak_admin_data wont be as comprehensive as the strong_admin_data (e.g. missing population, lat/lon, alternatenames etc) but
    // if we are using the allCountries data dump then this wont replace better data.
    //
    // Its essentially here so if we are using the smaller citiesNNNN data dumps we can still have a reasonable admin search dataset.
    // If we are using the allCountries data dump then this wont replace any better and performance cost is negligible for building the weak_admin_data compared to the strong_admin_data.

    let strong_admin_data = places_lf
        .with_column(
            // Determine admin level based on available admin codes
            when(col("admin4_code").is_not_null())
                .then(lit(5u8))
                .when(col("admin3_code").is_not_null())
                .then(lit(4u8))
                .when(col("admin2_code").is_not_null())
                .then(lit(3u8))
                .when(col("admin1_code").is_not_null())
                .then(lit(2u8))
                .when(col("admin0_code").is_not_null())
                .then(lit(1u8))
                .otherwise(lit(NULL))
                .cast(DataType::UInt8)
                .alias("admin_level_tmp"),
        )
        .with_column(
            // This will be top level admin regions (countries) so despite having a matching admin code we set to 0 as its admin_level_tmp value is 1
            when(
                col("feature_code")
                    .str()
                    .contains(lit(r"^PCL[A-Z]*|TERR$"), true),
            )
            .then(lit(0u8))
            // This will be first to fifth level admin regions (ADM1-ADM5). This should be the same as the number of admin codes.
            .when(col("feature_code").str().contains(lit(r"^ADM[1-5]$"), true))
            .then(
                // Extract the number from ADM1, ADM2, etc.
                col("feature_code")
                    .str()
                    .extract(lit(r"^ADM(\d)"), 1)
                    .cast(DataType::UInt8),
            )
            // Extract the number from PPLA2, PPLA3, etc. Default to 1 if just PPLA
            .when(
                col("feature_code")
                    .str()
                    .contains(lit(r"^PPLA[2-5]?$"), true),
            )
            .then(
                col("feature_code")
                    .str()
                    .extract(lit(r"^PPLA(\d)"), 1)
                    .fill_null(lit(1u8))
                    .cast(DataType::UInt8),
            )
            // This will be populated places (PPLC, PPLA2-PPLA5, PPLX), this can be the tmp value assigned as its a good proxy for admin level.
            .when(
                col("feature_code")
                    .str()
                    .contains(lit(r"^PPLC[D]?|PPLA[2-5]?|PPLX$"), true),
            )
            .then(col("admin_level_tmp"))
            // Capture any remaining admin regions
            .when(col("feature_class").eq(lit("A")))
            .then(col("admin_level_tmp"))
            // Deliberately ignoring PPL as these are not admin regions
            .otherwise(lit(NULL))
            .clip(lit(0u8), lit(5u8))
            .cast(DataType::UInt8)
            .alias("admin_level"),
        )
        // Ensures that admin_level 0 entries have a null admin1_code as they are countries (and most default admin1_code to 00 for the country entry)
        .with_column(
            when(col("admin_level").eq(lit(0u8)))
                .then(lit(NULL))
                .otherwise(col("admin1_code"))
                .alias("admin1_code"),
        )
        // Remove anything that we dont consider an admin region
        .filter(col("admin_level").is_not_null())
        // Join with country info to get ISO, ISO3, official country name, fips etc
        .join(
            country_info_lf.clone(),
            [col("geonameId")],
            [col("geonameId")],
            JoinArgs {
                how: JoinType::Left,
                ..Default::default()
            },
        )
        // Select and rename columns to match the desired output
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
        ]);

    // Building our weak admin data from country info and admin1/admin2 data
    // Reshape country info to match admin search schema
    let reshaped_country_info = country_info_lf
        .with_columns([
            col("Country").alias("name"),
            col("Country").alias("asciiname"),
            lit(0u8).cast(DataType::UInt8).alias("admin_level"),
            col("ISO").alias("admin0_code"),
            lit(NULL).cast(DataType::String).alias("admin1_code"),
            lit(NULL).cast(DataType::String).alias("admin2_code"),
            lit(NULL).cast(DataType::String).alias("admin3_code"),
            lit(NULL).cast(DataType::String).alias("admin4_code"),
            lit("A").alias("feature_class"),
            lit("PCLI").alias("feature_code"),
            lit(NULL).cast(DataType::Float32).alias("latitude"),
            lit(NULL).cast(DataType::Float32).alias("longitude"),
            col("Population").cast(DataType::Int64).alias("population"),
            lit(NULL)
                .cast(DataType::List(Box::new(DataType::String)))
                .alias("alternatenames"),
        ])
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
        ]);

    // Create the weak admin data by combining all the data together. This dataset wont be as comprehensive as the strong_admin_data but will fill in gaps if using a smaller places dataset.
    let weak_admin_data = concat(
        &[reshaped_country_info, admin1_lf, admin2_lf],
        UnionArgs {
            diagonal: true,
            ..Default::default()
        },
    )
    .expect("List is not empty");
    // Combine strong and weak admin data
    concat(
        &[strong_admin_data, weak_admin_data],
        UnionArgs {
            rechunk: true,
            ..Default::default()
        },
    )
    .expect("List is not empty")
    // Deduplicate based on geonameId, keeping the first (strong admin data will be first so preferred)
    .unique(col("geonameId").into_selector(), UniqueKeepStrategy::First)
    .sort(
        ["geonameId"],
        SortMultipleOptions::new().with_nulls_last(true),
    )
}
