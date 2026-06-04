use std::path::Path;

use polars::prelude::*;
use tracing::{info, info_span};

use super::error::Result;
use crate::raw::fetch::TempData;

pub mod create_admin_search;
pub mod create_place_search;

/// Generate processed data from raw sources and save to parquet files
pub fn generate_processed_data(temp_data: TempData) -> Result<(DataFrame, DataFrame)> {
    // Cant drop temp files until end of scope otherwise they get deleted and the lazy frames
    // cant read the data.
    let mut dfs = {
        let _span = info_span!("Transform Raw Data").entered();

        let admin_search_lf = create_admin_search::get_admin_search_lf(&temp_data);

        let place_search_lf =
            create_place_search::get_place_search_lf(&temp_data, admin_search_lf.clone());
        info!("Collecting transformed data");
        let transform_time = std::time::Instant::now();
        let plans = [admin_search_lf, place_search_lf]
            .into_iter()
            .map(|lf| lf.logical_plan)
            .collect();
        let dfs = LazyFrame::collect_all_with_engine(plans, Engine::Auto, OptFlags::default())?;
        info!(
            transform_time = ?transform_time.elapsed()
            , "Transforming data took"
        );
        dfs
    };

    let place_search_df = dfs.pop().expect("Place search should be last");
    let admin_search_df = dfs.pop().expect("Admin search should be first");

    drop(temp_data); // Drop temp data to clean up files

    Ok((admin_search_df, place_search_df))
}

pub fn save_processed_data_to_parquet(df: DataFrame, path: &Path) -> Result<()> {
    let sink_time = std::time::Instant::now();

    let mut df = df
        .lazy()
        .drop_nulls(col("geonameId").into_selector())
        .sort(["geonameId"], SortMultipleOptions::default())
        .collect()?;
    let mut file = std::fs::File::create(path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;

    info!(
        path = ?path.file_stem(),
        sink_time = ?sink_time.elapsed(),
        "Saved to parquet file"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::*;

    #[test]
    fn test_create_admin_search_actual_transformation() {
        // Use the actual test data from samples
        let temp_data = crate::raw::test_data::create_test_data();

        // Test the actual transformation
        let result = create_admin_search::get_admin_search_lf(&temp_data);
        let df = result.collect().unwrap();

        // Test that the transformation worked
        assert!(df.height() > 0, "Admin search should have results");

        // Test that admin_level was computed correctly
        assert_has_columns(&df, &["admin_level"]);
        assert_column_type(&df, "admin_level", &DataType::UInt8);

        // Test that only admin levels < 5 are included (as per your filter)
        let admin_levels: Vec<Option<u8>> = df
            .column("admin_level")
            .unwrap()
            .u8()
            .unwrap()
            .into_iter()
            .collect();
        for level in admin_levels.iter().flatten() {
            assert!(*level < 5, "All admin levels should be < 5, found: {level}");
        }

        // Test that country-level (PCLI) and state-level (ADM1) entries are present
        let feature_codes: Vec<Option<&str>> = df
            .column("feature_code")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert!(
            feature_codes.contains(&Some("PCLI")),
            "Should contain country-level entity"
        );
        assert!(
            feature_codes.contains(&Some("ADM1")),
            "Should contain state-level entity"
        );
    }

    #[test]
    fn test_create_place_search_actual_transformation() {
        // Use the actual test data from samples
        let temp_data = crate::raw::test_data::create_test_data();

        // Debug: Check input data
        let all_countries_df = temp_data.places.as_lazy_frame().collect().unwrap();
        println!(
            "All countries data (before anti-join): {} rows",
            all_countries_df.height()
        );

        // Create admin search
        let admin_search_lf = create_admin_search::get_admin_search_lf(&temp_data);
        let admin_search_df = admin_search_lf.clone().collect().unwrap();
        println!("Admin search data: {} rows", admin_search_df.height());

        // Debug: Test the anti-join directly
        let after_anti_join = temp_data
            .places
            .as_lazy_frame()
            .join(
                admin_search_lf.clone(),
                [col("geonameId")],
                [col("geonameId")],
                JoinArgs {
                    how: JoinType::Anti,
                    ..Default::default()
                },
            )
            .collect()
            .unwrap();
        println!("After anti-join: {} rows", after_anti_join.height());

        // Debug: Test the filter step
        let after_filter = temp_data
            .places
            .as_lazy_frame()
            .join(
                admin_search_lf.clone(),
                [col("geonameId")],
                [col("geonameId")],
                JoinArgs {
                    how: JoinType::Anti,
                    ..Default::default()
                },
            )
            .filter(
                (col("admin0_code").is_not_null()).and(
                    col("feature_class").is_in(
                        lit(Series::new(
                            "feature_class_to_keep".into(),
                            ["P", "S", "T", "H", "L", "V", "R"],
                        ))
                        .implode(),
                        false,
                    ),
                ),
            )
            .collect()
            .unwrap();
        println!("After filter: {} rows", after_filter.height());
        println!("After filter data: {after_filter:?}");

        // Debug: Check if feature_codes parsing is working
        let feature_codes_df = temp_data.feature_codes.as_lazy_frame().collect().unwrap();
        println!("Feature codes data: {} rows", feature_codes_df.height());
        println!("Feature codes: {feature_codes_df:?}");

        // Debug the actual transformation - run it but collect intermediate steps
        let result = create_place_search::get_place_search_lf(&temp_data, admin_search_lf);

        let df = result.collect().unwrap();
        println!("Final result: {} rows", df.height());
        if df.height() > 0 {
            println!("Final data: {df:?}");
        }

        // Test that transformation worked
        assert!(df.height() > 0, "Place search should have results");

        // Test required columns exist
        assert_has_columns(&df, &["importance_score", "importance_tier"]);
        assert_column_type(&df, "importance_score", &DataType::Float64);
        assert_column_type(&df, "importance_tier", &DataType::UInt8);

        // Test that importance values are in valid ranges
        assert_column_range(&df, "importance_score", 0.0f64, 1.0f64);
        assert_column_range(&df, "importance_tier", 1u8, 5u8);

        // Test that places (P class) are included, admin (A class) are excluded
        let feature_classes: Vec<Option<&str>> = df
            .column("feature_class")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert!(
            feature_classes.contains(&Some("P")),
            "Should contain places"
        );
        assert!(
            !feature_classes.contains(&Some("A")),
            "Should not contain admin entities after anti-join"
        );
    }
}
