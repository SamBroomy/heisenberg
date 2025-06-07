use super::error::Result;
use once_cell::sync::OnceCell;
use polars::prelude::*;
use std::path::{Path, PathBuf};
use tracing::{info, info_span};

pub mod create_admin_search;
pub mod create_place_search;

const ADMIN_SEARCH_PARQUET: &str = "admin_search.parquet";
const PLACE_SEARCH_PARQUET: &str = "place_search.parquet";

#[derive(Clone)]
pub struct LocationSearchData {
    admin_search_path: PathBuf,
    place_search_path: PathBuf,
    admin_search_df: OnceCell<LazyFrame>,
    place_search_df: OnceCell<LazyFrame>,
}

impl LocationSearchData {
    pub fn new() -> Result<Self> {
        Self::new_with_persistent_data()
    }

    fn new_with_persistent_data() -> Result<Self> {
        info!("LocationSearchData: Using persistent data storage");

        let processed_dir = crate::get_data_dir().join("processed");
        std::fs::create_dir_all(&processed_dir)?;

        let admin_search_path = processed_dir.join(ADMIN_SEARCH_PARQUET);
        let place_search_path = processed_dir.join(PLACE_SEARCH_PARQUET);

        if admin_search_path.exists() && place_search_path.exists() {
            info!("LocationSearchData: Loading existing Parquet files");
            return Ok(Self::load_parquet_files(
                admin_search_path,
                place_search_path,
            ));
        }

        info!("LocationSearchData: Generating processed data from raw sources");

        // Generate from raw data
        let mut dfs = {
            let _span = info_span!("Transform Raw Data").entered();
            let raw_data = super::raw::get_raw_data()?;
            let (all_countries_lf, country_info_lf, feature_codes_lf) =
                super::raw::get_raw_data_as_lazy_frames(&raw_data)?;

            let admin_search_lf = create_admin_search::get_admin_search_lf(
                all_countries_lf.clone(),
                country_info_lf,
            )?;
            let place_search_lf = create_place_search::get_place_search_lf(
                all_countries_lf,
                feature_codes_lf,
                admin_search_lf.clone(),
            )?;
            info!("Collecting transformed data");
            let transform_time = std::time::Instant::now();
            let dfs = collect_all([admin_search_lf, place_search_lf])?;
            info!(
                transform_time = ?transform_time.elapsed()
                , "Transforming data took"
            );
            dfs
        };

        let place_search_df = dfs.pop().expect("Place search should be last");
        let admin_search_df = dfs.pop().expect("Admin search should be first");

        let save_df_to_parquet = |lf: DataFrame, path: &Path| -> Result<()> {
            let sink_time = std::time::Instant::now();

            let mut df = lf
                .lazy()
                .drop_nulls(Some(vec!["geonameId".into()]))
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
        };
        save_df_to_parquet(admin_search_df, &admin_search_path)?;
        save_df_to_parquet(place_search_df, &place_search_path)?;

        Ok(Self::load_parquet_files(
            admin_search_path,
            place_search_path,
        ))
    }

    // ... rest of the methods remain the same ...
    fn load_parquet_files(admin_path: PathBuf, place_path: PathBuf) -> Self {
        Self {
            admin_search_path: admin_path,
            place_search_path: place_path,
            admin_search_df: OnceCell::new(),
            place_search_df: OnceCell::new(),
        }
    }

    fn get_data(path: &Path) -> Result<LazyFrame> {
        info!(
            path = ?path.file_stem(),
            "Loading and collecting into memory for the first time..."
        );
        let t_load = std::time::Instant::now();
        let df = LazyFrame::scan_parquet(path, Default::default())?
            .collect()
            .map(|df| df.lazy())
            .map_err(From::from);
        info!(
            time_collected = ?t_load.elapsed(),
            "Collected into memory"
        );
        df
    }

    pub fn admin_search_df(&self) -> Result<&LazyFrame> {
        self.admin_search_df
            .get_or_try_init(|| Self::get_data(&self.admin_search_path))
    }

    pub fn place_search_df(&self) -> Result<&LazyFrame> {
        self.place_search_df
            .get_or_try_init(|| Self::get_data(&self.place_search_path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_utils::*;

    #[test]
    fn test_location_search_data_new_in_test_env_uses_minimal_test_data_by_default() {
        // This should automatically use minimal test data because it's a cfg(test)
        // and TEST_DATA_SIZE is not set.
        let data = LocationSearchData::new().unwrap();

        assert_eq!(
            data.admin_search_path
                .to_str()
                .unwrap()
                .split('/')
                .next_back(),
            Some("admin_search.parquet"),
        );
        let admin_df = data.admin_search_df().unwrap().clone().collect().unwrap();
        // Minimal config has very few rows
        assert!(
            admin_df.height() >= 1 && admin_df.height() < 10,
            "Expected minimal admin data, got {} rows",
            admin_df.height()
        );
    }

    #[test]
    fn test_create_admin_search_actual_transformation() {
        // Use the actual test files
        let all_countries_file = create_test_all_countries_file();
        let country_info_file = create_test_country_info_file();

        let all_countries_lf =
            crate::raw::all_countries::get_all_countries_df(all_countries_file.path()).unwrap();
        let country_info_lf =
            crate::raw::country_info::get_country_info_df(country_info_file.path()).unwrap();

        // Test the actual transformation
        let result =
            create_admin_search::get_admin_search_lf(all_countries_lf, country_info_lf).unwrap();
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
        let all_countries_file = create_test_all_countries_file();
        let feature_codes_file = create_test_feature_codes_file();

        let all_countries_lf =
            crate::raw::all_countries::get_all_countries_df(all_countries_file.path()).unwrap();
        let feature_codes_lf =
            crate::raw::feature_codes::get_feature_codes_df(feature_codes_file.path()).unwrap();

        // Debug: Check input data
        let all_countries_df = all_countries_lf.clone().collect().unwrap();
        println!(
            "All countries data (before anti-join): {} rows",
            all_countries_df.height()
        );

        // Create empty admin search for anti-join
        let admin_search_lf = all_countries_lf.clone().limit(0);
        let admin_search_df = admin_search_lf.clone().collect().unwrap();
        println!(
            "Admin search data (empty): {} rows",
            admin_search_df.height()
        );

        // Debug: Test the anti-join directly
        let after_anti_join = all_countries_lf
            .clone()
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
        let after_filter = all_countries_lf
            .clone()
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
        println!("After filter data: {:?}", after_filter);

        // Debug: Check if feature_codes parsing is working
        let feature_codes_df = feature_codes_lf.clone().collect().unwrap();
        println!("Feature codes data: {} rows", feature_codes_df.height());
        println!("Feature codes: {:?}", feature_codes_df);

        // Debug the actual transformation - run it but collect intermediate steps
        let result = create_place_search::get_place_search_lf(
            all_countries_lf,
            feature_codes_lf,
            admin_search_lf,
        )
        .unwrap();

        let df = result.collect().unwrap();
        println!("Final result: {} rows", df.height());
        if df.height() > 0 {
            println!("Final data: {:?}", df);
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

    #[test]
    fn test_get_df_caching_mechanism() {
        // Create a test parquet file
        let test_data = df![
            "test_col" => [1, 2, 3],
            "another_col" => ["a", "b", "c"]
        ]
        .unwrap();

        let temp_file = tempfile::NamedTempFile::new().unwrap();

        // Write as parquet, not CSV
        let mut file = std::fs::File::create(temp_file.path()).unwrap();
        polars::prelude::ParquetWriter::new(&mut file)
            .finish(&mut test_data.clone())
            .unwrap();

        // Test that get_df loads the file correctly
        let result = LocationSearchData::get_data(temp_file.path()).unwrap();
        let df = result.collect().unwrap();

        assert_eq!(df.height(), 3);
        assert_has_columns(&df, &["test_col", "another_col"]);
    }
}
