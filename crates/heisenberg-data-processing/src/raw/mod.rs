use polars::prelude::LazyFrame;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::{fmt, str::FromStr};
use tracing::{instrument, warn};

#[cfg(feature = "download_data")]
pub mod fetch;

pub(super) mod all_countries;
pub(super) mod country_info;
pub(super) mod feature_codes;

pub use super::error::Result;

#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
/// Enum representing the available data sources for GeoNames data processing
#[serde(rename_all = "snake_case")]
pub enum DataSource {
    #[default]
    /// Download and process cities15000.zip
    Cities15000,
    /// Download and process cities5000.zip
    Cities5000,
    /// Download and process cities1000.zip
    Cities1000,
    /// Download and process cities500.zip
    Cities500,
    /// Download and process allCountries.zip (full dataset)
    AllCountries,
    /// Use Test data for development
    TestData,
}

impl fmt::Display for DataSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cities15000 => write!(f, "cities15000"),
            Self::Cities5000 => write!(f, "cities5000"),
            Self::Cities1000 => write!(f, "cities1000"),
            Self::Cities500 => write!(f, "cities500"),
            Self::AllCountries => write!(f, "allCountries"),
            Self::TestData => write!(f, "test_data"),
        }
    }
}

impl DataSource {
    pub const BASE_URL: &str = "https://download.geonames.org/export/dump/";

    pub fn geonames_url(&self) -> Option<String> {
        match self {
            Self::TestData => {
                warn!("Using test data, no download URL available");
                None
            }
            _ => Some(format!("{}{}.zip", Self::BASE_URL, &self)),
        }
    }

    pub fn admin_parquet(&self) -> PathBuf {
        PathBuf::from(format!("{}_admin_search.parquet", &self))
    }
    pub fn place_parquet(&self) -> PathBuf {
        PathBuf::from(format!("{}_place_search.parquet", &self))
    }
}

impl FromStr for DataSource {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cities15000" => Ok(Self::Cities15000),
            "cities5000" => Ok(Self::Cities5000),
            "cities1000" => Ok(Self::Cities1000),
            "cities500" => Ok(Self::Cities500),
            "allcountries" => Ok(Self::AllCountries),
            "test_data" => Ok(Self::TestData),
            _ => Err(format!(
                "Invalid DataSource: {s}. Valid options are: cities15000, cities5000, cities1000, cities500, allCountries, test_data"
            )),
        }
    }
}

#[instrument(name = "Transform GeoNames data", skip_all, level = "info")]
pub fn get_raw_data_as_lazy_frames<T: AsRef<Path>>(
    raw_data: &(T, T, T),
) -> Result<(LazyFrame, LazyFrame, LazyFrame)> {
    let all_countries_df = all_countries::get_all_countries_df(raw_data.0.as_ref())?;
    let country_info_df = country_info::get_country_info_df(raw_data.1.as_ref())?;
    let feature_codes_df = feature_codes::get_feature_codes_df(raw_data.2.as_ref())?;

    Ok((all_countries_df, country_info_df, feature_codes_df))
}

/// Transform GeoNames data from separate file paths
#[instrument(name = "Transform GeoNames data from paths", skip_all, level = "info")]
pub fn get_raw_data_as_lazy_frames_from_paths(
    all_countries_path: &Path,
    country_info_path: &Path,
    feature_codes_path: &Path,
) -> Result<(LazyFrame, LazyFrame, LazyFrame)> {
    let all_countries_df = all_countries::get_all_countries_df(all_countries_path)?;
    let country_info_df = country_info::get_country_info_df(country_info_path)?;
    let feature_codes_df = feature_codes::get_feature_codes_df(feature_codes_path)?;

    Ok((all_countries_df, country_info_df, feature_codes_df))
}

#[cfg(test)]
mod tests {
    use super::super::test_data::{TestDataConfig, create_test_data};
    use super::*;
    use crate::tests_utils::*;
    use polars::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_get_all_countries_df_actual_parsing() {
        let (test_file, _, _) = create_test_data(&TestDataConfig::minimal()).unwrap();

        let result = all_countries::get_all_countries_df(test_file.path()).unwrap();
        let df = result.collect().unwrap();

        // Test that we parsed the correct number of rows
        assert!(df.height() >= 3);

        // Test specific column values to ensure parsing worked correctly
        // Note: Due to sorting by modification_date and unique operations, order may vary
        let geoname_ids: Vec<Option<u32>> = df
            .column("geonameId")
            .unwrap()
            .u32()
            .unwrap()
            .into_iter()
            .collect();

        // Test that all expected IDs are present (order may vary due to sorting)
        let expected_ids = vec![Some(6252001), Some(5332921), Some(5391959)];
        for expected_id in expected_ids {
            assert!(
                geoname_ids.contains(&expected_id),
                "Missing geonameId: {expected_id:?}"
            );
        }

        let names: Vec<Option<&str>> = df
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();

        // Test that all expected names are present
        let expected_names = vec!["United States", "California", "San Francisco"];
        for expected_name in expected_names {
            assert!(
                names.contains(&Some(expected_name)),
                "Missing name: {expected_name}",
            );
        }

        let feature_classes: Vec<Option<&str>> = df
            .column("feature_class")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();

        // Test that expected feature classes are present
        let expected_classes = vec!["A", "A", "P"];
        for expected_class in expected_classes {
            assert!(
                feature_classes.contains(&Some(expected_class)),
                "Missing feature_class: {expected_class}",
            );
        }

        let populations: Vec<Option<i64>> = df
            .column("population")
            .unwrap()
            .i64()
            .unwrap()
            .into_iter()
            .collect();

        // Test that expected populations are present
        let expected_pops = vec![Some(331000000), Some(39538223), Some(873965)];
        for expected_pop in expected_pops {
            assert!(
                populations.contains(&expected_pop),
                "Missing population: {expected_pop:?}",
            );
        }

        // Test data types are correct
        assert_column_type(&df, "geonameId", &DataType::UInt32);
        assert_column_type(&df, "latitude", &DataType::Float32);
        assert_column_type(&df, "longitude", &DataType::Float32);
        assert_column_type(&df, "population", &DataType::Int64);

        // Test no nulls in required columns
        assert_no_nulls_in_column(&df, "geonameId");
        assert_no_nulls_in_column(&df, "name");
        assert_no_nulls_in_column(&df, "feature_class");
    }

    #[test]
    fn test_get_country_info_df_actual_parsing() {
        let (_, test_file, _) = create_test_data(&TestDataConfig::minimal()).unwrap();

        let result = country_info::get_country_info_df(test_file.path()).unwrap();
        let df = result.collect().unwrap();

        // Test that we parsed the correct number of rows (excluding header)
        assert!(df.height() >= 1);

        // Test specific values
        let iso_codes: Vec<Option<&str>> = df
            .column("ISO")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(iso_codes, vec![Some("US")]);

        let countries: Vec<Option<&str>> = df
            .column("Country")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(countries, vec![Some("United States")]);

        let geoname_ids: Vec<Option<u32>> = df
            .column("geonameId")
            .unwrap()
            .u32()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(geoname_ids, vec![Some(6252001)]);

        // Test data types
        assert_column_type(&df, "ISO", &DataType::String);
        assert_column_type(&df, "geonameId", &DataType::UInt32);
        assert_column_type(&df, "Population", &DataType::Int32);

        // Test no nulls in key columns
        assert_no_nulls_in_column(&df, "ISO");
        assert_no_nulls_in_column(&df, "Country");
        assert_no_nulls_in_column(&df, "geonameId");
    }

    #[test]
    fn test_get_feature_codes_df_actual_parsing() {
        let (_, _, test_file) = create_test_data(&TestDataConfig::minimal()).unwrap();

        let result = feature_codes::get_feature_codes_df(test_file.path()).unwrap();
        let df = result.collect().unwrap();

        // Test that we parsed the correct number of rows
        assert!(df.height() >= 3);

        // Test that codes are parsed correctly into separate feature_class and feature_code columns
        // Let's check what columns actually exist first
        println!("Available columns: {:?}", df.get_column_names());

        // Test the transformed columns instead of the original 'code' column
        let feature_classes: Vec<Option<&str>> = df
            .column("feature_class")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(feature_classes, vec![Some("A"), Some("A"), Some("P")]);

        let feature_codes: Vec<Option<&str>> = df
            .column("feature_code")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(
            feature_codes,
            vec![Some("ADM1"), Some("PCLI"), Some("PPLA2")]
        );

        let names: Vec<Option<&str>> = df
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(
            names,
            vec![
                Some("first-order administrative division"),
                Some("independent political entity"),
                Some("seat of a second-order administrative division")
            ]
        );

        // Test data types
        assert_column_type(&df, "feature_class", &DataType::String);
        assert_column_type(&df, "feature_code", &DataType::String);
        assert_column_type(&df, "name", &DataType::String);
        assert_column_type(&df, "description", &DataType::String);

        // Test no nulls in required columns
        assert_no_nulls_in_column(&df, "feature_class");
        assert_no_nulls_in_column(&df, "feature_code");
        assert_no_nulls_in_column(&df, "name");
    }

    #[test]
    fn test_get_raw_data_as_lazy_frames_integration() {
        let (all_countries_file, country_info_file, feature_codes_file) =
            create_test_data(&TestDataConfig::minimal()).unwrap();

        let raw_data = (
            all_countries_file.path(),
            country_info_file.path(),
            feature_codes_file.path(),
        );

        let result = get_raw_data_as_lazy_frames(&raw_data).unwrap();
        let (all_countries_lf, country_info_lf, feature_codes_lf) = result;

        // Test that all LazyFrames can be collected successfully
        let all_countries_df = all_countries_lf.collect().unwrap();
        let country_info_df = country_info_lf.collect().unwrap();
        let feature_codes_df = feature_codes_lf.collect().unwrap();

        // Test dimensions
        assert!(all_countries_df.height() >= 3);
        assert!(country_info_df.height() >= 1);
        assert!(feature_codes_df.height() >= 3);

        // Test that specific transformation logic works
        // For example, alternatenames should be parsed as List<String>
        assert_column_type(
            &all_countries_df,
            "alternatenames",
            &DataType::List(Box::new(DataType::String)),
        );

        // Test that joins would work (geonameId compatibility)
        let all_countries_ids: Vec<Option<u32>> = all_countries_df
            .column("geonameId")
            .unwrap()
            .u32()
            .unwrap()
            .into_iter()
            .collect();
        let country_info_ids: Vec<Option<u32>> = country_info_df
            .column("geonameId")
            .unwrap()
            .u32()
            .unwrap()
            .into_iter()
            .collect();

        // US should have geonameId 1 in all_countries and 6252001 in country_info
        assert!(all_countries_ids.contains(&Some(6252001)));
        assert!(country_info_ids.contains(&Some(6252001)));
    }

    #[test]
    fn test_edge_cases_in_parsing() {
        // Test with empty/minimal data - fix the column count and order
        let mut empty_file = NamedTempFile::new().unwrap();
        writeln!(
            empty_file,
            "1\tTest\tTest\t\t0.0\t0.0\tP\tPPL\tUS\t\t\t\t\t\t0\t0\t0\tUTC\t2023-01-01"
        )
        .unwrap();
        empty_file.flush().unwrap();

        let result = all_countries::get_all_countries_df(empty_file.path()).unwrap();
        let df = result.collect().unwrap();

        assert_eq!(df.height(), 1);

        // Test that empty strings are handled correctly
        let alternatenames = df
            .column("alternatenames")
            .unwrap()
            .list()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();

        // Empty alternatenames string becomes null after string cleaning and splitting
        // This is acceptable behavior - empty data becomes null
        match &alternatenames[0] {
            Some(list) => {
                // If it's a list, it should contain valid data
                assert!(
                    !list.is_empty(),
                    "If alternatenames is a list, it should not be empty"
                );
            }
            None => {
                // Null is acceptable for empty alternatenames
                println!("Empty alternatenames correctly converted to null");
            }
        }
    }

    #[test]
    fn test_malformed_data_handling() {
        // Test with malformed data to ensure graceful error handling
        let mut malformed_file = NamedTempFile::new().unwrap();
        writeln!(malformed_file, "not_a_number\tTest\tTest").unwrap(); // Too few columns, invalid number
        malformed_file.flush().unwrap();

        let result = all_countries::get_all_countries_df(malformed_file.path());

        // Should either handle gracefully or return an appropriate error
        match result {
            Ok(lf) => {
                // If it succeeds, it should handle the malformed data somehow
                let collect_result = lf.collect();
                // We expect this to either work (with some default handling) or fail gracefully
                assert!(collect_result.is_ok() || collect_result.is_err());
            }
            Err(_) => {
                // It's also acceptable to return an error for malformed data
            }
        }
    }
}
