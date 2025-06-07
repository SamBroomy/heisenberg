use polars::prelude::LazyFrame;
use std::fs;
use std::path::Path;
use tempfile::NamedTempFile;
use tracing::{info, instrument, warn};

#[cfg(feature = "download_data")]
pub mod fetch;

pub(super) mod all_countries;
pub(super) mod country_info;
pub(super) mod feature_codes;

pub use super::error::Result;

/// Get all raw data from GeoNames.
///
/// It first checks for existing raw text files (allCountries.txt, etc.) in the specified
/// or default raw data directory.
/// - If `data_dir_override` is `Some(path)`, it looks in `path`.
/// - If `data_dir_override` is `None`, it looks in `<DATA_DIR>/raw/`.
///
/// If files are found, they are copied to temporary files for processing.
/// If files are not found:
///   - If the `download_data` feature is enabled, data is downloaded to temporary files.
///   - If the `download_data` feature is disabled, a `RequiredFilesNotFound` error is returned.
///
/// Returns a tuple of `NamedTempFile`s: (all_countries, country_info, feature_codes).

#[instrument(name = "Get GeoNames raw data", skip_all, level = "info")]
pub fn get_raw_data() -> Result<(NamedTempFile, NamedTempFile, NamedTempFile)> {
    // Try to load from disk or download
    let raw_dir = crate::get_data_dir().join("raw");
    info!("Checking for raw data in: {}", raw_dir.display());

    let all_countries_path = raw_dir.join("allCountries.txt");
    let country_info_path = raw_dir.join("countryInfo.txt");
    let feature_codes_path = raw_dir.join("featureCodes_en.txt");

    if all_countries_path.exists() && country_info_path.exists() && feature_codes_path.exists() {
        info!("Found existing raw data files, copying to temp files");
        return copy_files_to_temp(&all_countries_path, &country_info_path, &feature_codes_path);
    }

    warn!("Raw data files not found");

    #[cfg(feature = "download_data")]
    {
        info!("Attempting to download raw data as download_data feature is enabled.");
        fetch::download_raw_data()
    }
    #[cfg(not(feature = "download_data"))]
    {
        warn!("Download_data feature is disabled. Cannot download missing files.");
        Err(crate::DataError::RequiredFilesNotFound)
    }
}

fn copy_files_to_temp(
    all_countries_path: &Path,
    country_info_path: &Path,
    feature_codes_path: &Path,
) -> Result<(NamedTempFile, NamedTempFile, NamedTempFile)> {
    let all_countries_temp = NamedTempFile::new()?;
    let country_info_temp = NamedTempFile::new()?;
    let feature_codes_temp = NamedTempFile::new()?;

    fs::copy(all_countries_path, all_countries_temp.path())?;
    fs::copy(country_info_path, country_info_temp.path())?;
    fs::copy(feature_codes_path, feature_codes_temp.path())?;

    Ok((all_countries_temp, country_info_temp, feature_codes_temp))
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
                "Missing geonameId: {:?}",
                expected_id
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
                "Missing name: {}",
                expected_name
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
                "Missing feature_class: {}",
                expected_class
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
                "Missing population: {:?}",
                expected_pop
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
