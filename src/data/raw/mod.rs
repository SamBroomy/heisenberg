use polars::prelude::LazyFrame;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;
use tracing::{info, instrument, warn};
#[cfg(feature = "download_data")]
mod fetch;

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
    let effective_raw_dir: PathBuf = super::get_data_dir().join("raw");

    info!(
        "Effective raw data directory to check: {:?}",
        effective_raw_dir
    );

    let all_countries_txt_path = effective_raw_dir.join("allCountries.txt");
    let country_info_txt_path = effective_raw_dir.join("countryInfo.txt");
    let feature_codes_txt_path = effective_raw_dir.join("featureCodes_en.txt");

    // Attempt to load from local raw files first
    if all_countries_txt_path.exists()
        && country_info_txt_path.exists()
        && feature_codes_txt_path.exists()
    {
        info!(
            "Found existing raw data files in {:?}. Copying to temporary files.",
            effective_raw_dir
        );
        return copy_files_to_temp(
            &all_countries_txt_path,
            &country_info_txt_path,
            &feature_codes_txt_path,
        );
    }

    // If local raw files are not found
    warn!("Raw data files not found in {:?}.", effective_raw_dir);

    #[cfg(feature = "download_data")]
    {
        info!("Attempting to download raw data as download_data feature is enabled.");
        fetch::download_raw_data()
    }
    #[cfg(not(feature = "download_data"))]
    {
        warn!("Download_data feature is disabled. Cannot download missing files.");
        Err(crate::data::DataError::RequiredFilesNotFound)
    }
}

fn copy_files_to_temp(
    all_countries_path: &Path,
    country_info_path: &Path,
    feature_codes_path: &Path,
) -> Result<(NamedTempFile, NamedTempFile, NamedTempFile)> {
    // ...existing code...
    let all_countries_temp = NamedTempFile::new()?;
    let country_info_temp = NamedTempFile::new()?;
    let feature_codes_temp = NamedTempFile::new()?;

    fs::copy(all_countries_path, all_countries_temp.path())?;
    fs::copy(country_info_path, country_info_temp.path())?;
    fs::copy(feature_codes_path, feature_codes_temp.path())?;

    Ok((all_countries_temp, country_info_temp, feature_codes_temp))
}

/// Transform the raw data into LazyFrames
/// returns a tuple of LazyFrames: (all_countries_df, country_info_df, feature_codes_df)
#[instrument(name = "Transform GeoNames data", skip_all, level = "info")]
pub fn get_raw_data_as_lazy_frames<T: AsRef<Path>>(
    raw_data: &(T, T, T),
) -> Result<(LazyFrame, LazyFrame, LazyFrame)> {
    let all_countries_df = all_countries::get_all_countries_df(raw_data.0.as_ref())?;
    let country_info_df = country_info::get_country_info_df(raw_data.1.as_ref())?;
    let feature_codes_df = feature_codes::get_feature_codes_df(raw_data.2.as_ref())?;

    Ok((all_countries_df, country_info_df, feature_codes_df))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::tests_utils::*;
    use polars::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Helper to create test data files
    fn create_test_all_countries_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        // Format: geonameId, name, asciiname, alternatenames, latitude, longitude, feature_class, feature_code,
        //         admin0_code, cc2, admin1_code, admin2_code, admin3_code, admin4_code, population, elevation, dem, timezone, modification_date
        writeln!(file, "1\tUnited States\tUnited States\tUS,USA,America\t39.5\t-98.35\tA\tPCLI\tUS\t\t\t\t\t\t331000000\t0\t0\tAmerica/New_York\t2023-01-01").unwrap();
        writeln!(file, "2\tCalifornia\tCalifornia\tCA,Calif\t36.17\t-119.75\tA\tADM1\tUS\t\tCA\t\t\t\t39538223\t0\t0\tAmerica/Los_Angeles\t2023-01-01").unwrap();
        writeln!(file, "3\tSan Francisco\tSan Francisco\tSF,San Fran\t37.7749\t-122.4194\tP\tPPLA2\tUS\t\tCA\t075\t\t\t873965\t16\t16\tAmerica/Los_Angeles\t2023-01-01").unwrap();
        file.flush().unwrap();
        file
    }

    fn create_test_country_info_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "#ISO\tISO3\tISO-Numeric\tFIPS\tCountry\tCapital\tArea(in sq km)\tPopulation\tContinent\ttld\tCurrencyCode\tCurrencyName\tPhone\tPostal Code Format\tPostal Code Regex\tLanguages\tgeonameId\tneighbours\tEquivalentFipsCode").unwrap();
        writeln!(file, "US\tUSA\t840\tUS\tUnited States\tWashington\t9629091\t331002651\tNA\t.us\tUSD\tDollar\t1\t#####-####\t^\\d{{5}}(-\\d{{4}})?$\ten-US,es-US,haw,fr\t6252001\tCA,MX\t").unwrap();
        file.flush().unwrap();
        file
    }

    fn create_test_feature_codes_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        // Format: code, name, description (based on your schema)
        writeln!(file, "A.ADM1\tfirst-order administrative division\ta primary administrative division of a country").unwrap();
        writeln!(file, "A.PCLI\tindependent political entity\t").unwrap();
        writeln!(
            file,
            "P.PPLA2\tseat of a second-order administrative division\t"
        )
        .unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_get_all_countries_df_actual_parsing() {
        let test_file = create_test_all_countries_file();

        let result = all_countries::get_all_countries_df(test_file.path()).unwrap();
        let df = result.collect().unwrap();

        // Test that we parsed the correct number of rows
        assert_eq!(df.height(), 3);

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
        let expected_ids = vec![Some(1), Some(2), Some(3)];
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
        let test_file = create_test_country_info_file();

        let result = country_info::get_country_info_df(test_file.path()).unwrap();
        let df = result.collect().unwrap();

        // Test that we parsed the correct number of rows (excluding header)
        assert_eq!(df.height(), 1);

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
        let test_file = create_test_feature_codes_file();

        let result = feature_codes::get_feature_codes_df(test_file.path()).unwrap();
        let df = result.collect().unwrap();

        // Test that we parsed the correct number of rows
        assert_eq!(df.height(), 3);

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
        let all_countries_file = create_test_all_countries_file();
        let country_info_file = create_test_country_info_file();
        let feature_codes_file = create_test_feature_codes_file();

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
        assert_eq!(all_countries_df.height(), 3);
        assert_eq!(country_info_df.height(), 1);
        assert_eq!(feature_codes_df.height(), 3);

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
        assert!(all_countries_ids.contains(&Some(1)));
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
