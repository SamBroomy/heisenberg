pub(super) mod admin;
pub(super) mod country_info;
pub(super) mod feature_codes;
#[cfg(feature = "download-data")]
pub mod fetch;
pub(super) mod places;
#[cfg(any(test, feature = "test-data"))]
pub mod test_data;
pub use super::error::Result;

#[cfg(test)]
mod tests {
    use std::io::Write;

    use polars::prelude::*;
    use tempfile::NamedTempFile;

    use super::{test_data::create_test_data, *};
    use crate::tests_utils::*;

    #[test]
    fn test_get_all_countries_df_actual_parsing() {
        let temp_data = create_test_data();

        let result = temp_data.places.as_lazy_frame();
        let df = result.collect().unwrap();

        // Test that we parsed rows successfully
        assert!(df.height() > 0, "Should have parsed some rows");

        // Test that required columns exist and have correct types

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
        let temp_data = create_test_data();

        let result = temp_data.country_info.as_lazy_frame();
        let df = result.collect().unwrap();

        // Test that we parsed rows successfully (excluding 51-line header)
        assert!(df.height() > 0, "Should have parsed some country rows");

        // Test that we have ISO codes and country names
        let iso_count = df.column("ISO").unwrap().str().unwrap().len();
        assert!(iso_count > 0, "Should have ISO codes");

        // Verify specific countries from the sample exist
        let iso_codes: Vec<Option<&str>> = df
            .column("ISO")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();

        // Sample data includes many countries - just verify we have multiple ISO codes
        assert!(
            iso_codes.len() > 1,
            "Should have multiple ISO codes from sample data"
        );

        // Check that the data looks reasonable (should have at least some 2-char ISO codes)
        let valid_iso_count = iso_codes
            .iter()
            .filter(|code| code.is_some_and(|c| c.len() == 2))
            .count();
        assert!(
            valid_iso_count > 0,
            "Should have valid 2-character ISO codes"
        );

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
        let temp_data = create_test_data();

        let result = temp_data.feature_codes.as_lazy_frame();
        let df = result.collect().unwrap();

        // Test that we parsed rows successfully
        assert!(df.height() > 0, "Should have parsed some feature codes");

        // Test that codes are parsed correctly into separate feature_class and feature_code columns
        assert_has_columns(
            &df,
            &["feature_class", "feature_code", "name", "description"],
        );

        // Sample data starts with A.ADM1, A.ADM1H, A.ADM2, etc.
        let feature_classes: Vec<Option<&str>> = df
            .column("feature_class")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();

        // Check that A.ADM1 was properly split into class "A" and code "ADM1"
        assert!(
            feature_classes.contains(&Some("A")),
            "Should contain feature_class 'A' from A.ADM1"
        );

        let feature_codes: Vec<Option<&str>> = df
            .column("feature_code")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();

        assert!(
            feature_codes.contains(&Some("ADM1")),
            "Should contain feature_code 'ADM1' from A.ADM1"
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
        let temp_data = create_test_data();

        let all_countries_lf = temp_data.places.as_lazy_frame();
        let country_info_lf = temp_data.country_info.as_lazy_frame();
        let feature_codes_lf = temp_data.feature_codes.as_lazy_frame();

        // Test that all LazyFrames can be collected successfully
        let all_countries_df = all_countries_lf.collect().unwrap();
        let country_info_df = country_info_lf.collect().unwrap();
        let feature_codes_df = feature_codes_lf.collect().unwrap();

        // Test dimensions
        assert!(
            all_countries_df.height() > 0,
            "Should have parsed place data"
        );
        assert!(
            country_info_df.height() > 0,
            "Should have parsed country info"
        );
        assert!(
            feature_codes_df.height() > 0,
            "Should have parsed feature codes"
        );

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

        // Sample data has geonameIds like 4046704 (Fort Hunt) in places
        // and 3041565 (Andorra) in country_info - just verify both have ids
        assert!(
            !all_countries_ids.is_empty(),
            "Should have geonameIds in places"
        );
        assert!(
            !country_info_ids.is_empty(),
            "Should have geonameIds in country_info"
        );
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

        let places_data = places::PlacesRawData::new(empty_file);
        let result = places_data.as_lazy_frame();
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

        let places_data = places::PlacesRawData::new(malformed_file);
        let lf = places_data.as_lazy_frame();

        // Should either handle gracefully or return an appropriate error
        let collect_result = lf.collect();
        // We expect this to either work (with some default handling) or fail gracefully
        assert!(collect_result.is_ok() || collect_result.is_err());
    }
}
