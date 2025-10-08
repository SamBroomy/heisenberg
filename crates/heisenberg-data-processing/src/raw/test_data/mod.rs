use std::io::Write;

use tempfile::NamedTempFile;
use tracing::info;

use crate::raw::fetch::TempData;

pub fn create_test_data() -> TempData {
    info!("Creating test data files");

    TempData::from_temp_files(
        places_data_test_data(),
        country_info_test_data(),
        admin1_test_data(),
        admin2_test_data(),
        feature_codes_test_data(),
    )
}

fn places_data_test_data() -> NamedTempFile {
    write_bytes_to_temp_file(GEONAME_DATA.as_bytes())
}

fn country_info_test_data() -> NamedTempFile {
    write_bytes_to_temp_file(COUNTRY_INFO_DATA.as_bytes())
}

fn admin1_test_data() -> NamedTempFile {
    write_bytes_to_temp_file(ADMIN1_DATA.as_bytes())
}

fn admin2_test_data() -> NamedTempFile {
    write_bytes_to_temp_file(ADMIN2_DATA.as_bytes())
}

fn feature_codes_test_data() -> NamedTempFile {
    write_bytes_to_temp_file(FEATURE_CODES_DATA.as_bytes())
}

fn write_bytes_to_temp_file(data: &[u8]) -> NamedTempFile {
    let mut file = NamedTempFile::new()
        .expect("Environment must support creating temp files - check /tmp permissions");
    file.write_all(data)
        .expect("Writing static test data to temp file should not fail - check disk space");
    file.flush().expect("Flushing temp file should not fail");
    file
}

static ADMIN1_DATA: &str = include_str!("samples/admin1_codes_sample.txt");
static ADMIN2_DATA: &str = include_str!("samples/admin2_codes_sample.txt");
static COUNTRY_INFO_DATA: &str = include_str!("samples/country_info_sample.txt");
static FEATURE_CODES_DATA: &str = include_str!("samples/feature_codes_en_sample.txt");
static GEONAME_DATA: &str = include_str!("samples/geoname_sample.txt");
