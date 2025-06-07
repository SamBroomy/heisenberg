use once_cell::sync::Lazy;
use std::path::PathBuf;
use tracing::warn;

pub mod processed;
pub mod raw;
pub mod test_data;

// DataError will be imported from the error module below

static TEST_DATA_DIR: Lazy<tempfile::TempDir> = Lazy::new(|| {
    tempfile::TempDir::new().expect("Failed to create global temporary test data directory")
});

pub const DATA_DIR_DEFAULT: &str = "./hberg_data";

/// Centralized function to determine if we should use test data.
pub fn should_use_test_data() -> bool {
    let is_test_environment = cfg!(test) || cfg!(doctest);

    #[cfg(feature = "test_data")]
    let explicit_test_data = std::env::var("USE_TEST_DATA")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    #[cfg(not(feature = "test_data"))]
    let explicit_test_data = false;

    is_test_environment || explicit_test_data
}

/// Get the appropriate test data config based on environment.
#[cfg(any(test, doctest, feature = "test_data"))]
pub fn get_test_data_config() -> test_data::TestDataConfig {
    if cfg!(doctest) {
        test_data::TestDataConfig::minimal()
    } else {
        match std::env::var("TEST_DATA_SIZE").as_deref() {
            Ok("sample") => test_data::TestDataConfig::sample(),
            _ => test_data::TestDataConfig::minimal(),
        }
    }
}

/// Global data directory path that automatically determines the appropriate location.
pub static DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
    if should_use_test_data() {
        let temp_dir = TEST_DATA_DIR.path().to_path_buf();
        warn!(temp_dir = ?temp_dir, "Using temporary data directory for tests");
        temp_dir
    } else {
        let dir = std::env::var("DATA_DIR").unwrap_or_else(|_| DATA_DIR_DEFAULT.to_string());
        PathBuf::from(dir)
    }
});

mod error {
    use polars::prelude::PolarsError;
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum DataError {
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),
        #[error("Polars error: {0}")]
        Polars(#[from] PolarsError),
        #[cfg(feature = "download_data")]
        #[error("HTTP error: {0}")]
        Http(#[from] reqwest::Error),
        #[cfg(feature = "download_data")]
        #[error("Join error: {0}")]
        JoinError(#[from] tokio::task::JoinError),
        #[cfg(feature = "download_data")]
        #[error("Zip error: {0}")]
        ZipError(#[from] zip::result::ZipError),
        #[error("No data directory provided and download_data feature is disabled")]
        NoDataDirProvided,
        #[error("Required data files not found in the provided directory")]
        RequiredFilesNotFound,
    }

    pub type Result<T> = std::result::Result<T, DataError>;
}

pub use error::{DataError, Result};

// Re-export main types
pub use processed::LocationSearchData;
pub use test_data::{TestDataConfig, create_test_data};

/// Generate embedded dataset for build-time inclusion
///
/// This function can be called from build.rs to create optimized datasets
/// that ship with the library binary.
pub fn generate_embedded_dataset(source: EmbeddedDataSource) -> Result<EmbeddedDataset> {
    match source {
        EmbeddedDataSource::TestData(config) => generate_from_test_data(config),
        EmbeddedDataSource::Cities15000 => generate_from_cities15000(),
        EmbeddedDataSource::CustomFilter {
            min_population,
            countries,
        } => generate_from_custom_filter(min_population, countries),
    }
}

#[derive(Debug, Clone)]
pub enum EmbeddedDataSource {
    /// Use enhanced test data (good for development)
    TestData(TestDataConfig),
    /// Download and process cities15000.zip (production recommended)
    Cities15000,
    /// Custom filtering criteria
    CustomFilter {
        min_population: u32,
        countries: Option<Vec<String>>,
    },
}

#[derive(Debug)]
pub struct EmbeddedDataset {
    pub admin_data: polars::prelude::DataFrame,
    pub place_data: polars::prelude::DataFrame,
    pub metadata: EmbeddedMetadata,
}

#[derive(Debug, Clone)]
pub struct EmbeddedMetadata {
    pub version: String,
    pub source: String,
    pub generated_at: String,
    pub description: String,
    pub admin_rows: usize,
    pub place_rows: usize,
    pub size_bytes: usize,
}

fn generate_from_test_data(config: TestDataConfig) -> Result<EmbeddedDataset> {
    tracing::info!(
        "Generating embedded dataset from test data with config: {:?}",
        config
    );

    let (all_countries_file, country_info_file, feature_codes_file) =
        test_data::create_test_data(&config)?;
    let (all_countries_lf, country_info_lf, feature_codes_lf) = raw::get_raw_data_as_lazy_frames(
        &(all_countries_file, country_info_file, feature_codes_file),
    )?;

    let admin_lf = processed::create_admin_search::get_admin_search_lf(
        all_countries_lf.clone(),
        country_info_lf,
    )?;
    let place_lf = processed::create_place_search::get_place_search_lf(
        all_countries_lf,
        feature_codes_lf,
        admin_lf.clone(),
    )?;

    let admin_data = admin_lf.collect()?;
    let place_data = place_lf.collect()?;

    let metadata = EmbeddedMetadata {
        version: "1.0.0".to_string(),
        source: "enhanced_test_data".to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        description: "Embedded dataset from enhanced test data".to_string(),
        admin_rows: admin_data.height(),
        place_rows: place_data.height(),
        size_bytes: 0, // Will be calculated when serializing
    };

    Ok(EmbeddedDataset {
        admin_data,
        place_data,
        metadata,
    })
}

fn generate_from_cities15000() -> Result<EmbeddedDataset> {
    tracing::info!("Generating embedded dataset from cities15000.zip");

    // This will download cities15000.zip and process it
    // For now, fall back to test data until we implement the download logic
    generate_from_test_data(TestDataConfig::sample())
}

fn generate_from_custom_filter(
    min_population: u32,
    countries: Option<Vec<String>>,
) -> Result<EmbeddedDataset> {
    tracing::info!(
        "Generating embedded dataset with custom filter: min_pop={}, countries={:?}",
        min_population,
        countries
    );

    // This will apply custom filtering to a full dataset
    // For now, fall back to test data
    generate_from_test_data(TestDataConfig::sample())
}

#[cfg(test)]
pub(crate) mod tests_utils {
    use num_traits::NumCast;
    use polars::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    pub fn assert_has_columns(df: &DataFrame, expected_columns: &[&str]) {
        let actual_columns: Vec<_> = df.get_column_names().iter().map(|s| s.as_str()).collect();
        for expected_col in expected_columns {
            assert!(
                actual_columns.contains(expected_col),
                "Missing column: {}. Available columns: {:?}",
                expected_col,
                actual_columns
            );
        }
    }

    pub fn assert_column_type(df: &DataFrame, column: &str, expected_type: &DataType) {
        let actual_type = df
            .column(column)
            .unwrap_or_else(|_| panic!("Column '{}' not found", column))
            .dtype();
        assert_eq!(
            actual_type, expected_type,
            "Column '{}' has wrong type. Expected: {:?}, Got: {:?}",
            column, expected_type, actual_type
        );
    }

    pub fn assert_no_nulls_in_column(df: &DataFrame, column: &str) {
        let null_count = df
            .column(column)
            .unwrap_or_else(|_| panic!("Column '{}' not found", column))
            .null_count();
        assert_eq!(
            null_count, 0,
            "Column '{}' contains {} null values",
            column, null_count
        );
    }

    pub fn assert_column_range<T>(df: &DataFrame, column: &str, min_val: T, max_val: T)
    where
        T: std::fmt::Debug + NumCast + std::cmp::PartialOrd + Clone + 'static,
    {
        let series = df
            .column(column)
            .unwrap_or_else(|_| panic!("Column '{}' not found", column));

        if let (Ok(min_actual), Ok(max_actual)) = (
            series
                .max_reduce()
                .and_then(|m| m.as_any_value().try_extract::<T>()),
            series
                .max_reduce()
                .and_then(|m| m.as_any_value().try_extract::<T>()),
        ) {
            assert!(
                min_actual >= min_val,
                "Column '{}' min value {:?} is below expected minimum {:?}",
                column,
                min_actual,
                min_val
            );
            assert!(
                max_actual <= max_val,
                "Column '{}' max value {:?} is above expected maximum {:?}",
                column,
                max_actual,
                max_val
            );
        }
    }

    pub fn create_test_all_countries_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "6252001\tUnited States\tUnited States\tUS,USA,America\t39.5\t-98.35\tA\tPCLI\tUS\t\t\t\t\t\t331000000\t0\t0\tAmerica/New_York\t2023-01-01").unwrap();
        writeln!(file, "5332921\tCalifornia\tCalifornia\tCA,Calif\t36.17\t-119.75\tA\tADM1\tUS\t\tCA\t\t\t\t39538223\t0\t0\tAmerica/Los_Angeles\t2023-01-01").unwrap();
        writeln!(file, "5391959\tSan Francisco\tSan Francisco\tSF,San Fran\t37.7749\t-122.4194\tP\tPPLA2\tUS\t\tCA\t075\t\t\t873965\t16\t16\tAmerica/Los_Angeles\t2023-01-01").unwrap();
        file.flush().unwrap();
        file
    }

    pub fn create_test_country_info_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=51 {
            writeln!(file, "# Header line {}", i).unwrap();
        }
        writeln!(file, "US\tUSA\t840\tUS\tUnited States\tWashington\t9629091\t331002651\tNA\t.us\tUSD\tDollar\t1\t#####-####\t^\\d{{5}}(-\\d{{4}})?$\ten-US,es-US,haw,fr\t6252001\tCA,MX\t").unwrap();
        file.flush().unwrap();
        file
    }

    pub fn create_test_feature_codes_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_embedded_from_test_data() {
        let source = EmbeddedDataSource::TestData(TestDataConfig::minimal());
        let dataset = generate_embedded_dataset(source).expect("Should generate dataset");

        assert!(dataset.admin_data.height() > 0, "Should have admin data");
        assert!(dataset.place_data.height() > 0, "Should have place data");
        assert_eq!(dataset.metadata.source, "enhanced_test_data");

        println!(
            "Generated embedded dataset: {} admin rows, {} place rows",
            dataset.metadata.admin_rows, dataset.metadata.place_rows
        );
    }
}
