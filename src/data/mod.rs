use once_cell::sync::Lazy;
pub use processed::LocationSearchData;
use std::path::PathBuf;
use tracing::warn;

mod processed;
mod raw;
#[cfg(any(test, doctest, feature = "test_data"))]
pub mod test_data;
pub(crate) use error::DataError;

static TEST_DATA_DIR: Lazy<tempfile::TempDir> = Lazy::new(|| {
    tempfile::TempDir::new().expect("Failed to create global temporary test data directory")
});
pub const DATA_DIR_DEFAULT: &str = "./hberg_data";

/// Centralized function to determine if we should use test data.
///
/// Returns true if:
/// - We're in a test environment (cfg!(test) or cfg!(doctest))
/// - OR the test_data feature is enabled AND USE_TEST_DATA env var is set to true
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
///
/// - For doctests: always minimal (fast)
/// - For regular tests or explicit test data: configurable via TEST_DATA_SIZE env var
#[cfg(any(test, doctest, feature = "test_data"))]
pub fn get_test_data_config() -> crate::data::test_data::TestDataConfig {
    if cfg!(doctest) {
        crate::data::test_data::TestDataConfig::minimal()
    } else {
        match std::env::var("TEST_DATA_SIZE").as_deref() {
            Ok("sample") => crate::data::test_data::TestDataConfig::sample(),
            _ => crate::data::test_data::TestDataConfig::minimal(),
        }
    }
}

/// Global data directory path that automatically determines the appropriate location.
///
/// The directory is chosen based on the following priority:
/// 1. If running under `cargo test` or `cargo doctest` -> temporary directory (auto-cleanup)
/// 2. If `test_data` feature is enabled AND `USE_TEST_DATA` env var is set -> temporary directory
/// 3. Otherwise -> persistent directory from `DATA_DIR` env var or default `./hberg_data`
pub static DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
    if should_use_test_data() {
        // Use temporary directory - create it once and it will live for the program duration
        let temp_dir = TEST_DATA_DIR.path().to_path_buf();
        warn!(temp_dir = ?temp_dir, "Using temporary data directory for tests");
        temp_dir
    } else {
        // Use persistent directory
        let dir = std::env::var("DATA_DIR").unwrap_or_else(|_| DATA_DIR_DEFAULT.to_string());
        PathBuf::from(dir)
    }
});

// ... rest of the file (error module, tests_utils) remains the same ...
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

#[cfg(test)]
mod tests_utils {
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
