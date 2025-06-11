use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
    sync::LazyLock,
};

pub use error::{DataError, Result};
use polars::prelude::LazyFrame;
use serde::{Deserialize, Serialize};
pub use test_data::{TestDataConfig, create_test_data};
use tracing::{info, warn};

use crate::processed::{generate_processed_data, save_processed_data_to_parquet};

pub mod embedded;
pub mod error;
pub mod processed;
pub mod raw;
pub mod test_data;

pub const DATA_DIR_DEFAULT: &str = "heisenberg_data";

pub static DATA_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    std::env::var("DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| get_default_data_dir())
});

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
    pub const PROCESSED_DIR: &str = "processed";

    pub fn data_source_dir(&self) -> PathBuf {
        DATA_DIR.join(self.to_string())
    }

    fn processed_dir(&self) -> PathBuf {
        self.data_source_dir().join(Self::PROCESSED_DIR)
    }

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
        self.processed_dir().join("admin_search.parquet")
    }

    pub fn place_parquet(&self) -> PathBuf {
        self.processed_dir().join("place_search.parquet")
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
            "test_data" | "test" => Ok(Self::TestData),
            _ => Err(format!(
                "Invalid DataSource: {s}. Valid options are: cities15000, cities5000, cities1000, cities500, allCountries, test_data"
            )),
        }
    }
}

static TEST_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

static TEST_DIR: LazyLock<tempfile::TempDir> =
    LazyLock::new(|| tempfile::tempdir().expect("Failed to create temporary test directory"));

/// Get the default data directory based on environment and platform
fn get_default_data_dir() -> PathBuf {
    // Check if we're in a doctest environment
    if std::env::var("CARGO_TARGET_TMPDIR").is_ok() {
        // In doctests, create a unique directory
        let test_id = TEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        return TEST_DIR
            .path()
            .join(format!("heisenberg_doctest_{test_id}"));
    }
    #[cfg(any(test, doctest))]
    {
        TEST_DIR.path().to_path_buf().join(format!(
            "heisenberg_data_test_{}",
            TEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ))
    }
    #[cfg(not(any(test, doctest)))]
    {
        // 1. Check for explicit override
        if let Ok(data_dir) = std::env::var("HEISENBERG_DATA_DIR") {
            return PathBuf::from(data_dir);
        }

        // 2. Check for workspace root via environment (set by build scripts)
        if let Ok(workspace_root) = std::env::var("CARGO_WORKSPACE_DIR") {
            return PathBuf::from(workspace_root).join(DATA_DIR_DEFAULT);
        }

        // 3. Development detection
        if std::env::var("CARGO_PKG_NAME").is_ok() {
            // We're being built by cargo, use relative to workspace
            return PathBuf::from(format!("../../{DATA_DIR_DEFAULT}"));
        }

        // 4. Production: use system directories
        #[cfg(feature = "system-dirs")]
        {
            if let Some(proj_dirs) = directories::ProjectDirs::from("com", "yourorg", "heisenberg")
            {
                return proj_dirs.cache_dir().to_path_buf();
            }
        }

        // 5. Final fallback
        PathBuf::from(format!("./{DATA_DIR_DEFAULT}"))
    }
}

fn load_single_parquet_file(path: &Path) -> Result<LazyFrame> {
    LazyFrame::scan_parquet(path, Default::default()).map_err(Into::into)
}

fn load_parquet_files(admin_path: &Path, place_path: &Path) -> Result<(LazyFrame, LazyFrame)> {
    let admin_df = load_single_parquet_file(admin_path)?;
    let place_df = load_single_parquet_file(place_path)?;
    Ok((admin_df, place_df))
}

/// Check if both admin and place files exist and are readable
fn validate_data_files(data_source: &DataSource) -> Result<(PathBuf, PathBuf)> {
    let admin_path = data_source.admin_parquet();
    let place_path = data_source.place_parquet();

    // Check if both files exist
    if !admin_path.exists() || !place_path.exists() {
        return Err(DataError::RequiredFilesNotFound);
    }

    // Try to validate files by attempting to read their metadata
    if let Err(e) = LazyFrame::scan_parquet(&admin_path, Default::default()) {
        warn!("Admin file corrupted or unreadable: {}", e);
        return Err(DataError::RequiredFilesNotFound);
    }

    if let Err(e) = LazyFrame::scan_parquet(&place_path, Default::default()) {
        warn!("Place file corrupted or unreadable: {}", e);
        return Err(DataError::RequiredFilesNotFound);
    }

    Ok((admin_path, place_path))
}

/// Remove existing data files to force regeneration
fn clean_data_files(data_source: &DataSource) -> Result<()> {
    let admin_path = data_source.admin_parquet();
    let place_path = data_source.place_parquet();

    if admin_path.exists() {
        std::fs::remove_file(&admin_path)?;
        info!("Removed corrupted admin file: {:?}", admin_path);
    }

    if place_path.exists() {
        std::fs::remove_file(&place_path)?;
        info!("Removed corrupted place file: {:?}", place_path);
    }

    Ok(())
}

/// Ensure both admin and place data files exist and are valid, regenerating if necessary
fn ensure_data_files(data_source: &DataSource) -> Result<(PathBuf, PathBuf)> {
    // First try to validate existing files
    match validate_data_files(data_source) {
        Ok(paths) => {
            info!("Using existing processed data for {}", data_source);
            return Ok(paths);
        }
        Err(_) => {
            info!(
                "Data files missing or corrupted for {}, regenerating...",
                data_source
            );
            // Clean up any partial files
            clean_data_files(data_source)?;
        }
    }

    // Generate new data files
    #[cfg(feature = "download_data")]
    {
        let temp_files = crate::raw::fetch::download_data(data_source)?;

        info!("Generating processed data for {}", data_source);
        std::fs::create_dir_all(data_source.processed_dir())?;

        let (admin_df, place_df) = generate_processed_data(&temp_files)?;

        let admin_path = data_source.admin_parquet();
        let place_path = data_source.place_parquet();

        save_processed_data_to_parquet(admin_df, &admin_path)?;
        save_processed_data_to_parquet(place_df, &place_path)?;

        info!(
            "Processed data saved to {:?} and {:?}",
            admin_path, place_path
        );

        // Validate the newly created files
        validate_data_files(data_source)
    }
    #[cfg(not(feature = "download_data"))]
    {
        warn!("download_data feature not enabled, cannot regenerate data");
        Err(DataError::RequiredFilesNotFound)
    }
}

/// Get both admin and place data as LazyFrames
pub fn get_data(data_source: &DataSource) -> Result<(LazyFrame, LazyFrame)> {
    let (admin_path, place_path) = ensure_data_files(data_source)?;
    load_parquet_files(&admin_path, &place_path)
}

/// Get only admin search data as LazyFrame
///
/// This function ensures data consistency by validating that both admin and place files exist.
/// If either file is missing or corrupted, both will be regenerated.
pub fn get_admin_data(data_source: &DataSource) -> Result<LazyFrame> {
    let (admin_path, _place_path) = ensure_data_files(data_source)?;
    load_single_parquet_file(&admin_path)
}

/// Get only place search data as LazyFrame
///
/// This function ensures data consistency by validating that both admin and place files exist.
/// If either file is missing or corrupted, both will be regenerated.
pub fn get_place_data(data_source: &DataSource) -> Result<LazyFrame> {
    let (_admin_path, place_path) = ensure_data_files(data_source)?;
    load_single_parquet_file(&place_path)
}

/// Check if processed data exists for the given data source without loading it
pub fn data_exists(data_source: &DataSource) -> bool {
    validate_data_files(data_source).is_ok()
}

/// Force regeneration of processed data for the given data source
///
/// This will delete existing files and download/process fresh data.
pub fn regenerate_data(data_source: &DataSource) -> Result<(LazyFrame, LazyFrame)> {
    info!("Force regenerating data for {}", data_source);

    // Clean existing files
    clean_data_files(data_source)?;

    // Generate fresh data
    get_data(data_source)
}

#[cfg(test)]
pub(crate) mod tests_utils {
    use std::{io::Write, sync::atomic::AtomicUsize};

    use num_traits::NumCast;
    use polars::prelude::*;
    use tempfile::NamedTempFile;
    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    pub fn assert_has_columns(df: &DataFrame, expected_columns: &[&str]) {
        let actual_columns: Vec<_> = df.get_column_names().iter().map(|s| s.as_str()).collect();
        for expected_col in expected_columns {
            assert!(
                actual_columns.contains(expected_col),
                "Missing column: {expected_col}. Available columns: {actual_columns:?}"
            );
        }
    }

    pub fn assert_column_type(df: &DataFrame, column: &str, expected_type: &DataType) {
        let actual_type = df
            .column(column)
            .unwrap_or_else(|_| panic!("Column '{column}' not found"))
            .dtype();
        assert_eq!(
            actual_type, expected_type,
            "Column '{column}' has wrong type. Expected: {expected_type:?}, Got: {actual_type:?}"
        );
    }

    pub fn assert_no_nulls_in_column(df: &DataFrame, column: &str) {
        let null_count = df
            .column(column)
            .unwrap_or_else(|_| panic!("Column '{column}' not found"))
            .null_count();
        assert_eq!(
            null_count, 0,
            "Column '{column}' contains {null_count} null values"
        );
    }

    pub fn assert_column_range<T>(df: &DataFrame, column: &str, min_val: T, max_val: T)
    where
        T: std::fmt::Debug + NumCast + std::cmp::PartialOrd + Clone + 'static,
    {
        let series = df
            .column(column)
            .unwrap_or_else(|_| panic!("Column '{column}' not found"));

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
                "Column '{column}' min value {min_actual:?} is below expected minimum {min_val:?}"
            );
            assert!(
                max_actual <= max_val,
                "Column '{column}' max value {max_actual:?} is above expected maximum {max_val:?}"
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
            writeln!(file, "# Header line {i}").unwrap();
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
