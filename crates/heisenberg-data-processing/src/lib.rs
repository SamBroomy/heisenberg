use crate::processed::{generate_processed_data, save_processed_data_to_parquet};
pub use error::{DataError, Result};
use polars::prelude::LazyFrame;
pub use raw::DataSource;
use std::{
    path::{Path, PathBuf},
    sync::LazyLock,
};
pub use test_data::{TestDataConfig, create_test_data};
use tracing::{info, warn};

pub mod embedded;
pub mod error;
pub mod processed;
pub mod raw;
pub mod test_data;

pub const DATA_DIR_DEFAULT: &str = "./heisenberg_data";

pub static DATA_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    std::env::var("DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DATA_DIR_DEFAULT))
});
pub static PROCESSED_DIR: LazyLock<PathBuf> = LazyLock::new(|| DATA_DIR.join("processed"));

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
    let admin_path = PROCESSED_DIR.join(data_source.admin_parquet());
    let place_path = PROCESSED_DIR.join(data_source.place_parquet());

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
    let admin_path = PROCESSED_DIR.join(data_source.admin_parquet());
    let place_path = PROCESSED_DIR.join(data_source.place_parquet());

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
        std::fs::create_dir_all(PROCESSED_DIR.as_path())?;

        let (admin_df, place_df) = generate_processed_data(&temp_files)?;

        let admin_path = PROCESSED_DIR.join(data_source.admin_parquet());
        let place_path = PROCESSED_DIR.join(data_source.place_parquet());

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
