use polars::prelude::LazyFrame;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;
use tracing::{info, instrument, warn};
#[cfg(feature = "download_data")]
mod fetch;

mod all_countries;
mod country_info;
mod feature_codes;

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
