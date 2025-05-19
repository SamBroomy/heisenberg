mod fetch;

mod all_countries;
mod country_info;
mod feature_codes;

use std::path::Path;

pub use super::error:: Result;
use polars::prelude::LazyFrame;
use tempfile::NamedTempFile;
use tracing::instrument;

/// Get all raw data from GeoNames
/// returns a tuple of NamedTempFiles: (all_countries_df, country_info_df, feature_codes_df)
#[instrument(name = "Download GeoNames data", skip_all, level = "info")]
pub fn get_raw_data() -> Result<(NamedTempFile, NamedTempFile, NamedTempFile)> {
    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        let client = reqwest::Client::new();

        let (all_countries_df, country_info_df, feature_codes_df) = tokio::try_join!(
            all_countries::download_all_countries(&client),
            country_info::download_country_info(&client),
            feature_codes::download_feature_codes(&client),
        )?;

        Ok((all_countries_df, country_info_df, feature_codes_df))
    })
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
