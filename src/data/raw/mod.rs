mod fetch;

mod all_countries;
mod country_info;
mod feature_codes;

use all_countries::get_all_countries;
use country_info::get_country_info;
use feature_codes::get_feature_codes;

use anyhow::Result;
use polars::prelude::LazyFrame;

//// Get all raw data as LazyFrames
/// returns a tuple of LazyFrames: (all_countries_df, country_info_df, feature_codes_df)
pub fn get_raw_data() -> Result<(LazyFrame, LazyFrame, LazyFrame)> {
    let client = reqwest::blocking::Client::builder().build()?;
    let all_countries_df = get_all_countries(&client)?;
    let country_info_df = get_country_info(&client)?;
    let feature_codes_df = get_feature_codes(&client)?;

    Ok((all_countries_df, country_info_df, feature_codes_df))
}
