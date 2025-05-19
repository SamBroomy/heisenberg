use anyhow::Result;
use polars::prelude::*;
use reqwest::Client;
use std::path::Path;
use tempfile::NamedTempFile;

use super::fetch::download_zip_and_extract_first_entry_to_temp_file;

const ALL_COUNTRIES_URL: &str = "https://download.geonames.org/export/dump/allCountries.zip";

pub async fn download_all_countries(client: &Client) -> Result<NamedTempFile> {
    download_zip_and_extract_first_entry_to_temp_file(client, ALL_COUNTRIES_URL).await
}

const ALL_COUNTRIES_SCHEMA: [(PlSmallStr, DataType); 19] = [
    (PlSmallStr::from_static("geonameId"), DataType::UInt32),
    (PlSmallStr::from_static("name"), DataType::String),
    (PlSmallStr::from_static("asciiname"), DataType::String),
    (PlSmallStr::from_static("alternatenames"), DataType::String),
    (PlSmallStr::from_static("latitude"), DataType::Float32),
    (PlSmallStr::from_static("longitude"), DataType::Float32),
    (PlSmallStr::from_static("feature_class"), DataType::String),
    (PlSmallStr::from_static("feature_code"), DataType::String),
    (PlSmallStr::from_static("admin0_code"), DataType::String),
    (PlSmallStr::from_static("cc2"), DataType::String),
    (PlSmallStr::from_static("admin1_code"), DataType::String),
    (PlSmallStr::from_static("admin2_code"), DataType::String),
    (PlSmallStr::from_static("admin3_code"), DataType::String),
    (PlSmallStr::from_static("admin4_code"), DataType::String),
    (PlSmallStr::from_static("population"), DataType::Int64),
    (PlSmallStr::from_static("elevation"), DataType::Int32),
    (PlSmallStr::from_static("dem"), DataType::Int32),
    (PlSmallStr::from_static("timezone"), DataType::String),
    (PlSmallStr::from_static("modification_date"), DataType::Date),
];

pub fn get_all_countries_df(path: impl AsRef<Path>) -> Result<LazyFrame> {
    Ok(LazyCsvReader::new(path)
        .with_separator(b'\t')
        .with_has_header(false)
        .with_schema(Some(Schema::from_iter(ALL_COUNTRIES_SCHEMA).into()))
        .finish()?
        .sort(
            ["modification_date"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_nulls_last(true),
        )
        .unique_stable(
            Some(vec![
                PlSmallStr::from_static("name"),
                PlSmallStr::from_static("asciiname"),
                PlSmallStr::from_static("feature_class"),
                PlSmallStr::from_static("feature_code"),
                PlSmallStr::from_static("admin0_code"),
                PlSmallStr::from_static("admin1_code"),
                PlSmallStr::from_static("admin2_code"),
                PlSmallStr::from_static("admin3_code"),
                PlSmallStr::from_static("admin4_code"),
                PlSmallStr::from_static("timezone"),
            ]),
            UniqueKeepStrategy::First,
        )
        .with_column(
            dtype_col(&DataType::String)
                .str()
                .strip_chars(lit("\"':"))
                .str()
                .strip_chars(lit("")),
        )
        .with_column(col("alternatenames").str().split(lit(","))))
}
