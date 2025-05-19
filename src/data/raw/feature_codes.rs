use super::Result;
use polars::prelude::*;
use reqwest::Client;
use std::path::Path;
use tempfile::NamedTempFile;

use super::fetch::download_to_temp_file;

const FEATURE_CODES_URL: &str = "https://download.geonames.org/export/dump/featureCodes_en.txt";
pub async fn download_feature_codes(client: &Client) -> Result<NamedTempFile> {
    download_to_temp_file(client, FEATURE_CODES_URL).await
}

const FEATURE_CODES_SCHEMA: [(PlSmallStr, DataType); 3] = [
    (PlSmallStr::from_static("code"), DataType::String),
    (PlSmallStr::from_static("name"), DataType::String),
    (PlSmallStr::from_static("description"), DataType::String),
];

pub fn get_feature_codes_df(path: impl AsRef<Path>) -> Result<LazyFrame> {
    Ok(LazyCsvReader::new(path)
        .with_separator(b'\t')
        .with_has_header(false)
        .with_schema(Some(Schema::from_iter(FEATURE_CODES_SCHEMA).into()))
        .finish()?
        .with_column(col("code").str().split(lit(".")).alias("_tmp"))
        .with_columns(vec![
            col("_tmp").list().first().alias("feature_class"),
            col("_tmp").list().last().alias("feature_code"),
        ])
        .drop(["_tmp", "code"])
        .with_column(
            dtype_col(&DataType::String)
                .str()
                .strip_chars(lit("\"':"))
                .str()
                .strip_chars(lit("")),
        ))
}
