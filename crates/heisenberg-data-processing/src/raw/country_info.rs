use std::path::Path;

use polars::prelude::*;

use super::Result;

const COUNTRY_INFO_SCHEMA: [(PlSmallStr, DataType); 19] = [
    (PlSmallStr::from_static("ISO"), DataType::String),
    (PlSmallStr::from_static("ISO3"), DataType::String),
    (PlSmallStr::from_static("ISO_Numeric"), DataType::Int32),
    (PlSmallStr::from_static("fips"), DataType::String),
    (PlSmallStr::from_static("Country"), DataType::String),
    (PlSmallStr::from_static("Capital"), DataType::String),
    (PlSmallStr::from_static("Area"), DataType::Float32),
    (PlSmallStr::from_static("Population"), DataType::Int32),
    (
        PlSmallStr::from_static("Continent"),
        DataType::Categorical(None, CategoricalOrdering::Lexical),
    ),
    (PlSmallStr::from_static("tld"), DataType::String),
    (PlSmallStr::from_static("CurrencyCode"), DataType::String),
    (PlSmallStr::from_static("CurrencyName"), DataType::String),
    (PlSmallStr::from_static("Phone"), DataType::String),
    (
        PlSmallStr::from_static("Postal_Code_Format"),
        DataType::String,
    ),
    (
        PlSmallStr::from_static("Postal_Code_Regex"),
        DataType::String,
    ),
    (PlSmallStr::from_static("Languages"), DataType::String),
    (PlSmallStr::from_static("geonameId"), DataType::UInt32),
    (PlSmallStr::from_static("neighbours"), DataType::String),
    (
        PlSmallStr::from_static("EquivalentFipsCode"),
        DataType::String,
    ),
];

pub fn get_country_info_df(path: impl AsRef<Path>) -> Result<LazyFrame> {
    Ok(LazyCsvReader::new(path)
        .with_separator(b'\t')
        .with_has_header(false)
        .with_schema(Some(Schema::from_iter(COUNTRY_INFO_SCHEMA).into()))
        .with_skip_lines(51)
        .finish()?
        .with_column(
            dtype_col(&DataType::String)
                .str()
                .strip_chars(lit("\"':"))
                .str()
                .strip_chars(lit("")),
        ))
}
