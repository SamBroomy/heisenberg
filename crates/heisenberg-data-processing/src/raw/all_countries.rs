use std::path::Path;

use polars::prelude::*;

use super::Result;

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

pub fn get_all_countries_df(path: impl Into<Arc<Path>>) -> Result<LazyFrame> {
    Ok(LazyCsvReader::new(PlPath::Local(path.into()))
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
            Some(cols([
                "name",
                "asciiname",
                "feature_class",
                "feature_code",
                "admin0_code",
                "admin1_code",
                "admin2_code",
                "admin3_code",
                "admin4_code",
                "timezone",
            ])),
            UniqueKeepStrategy::First,
        )
        .with_column(
            dtype_col(&DataType::String)
                .as_selector()
                .as_expr()
                .str()
                .strip_chars(lit("\"':"))
                .str()
                .strip_chars(lit("")),
        )
        .with_column(col("alternatenames").str().split(lit(","))))
}
