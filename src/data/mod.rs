use std::path::Path;

use anyhow::Result;
use polars::prelude::*;

mod admin_search;
pub use admin_search::get_admin_search_data;

pub fn all_countries(path: &Path) -> Result<LazyFrame> {
    let schema = [
        ("geonameId".into(), DataType::UInt32),
        ("name".into(), DataType::String),
        ("asciiname".into(), DataType::String),
        ("alternatenames".into(), DataType::String),
        ("latitude".into(), DataType::Float32),
        ("longitude".into(), DataType::Float32),
        ("feature_class".into(), DataType::String),
        ("feature_code".into(), DataType::String),
        ("admin0_code".into(), DataType::String),
        ("cc2".into(), DataType::String),
        ("admin1_code".into(), DataType::String),
        ("admin2_code".into(), DataType::String),
        ("admin3_code".into(), DataType::String),
        ("admin4_code".into(), DataType::String),
        ("population".into(), DataType::Int64),
        ("elevation".into(), DataType::Int32),
        ("dem".into(), DataType::Int32),
        ("timezone".into(), DataType::String),
        ("modification_date".into(), DataType::Date),
    ];

    let col_names = [
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
    ];

    Ok(LazyCsvReader::new(path)
        .with_separator(b'\t')
        .with_has_header(false)
        .with_schema(Some(Schema::from_iter(schema).into()))
        .finish()?
        .with_column(
            dtype_col(&DataType::String)
                .str()
                .strip_chars(lit("\"':"))
                .str()
                .strip_chars(lit("")),
        )
        .sort(
            ["modification_date"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_nulls_last(true),
        )
        .unique_stable(
            Some(col_names.into_iter().map(From::from).collect()),
            UniqueKeepStrategy::First,
        )
        .filter(
            all_horizontal(
                col_names
                    .into_iter()
                    .map(|name| col(name).is_null())
                    .collect::<Vec<_>>(),
            )?
            .not(),
        ))
}

pub fn country_info(path: &Path) -> Result<LazyFrame> {
    let schema = [
        ("ISO".into(), DataType::String),
        ("ISO3".into(), DataType::String),
        ("ISO_Numeric".into(), DataType::Int32),
        ("fips".into(), DataType::String),
        ("Country".into(), DataType::String),
        ("Capital".into(), DataType::String),
        ("Area".into(), DataType::Float32),
        ("Population".into(), DataType::Int32),
        (
            "Continent".into(),
            DataType::Categorical(None, CategoricalOrdering::Lexical),
        ),
        ("tld".into(), DataType::String),
        ("CurrencyCode".into(), DataType::String),
        ("CurrencyName".into(), DataType::String),
        ("Phone".into(), DataType::String),
        ("Postal_Code_Format".into(), DataType::String),
        ("Postal_Code_Regex".into(), DataType::String),
        ("Languages".into(), DataType::String),
        ("geonameId".into(), DataType::UInt32),
        ("neighbours".into(), DataType::String),
        ("EquivalentFipsCode".into(), DataType::String),
    ];

    Ok(LazyCsvReader::new(path)
        .with_separator(b'\t')
        .with_has_header(false)
        .with_schema(Some(Schema::from_iter(schema).into()))
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
