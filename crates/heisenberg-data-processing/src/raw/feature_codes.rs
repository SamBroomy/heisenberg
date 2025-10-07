use std::path::Path;

use polars::prelude::*;

use super::Result;

const FEATURE_CODES_SCHEMA: [(PlSmallStr, DataType); 3] = [
    (PlSmallStr::from_static("code"), DataType::String),
    (PlSmallStr::from_static("name"), DataType::String),
    (PlSmallStr::from_static("description"), DataType::String),
];

pub fn get_feature_codes_df(path: impl Into<Arc<Path>>) -> Result<LazyFrame> {
    Ok(LazyCsvReader::new(PlPath::Local(path.into()))
        .with_separator(b'\t')
        .with_has_header(false)
        .with_schema(Some(Schema::from_iter(FEATURE_CODES_SCHEMA).into()))
        .finish()?
        .with_column(col("code").str().split(lit(".")).alias("_tmp"))
        .with_columns(vec![
            col("_tmp").list().first().alias("feature_class"),
            col("_tmp").list().last().alias("feature_code"),
        ])
        .drop(cols(["_tmp", "code"]))
        .with_column(
            dtype_col(&DataType::String)
                .as_selector()
                .as_expr()
                .str()
                .strip_chars(lit("\"':"))
                .str()
                .strip_chars(lit("")),
        ))
}
