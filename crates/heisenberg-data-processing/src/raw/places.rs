use polars::prelude::*;
use tempfile::NamedTempFile;

const PLACES_FILE_SCHEMA: [(PlSmallStr, DataType); 19] = [
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

pub struct PlacesRawData {
    data: NamedTempFile,
}

impl PlacesRawData {
    pub fn new(file: NamedTempFile) -> Self {
        Self { data: file }
    }

    pub fn as_lazy_frame(&self) -> LazyFrame {
        LazyCsvReader::new(PlRefPath::new(
            self.data
                .path()
                .to_str()
                .expect("Failed to convert place data file path to string"),
        ))
        .with_separator(b'\t')
        .with_has_header(false)
        .with_schema(Some(Schema::from_iter(PLACES_FILE_SCHEMA).into()))
        .with_quote_char(None)
        .with_ignore_errors(true)
        .with_truncate_ragged_lines(true)
        .finish()
        .expect("Failed to read place data `allCountries.txt` or `cities*.txt`")
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
                .strip_chars(lit("\"'"))
                .str()
                .strip_chars(lit("")),
        )
        .with_column(col("alternatenames").str().split(lit(",")))
    }
}
