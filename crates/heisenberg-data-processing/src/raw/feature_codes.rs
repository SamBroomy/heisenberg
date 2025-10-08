use polars::prelude::*;
use tempfile::NamedTempFile;

const FEATURE_CODES_SCHEMA: [(PlSmallStr, DataType); 3] = [
    (PlSmallStr::from_static("code"), DataType::String),
    (PlSmallStr::from_static("name"), DataType::String),
    (PlSmallStr::from_static("description"), DataType::String),
];
pub struct FeatureCodesRawData {
    data: NamedTempFile,
}

impl FeatureCodesRawData {
    pub fn new(file: NamedTempFile) -> Self {
        Self { data: file }
    }

    pub fn as_lazy_frame(&self) -> LazyFrame {
        LazyCsvReader::new(PlPath::Local(self.data.path().into()))
            .with_separator(b'\t')
            .with_has_header(false)
            .with_schema(Some(Schema::from_iter(FEATURE_CODES_SCHEMA).into()))
            .with_quote_char(None)
            .finish()
            .expect("Failed to read `feature_codes.txt`")
            .with_column(col("code").str().split(lit(".")).alias("_tmp"))
            .with_columns([
                dtype_col(&DataType::String)
                    .as_selector()
                    .as_expr()
                    .str()
                    .strip_chars(lit("\"'"))
                    .str()
                    .strip_chars(lit("")),
                col("_tmp").list().first().alias("feature_class"),
                col("_tmp").list().last().alias("feature_code"),
            ])
            .drop(cols(["_tmp", "code"]))
    }
}
