use polars::prelude::*;
use tempfile::NamedTempFile;

const ADMIN_DATA_SCHEMA: [(PlSmallStr, DataType); 4] = [
    (PlSmallStr::from_static("code"), DataType::String),
    (PlSmallStr::from_static("name"), DataType::String),
    (PlSmallStr::from_static("asciiname"), DataType::String),
    (PlSmallStr::from_static("geonameId"), DataType::UInt32),
];

pub struct Admin1CodeRawData {
    data: NamedTempFile,
}

impl Admin1CodeRawData {
    pub fn new(data: NamedTempFile) -> Self {
        Self { data }
    }

    pub fn as_lazy_frame(&self) -> LazyFrame {
        LazyCsvReader::new(PlPath::Local(self.data.path().into()))
            .with_separator(b'\t')
            .with_has_header(false)
            .with_schema(Some(Schema::from_iter(ADMIN_DATA_SCHEMA).into()))
            .with_quote_char(None)
            .finish()
            .expect("Failed to read `admin1Codes.txt`")
            .with_columns([
                dtype_col(&DataType::String)
                    .as_selector()
                    .as_expr()
                    .str()
                    .strip_chars(lit("\"'"))
                    .str()
                    .strip_chars(lit("")),
                // Split code into admin0_code.admin1_code
                col("code").str().split(lit(".")).alias("_code_parts"),
                lit("A").alias("feature_class"),
                lit("ADM1").alias("feature_code"),
                lit(1u8).cast(DataType::UInt8).alias("admin_level"),
            ])
            .with_columns([
                col("_code_parts").list().first().alias("admin0_code"),
                col("_code_parts").list().last().alias("admin1_code"),
            ])
            .drop(cols(["_code_parts", "code"]))
    }
}

pub struct Admin2CodeRawData {
    data: NamedTempFile,
}

impl Admin2CodeRawData {
    pub fn new(file: NamedTempFile) -> Self {
        Self { data: file }
    }

    pub fn as_lazy_frame(&self) -> LazyFrame {
        LazyCsvReader::new(PlPath::Local(self.data.path().into()))
            .with_separator(b'\t')
            .with_has_header(false)
            .with_schema(Some(Schema::from_iter(ADMIN_DATA_SCHEMA).into()))
            .with_quote_char(None)
            .finish()
            .expect("Failed to read admin2Codes.txt")
            .with_columns([
                dtype_col(&DataType::String)
                    .as_selector()
                    .as_expr()
                    .str()
                    .strip_chars(lit("\"'"))
                    .str()
                    .strip_chars(lit("")),
                col("code").str().split(lit(".")).alias("_code_parts"),
                lit("A").alias("feature_class"),
                lit("ADM2").alias("feature_code"),
                lit(2u8).cast(DataType::UInt8).alias("admin_level"),
            ])
            // Split code into admin0_code.admin1_code.admin2_code
            .with_columns([
                col("_code_parts")
                    .list()
                    .get(lit(0), false)
                    .alias("admin0_code"),
                col("_code_parts")
                    .list()
                    .get(lit(1), false)
                    .alias("admin1_code"),
                col("_code_parts")
                    .list()
                    .get(lit(2), false)
                    .alias("admin2_code"),
            ])
            .drop(cols(["_code_parts", "code"]))
    }
}
