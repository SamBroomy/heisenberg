use std::{path::PathBuf, sync::LazyLock};

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use tracing::info;

use crate::{
    Result,
    processed::{generate_processed_data, save_processed_data_to_parquet},
    raw::DataSource,
    test_data::TestDataConfig,
};

// Sweaty workaround to let us use the same string as a static here and also in the include_bytes! macro
#[macro_export]
macro_rules! embedded_file_paths {
    (admin) => {
        "embedded_admin_search.parquet"
    };
    (place) => {
        "embedded_place_search.parquet"
    };
    (metadata) => {
        "embedded_data_metadata.json"
    };
}

// Also provide constants for runtime use
pub static ADMIN_DATA_PATH: &str = embedded_file_paths!(admin);
pub static PLACE_DATA_PATH: &str = embedded_file_paths!(place);
pub static METADATA_PATH: &str = embedded_file_paths!(metadata);

static EMBEDDED_DIR_DEFAULT: &str = "src/data/embedded";

pub static EMBEDDED_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    std::env::var("EMBEDDED_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(EMBEDDED_DIR_DEFAULT))
});

/// Generate embedded data from cities15000.zip and write as Rust source files
pub fn generate_embedded_dataset(data_source: DataSource) -> Result<()> {
    #[cfg(feature = "download_data")]
    {
        crate::raw::fetch::download_data(&data_source)
            .and_then(|temp_files| embed_data(&temp_files, data_source))
    }

    #[cfg(not(feature = "download_data"))]
    {
        tracing::warn!("download_data feature not enabled, falling back to test data");
        Self::generate_test_data_rust_code(output_dir)
    }
}

/// Generate embedded data from test data and write as Rust source files
pub fn generate_test_data_rust_code() -> Result<()> {
    tracing::info!("Generating embedded dataset from test data");

    crate::test_data::create_test_data(&TestDataConfig::sample())
        .and_then(|temp_files| embed_data(&temp_files, DataSource::TestData))
}

fn embed_data(
    temp_files: &(NamedTempFile, NamedTempFile, NamedTempFile),
    data_source: DataSource,
) -> Result<()> {
    info!("Generating processed data for {}", data_source);

    std::fs::create_dir_all(EMBEDDED_DIR.as_path())?;

    let (admin_df, place_df) = generate_processed_data(temp_files)?;

    let metadata = EmbeddedMetadata::from_dfs(&admin_df, &place_df, data_source)?;
    let metadata_path = EMBEDDED_DIR.join(METADATA_PATH);
    metadata.write_to_file(&metadata_path)?;

    let admin_path = EMBEDDED_DIR.join(ADMIN_DATA_PATH);
    let place_path = EMBEDDED_DIR.join(PLACE_DATA_PATH);

    save_processed_data_to_parquet(admin_df.clone(), &admin_path)?;
    save_processed_data_to_parquet(place_df.clone(), &place_path)?;
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DataFrameMetadata {
    pub rows: usize,
    pub size_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EmbeddedMetadata {
    pub version: String,
    pub source: DataSource,
    pub generated_at: String,
    pub admin_df: DataFrameMetadata,
    pub place_df: DataFrameMetadata,
}

impl EmbeddedMetadata {
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");

    pub fn new(
        source: DataSource,
        admin_rows: usize,
        place_rows: usize,
        admin_size_bytes: usize,
        place_size_bytes: usize,
    ) -> Self {
        let admin_df_metadata = DataFrameMetadata {
            rows: admin_rows,
            size_bytes: admin_size_bytes,
        };
        let place_df_metadata = DataFrameMetadata {
            rows: place_rows,
            size_bytes: place_size_bytes,
        };
        Self {
            version: Self::VERSION.to_string(),
            source,
            generated_at: chrono::Utc::now().to_rfc3339(),
            admin_df: admin_df_metadata,
            place_df: place_df_metadata,
        }
    }

    pub fn from_dfs(
        admin_df: &DataFrame,
        place_df: &DataFrame,
        data_source: DataSource,
    ) -> Result<Self> {
        Ok(Self::new(
            data_source,
            admin_df.height(),
            place_df.height(),
            admin_df.estimated_size(),
            place_df.estimated_size(),
        ))
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    pub fn write_to_file(&self, path: &std::path::Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(Into::into)
    }

    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(Into::into)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(Into::into)
    }
}
