use std::{
    hash::Hash,
    path::{Path, PathBuf},
    sync::LazyLock,
};

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

#[cfg(any(test, feature = "test-data"))]
use crate::raw::test_data::create_test_data;
use crate::{
    DataSource, Result,
    processed::{generate_processed_data, save_processed_data_to_parquet},
    raw::fetch::TempData,
};

// File names for embedded data (used by build.rs and data generation)
pub static ADMIN_DATA_PATH: &str = "embedded_admin_search.parquet";
pub static PLACE_DATA_PATH: &str = "embedded_place_search.parquet";
pub static METADATA_PATH: &str = "embedded_data_metadata.json";

static EMBEDDED_DIR_DEFAULT: &str = "src/data/embedded";

pub static EMBEDDED_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    // First check OUT_DIR (for build script usage)
    std::env::var("OUT_DIR")
        .ok()
        .map(|out_dir| PathBuf::from(out_dir).join("embedded_data"))
        .or_else(|| std::env::var("EMBEDDED_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(EMBEDDED_DIR_DEFAULT))
});

/// Generate embedded data from cities15000.zip and write as Rust source files
pub fn generate_embedded_dataset(data_source: DataSource) -> Result<()> {
    generate_embedded_dataset_to_dir(data_source, &EMBEDDED_DIR)
}

/// Generate embedded dataset to a specific directory (for build scripts with custom `OUT_DIR`)
pub fn generate_embedded_dataset_to_dir(data_source: DataSource, output_dir: &Path) -> Result<()> {
    #[cfg(feature = "download-data")]
    {
        crate::raw::fetch::download_data(&data_source)
            .and_then(|temp_data| embed_data_to_dir(temp_data, data_source, output_dir))?;
        Ok(())
    }
    #[cfg(all(not(feature = "download-data"), any(test, feature = "test-data")))]
    {
        tracing::warn!("download_data feature not enabled, falling back to test data");
        generate_test_data_to_dir(output_dir)
    }

    #[cfg(all(not(feature = "download-data"), not(any(test, feature = "test-data"))))]
    {
        compile_error!(
            "Either 'download_data' or 'test_data' feature must be enabled to generate embedded datasets"
        )
    }
}

/// Generate embedded data from test data and write as Rust source files
#[cfg(any(test, feature = "test-data"))]
pub fn generate_test_data_rust_code() -> Result<()> {
    generate_test_data_to_dir(&EMBEDDED_DIR)
}

#[cfg(any(test, feature = "test-data"))]
fn generate_test_data_to_dir(output_dir: &Path) -> Result<()> {
    tracing::info!("Generating embedded dataset from test data");

    let test_temp_files = create_test_data();
    embed_data_to_dir(test_temp_files, DataSource::TestData, output_dir)
}

fn embed_data_to_dir(
    temp_data: TempData,
    data_source: DataSource,
    output_dir: &Path,
) -> Result<()> {
    info!(
        "Generating processed data for {} to {:?}",
        data_source, output_dir
    );

    std::fs::create_dir_all(output_dir)?;

    let (admin_df, place_df) = generate_processed_data(temp_data)?;

    let metadata = EmbeddedMetadata::from_dfs(&admin_df, &place_df, data_source)?;
    let metadata_path = output_dir.join(METADATA_PATH);
    metadata.write_to_file(&metadata_path)?;

    let admin_path = output_dir.join(ADMIN_DATA_PATH);
    let place_path = output_dir.join(PLACE_DATA_PATH);

    save_processed_data_to_parquet(admin_df.clone(), &admin_path)?;
    save_processed_data_to_parquet(place_df.clone(), &place_path)?;
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, Hash)]
pub struct DataFrameMetadata {
    pub rows: usize,
    pub size_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, Hash)]
pub struct EmbeddedMetadata {
    pub version: String,
    pub source: DataSource,
    pub generated_at: String,
    pub admin_df: DataFrameMetadata,
    pub place_df: DataFrameMetadata,
}

impl EmbeddedMetadata {
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");

    #[must_use]
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

    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(Into::into)
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(Into::into)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(Into::into)
    }
}
