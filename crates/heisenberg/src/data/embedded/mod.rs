/// Generated embedded data module
use std::sync::OnceLock;

use bytes::Bytes;
use heisenberg_data_processing::{embedded::EmbeddedMetadata, embedded_file_paths};
use polars::prelude::*;

use super::error::{HeisenbergDataError, Result};

static RAW_METADATA: &[u8] = include_bytes!(embedded_file_paths!(metadata));
static ADMIN_SEARCH_DATA: &[u8] = include_bytes!(embedded_file_paths!(admin));
static PLACE_SEARCH_DATA: &[u8] = include_bytes!(embedded_file_paths!(place));

pub(crate) static METADATA: std::sync::LazyLock<EmbeddedMetadata> =
    std::sync::LazyLock::new(|| {
        EmbeddedMetadata::from_bytes(RAW_METADATA).expect("Failed to parse embedded metadata")
    });

// Use OnceLock for fallible LazyFrame loading
static ADMIN_SEARCH_CELL: OnceLock<LazyFrame> = OnceLock::new();
static PLACE_SEARCH_CELL: OnceLock<LazyFrame> = OnceLock::new();

/// Get admin search data as LazyFrame (cached after first load)
pub(crate) fn load_embedded_admin_search_data() -> Result<LazyFrame> {
    ADMIN_SEARCH_CELL
        .get_or_try_init(|| {
            if ADMIN_SEARCH_DATA.is_empty() {
                return Err(HeisenbergDataError::DataSourceError(
                    "Embedded admin search data is empty".to_string(),
                ));
            }
            load_embedded_data(ADMIN_SEARCH_DATA).map_err(|e| {
                tracing::error!("Failed to load embedded admin search data: {}", e);
                e
            })
        })
        .cloned()
}

pub(crate) fn load_embedded_place_search_data() -> Result<LazyFrame> {
    PLACE_SEARCH_CELL
        .get_or_try_init(|| {
            if PLACE_SEARCH_DATA.is_empty() {
                return Err(HeisenbergDataError::DataSourceError(
                    "Embedded place search data is empty".to_string(),
                ));
            }
            load_embedded_data(PLACE_SEARCH_DATA).map_err(|e| {
                tracing::error!("Failed to load embedded place search data: {}", e);
                e
            })
        })
        .cloned()
}

fn load_embedded_data(data: &'static [u8]) -> Result<LazyFrame> {
    let bytes = Bytes::from_static(data);
    let source = ScanSource::Buffer(bytes.into()).into_sources();
    LazyFrame::scan_parquet_sources(source, Default::default()).map_err(From::from)
}
