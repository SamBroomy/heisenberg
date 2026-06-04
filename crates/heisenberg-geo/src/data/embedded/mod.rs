/// Generated embedded data module
use std::sync::{LazyLock, OnceLock};

use heisenberg_data_processing::embedded::EmbeddedMetadata;
use polars::{
    io::HiveOptions,
    prelude::{ScanSource, *},
};

use super::error::{HeisenbergDataError, Result};

static RAW_METADATA: &[u8] = include_bytes!(concat!(
    env!("EMBEDDED_DATA_DIR"),
    "/",
    env!("METADATA_PATH")
));
static ADMIN_SEARCH_DATA: &[u8] = include_bytes!(concat!(
    env!("EMBEDDED_DATA_DIR"),
    "/",
    env!("ADMIN_DATA_PATH")
));
static PLACE_SEARCH_DATA: &[u8] = include_bytes!(concat!(
    env!("EMBEDDED_DATA_DIR"),
    "/",
    env!("PLACE_DATA_PATH")
));

pub static METADATA: LazyLock<EmbeddedMetadata> = LazyLock::new(|| {
    EmbeddedMetadata::from_bytes(RAW_METADATA).expect("Failed to parse embedded metadata")
});

// Use OnceLock for fallible LazyFrame loading
static ADMIN_SEARCH_CELL: OnceLock<LazyFrame> = OnceLock::new();
static PLACE_SEARCH_CELL: OnceLock<LazyFrame> = OnceLock::new();

/// Get admin search data as `LazyFrame` (cached after first load)
pub fn load_embedded_admin_search_data() -> Result<LazyFrame> {
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

pub fn load_embedded_place_search_data() -> Result<LazyFrame> {
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
    // let bytes = Bytes::from_static(data);
    let buffer = polars_buffer::Buffer::from_static(data);
    let source = ScanSource::Buffer(buffer).into_sources();

    LazyFrame::scan_parquet_sources(
        source,
        ScanArgsParquet {
            hive_options: HiveOptions::new_disabled(),
            ..Default::default()
        },
    )
    .map_err(From::from)
}
