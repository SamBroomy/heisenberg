use crate::error::Result;
use polars::prelude::*;
use std::path::Path;

// // Embedded data files (generated at build time)
// const EMBEDDED_ADMIN: &[u8] = include_bytes!("admin_search.parquet");
// const EMBEDDED_PLACES: &[u8] = include_bytes!("place_search.parquet");
// const EMBEDDED_METADATA: &[u8] = include_bytes!("metadata.json");

// /// Embedded dataset container
// #[derive(Clone)]
// pub struct EmbeddedData {
//     pub admin_search: LazyFrame,
//     pub place_search: LazyFrame,
//     pub metadata: EmbeddedMetadata,
// }

// #[derive(Debug, Clone, serde::Deserialize)]
// pub struct EmbeddedMetadata {
//     pub version: String,
//     pub source: String,
//     pub generated_at: String,
//     pub description: String,
//     pub admin_rows: usize,
//     pub place_rows: usize,
// }

// /// Load embedded dataset that ships with the library
// pub fn load_embedded_data() -> Result<EmbeddedData, heisenberg_data_processing::DataError> {
//     tracing::info!("Loading embedded dataset from built-in data");

//     // Load metadata
//     let metadata_str = std::str::from_utf8(EMBEDDED_METADATA).map_err(|e| {
//         heisenberg_data_processing::DataError::Io(std::io::Error::new(
//             std::io::ErrorKind::InvalidData,
//             e,
//         ))
//     })?;
//     let metadata: EmbeddedMetadata = serde_json::from_str(metadata_str).map_err(|e| {
//         heisenberg_data_processing::DataError::Io(std::io::Error::new(
//             std::io::ErrorKind::InvalidData,
//             e,
//         ))
//     })?;

//     // For the actual implementation, we need to load from the embedded bytes
//     // This is tricky because Polars doesn't directly support reading from &[u8]
//     // We'll need to write to temp files or use a different approach

//     // For now, create a simple fallback using the test data generation
//     let (admin_lf, place_lf) = create_embedded_dataset_fallback()?;

//     tracing::info!(
//         "Loaded embedded dataset: {} admin rows, {} place rows",
//         metadata.admin_rows,
//         metadata.place_rows
//     );

//     Ok(EmbeddedData {
//         admin_search: admin_lf,
//         place_search: place_lf,
//         metadata,
//     })
// }

// fn create_embedded_dataset_fallback()
// -> Result<(LazyFrame, LazyFrame), heisenberg_data_processing::DataError> {
//     // Use the data processing subcrate to generate the embedded dataset
//     use heisenberg_data_processing::{
//         EmbeddedDataSource, TestDataConfig, generate_embedded_dataset,
//     };

//     tracing::info!("Generating embedded dataset fallback using test data");

//     let source = EmbeddedDataSource::TestData(TestDataConfig::sample());
//     let dataset = generate_embedded_dataset(source)?;

//     Ok((dataset.admin_data.lazy(), dataset.place_data.lazy()))
// }
/// Load embedded admin search data
pub(crate) fn load_admin_search_data() -> Result<LazyFrame> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/data/embedded/admin_search.parquet");

    Ok(LazyFrame::scan_parquet(&path, Default::default())?)
}

/// Load embedded place search data
pub(crate) fn load_place_search_data() -> Result<LazyFrame> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/data/embedded/place_search.parquet");

    Ok(LazyFrame::scan_parquet(&path, Default::default())?)
}

/// Check if embedded data files exist
pub fn embedded_data_exists() -> bool {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let admin_path = manifest_dir.join("src/data/embedded/admin_search.parquet");
    let place_path = manifest_dir.join("src/data/embedded/place_search.parquet");

    admin_path.exists() && place_path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_data_exists() {
        assert!(
            embedded_data_exists(),
            "Embedded data files should exist after build"
        );
    }

    #[test]
    fn test_load_admin_search_data() {
        let lf = load_admin_search_data().expect("Should load admin search data");
        let df = lf.collect().expect("Should collect admin data");
        assert!(df.height() > 0, "Admin data should not be empty");
    }

    #[test]
    fn test_load_place_search_data() {
        let lf = load_place_search_data().expect("Should load place search data");
        let df = lf.collect().expect("Should collect place data");
        assert!(df.height() > 0, "Place data should not be empty");
    }
}
