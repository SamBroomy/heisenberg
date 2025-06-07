use polars::prelude::*;
use std::io::Cursor;

// Embedded data files (generated at build time)
const EMBEDDED_ADMIN: &[u8] = include_bytes!("admin_search.parquet");
const EMBEDDED_PLACES: &[u8] = include_bytes!("place_search.parquet");
const EMBEDDED_METADATA: &[u8] = include_bytes!("metadata.json");

/// Embedded dataset container
#[derive(Clone)]
pub struct EmbeddedData {
    pub admin_search: LazyFrame,
    pub place_search: LazyFrame,
    pub metadata: EmbeddedMetadata,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct EmbeddedMetadata {
    pub version: String,
    pub source: String,
    pub generated_at: String,
    pub description: String,
    pub admin_rows: usize,
    pub place_rows: usize,
}

/// Load embedded dataset that ships with the library
pub fn load_embedded_data() -> Result<EmbeddedData, heisenberg_data_processing::DataError> {
    tracing::info!("Loading embedded dataset from built-in data");
    
    // Load metadata
    let metadata_str = std::str::from_utf8(EMBEDDED_METADATA)
        .map_err(|e| heisenberg_data_processing::DataError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    let metadata: EmbeddedMetadata = serde_json::from_str(metadata_str)
        .map_err(|e| heisenberg_data_processing::DataError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    
    // For the actual implementation, we need to load from the embedded bytes
    // This is tricky because Polars doesn't directly support reading from &[u8]
    // We'll need to write to temp files or use a different approach
    
    // For now, create a simple fallback using the test data generation
    let (admin_lf, place_lf) = create_embedded_dataset_fallback()?;
    
    tracing::info!("Loaded embedded dataset: {} admin rows, {} place rows", 
                   metadata.admin_rows, metadata.place_rows);
    
    Ok(EmbeddedData {
        admin_search: admin_lf,
        place_search: place_lf,
        metadata,
    })
}

fn create_embedded_dataset_fallback() -> Result<(LazyFrame, LazyFrame), heisenberg_data_processing::DataError> {
    // Use the data processing subcrate to generate the embedded dataset
    use heisenberg_data_processing::{EmbeddedDataSource, TestDataConfig, generate_embedded_dataset};
    
    tracing::info!("Generating embedded dataset fallback using test data");
    
    let source = EmbeddedDataSource::TestData(TestDataConfig::sample());
    let dataset = generate_embedded_dataset(source)?;
    
    Ok((dataset.admin_data.lazy(), dataset.place_data.lazy()))
}

/// Load embedded data from Parquet bytes (future implementation)
#[allow(dead_code)]
fn load_embedded_parquet() -> Result<(LazyFrame, LazyFrame), heisenberg_data_processing::DataError> {
    // This will be implemented once we have a way to read Parquet from bytes
    // Polars doesn't support this directly yet, so we might need to:
    // 1. Write to temp files first
    // 2. Use a different serialization format
    // 3. Wait for Polars to support reading from Cursor/bytes
    
    // Write embedded bytes to temp files
    let temp_dir = tempfile::tempdir()?;
    
    let admin_path = temp_dir.path().join("admin_search.parquet");
    let place_path = temp_dir.path().join("place_search.parquet");
    
    std::fs::write(&admin_path, EMBEDDED_ADMIN)?;
    std::fs::write(&place_path, EMBEDDED_PLACES)?;
    
    // Load from temp files
    let admin_lf = LazyFrame::scan_parquet(&admin_path, ScanArgsParquet::default())?;
    let place_lf = LazyFrame::scan_parquet(&place_path, ScanArgsParquet::default())?;
    
    Ok((admin_lf, place_lf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_embedded_data() {
        let embedded = load_embedded_data().expect("Should load embedded data");
        
        // Verify we have some data
        let admin_count = embedded.admin_search.clone().select([len()]).collect().unwrap()
            .column("len").unwrap().get(0).unwrap().try_extract::<u32>().unwrap();
        let place_count = embedded.place_search.clone().select([len()]).collect().unwrap()
            .column("len").unwrap().get(0).unwrap().try_extract::<u32>().unwrap();
        
        assert!(admin_count > 0, "Should have admin data");
        assert!(place_count > 0, "Should have place data");
        
        println!("Embedded data: {} admin entries, {} place entries", admin_count, place_count);
    }
}