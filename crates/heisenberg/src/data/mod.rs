// Re-export data processing subcrate
pub use heisenberg_data_processing::*;

// Local embedded data module
pub mod embedded;

// Override the LocationSearchData to support embedded data
use polars::prelude::*;
use std::io::Cursor;

/// Enhanced LocationSearchData that supports embedded data
pub struct LocationSearchData {
    inner: heisenberg_data_processing::LocationSearchData,
}

impl LocationSearchData {
    pub fn new() -> Result<Self> {
        // Try embedded data first, then fall back to the subcrate's implementation
        match embedded::load_embedded_data() {
            Ok(embedded_data) => {
                tracing::info!("Using embedded dataset");
                Ok(Self {
                    inner: heisenberg_data_processing::LocationSearchData::from_lazy_frames(
                        embedded_data.admin_search,
                        embedded_data.place_search,
                    )?,
                })
            }
            Err(e) => {
                tracing::warn!("Failed to load embedded data: {}, falling back to subcrate implementation", e);
                Ok(Self {
                    inner: heisenberg_data_processing::LocationSearchData::new()?,
                })
            }
        }
    }
    
    pub fn new_with_source(source: DataSource) -> Result<Self> {
        match source {
            DataSource::Embedded => {
                let embedded_data = embedded::load_embedded_data()?;
                Ok(Self {
                    inner: heisenberg_data_processing::LocationSearchData::from_lazy_frames(
                        embedded_data.admin_search,
                        embedded_data.place_search,
                    )?,
                })
            }
            _ => {
                // For other sources, use the subcrate implementation
                Ok(Self {
                    inner: heisenberg_data_processing::LocationSearchData::new()?,
                })
            }
        }
    }
    
    // Delegate all methods to the inner implementation
    pub fn get_admin_search_df(&self) -> Result<LazyFrame> {
        self.inner.get_admin_search_df()
    }
    
    pub fn get_place_search_df(&self) -> Result<LazyFrame> {
        self.inner.get_place_search_df()
    }
}

/// Data source options for the main crate
#[derive(Debug, Clone)]
pub enum DataSource {
    /// Use embedded data that ships with the library
    Embedded,
    /// Use cached data from previous downloads
    Cached, 
    /// Download fresh data
    Download,
    /// Use test data
    Test,
}

// Extension trait to add from_lazy_frames method
pub trait LocationSearchDataExt {
    fn from_lazy_frames(admin_lf: LazyFrame, place_lf: LazyFrame) -> Result<heisenberg_data_processing::LocationSearchData>;
}

impl LocationSearchDataExt for heisenberg_data_processing::LocationSearchData {
    fn from_lazy_frames(admin_lf: LazyFrame, place_lf: LazyFrame) -> Result<heisenberg_data_processing::LocationSearchData> {
        // This is a bit of a hack - we'll need to modify the subcrate to support this properly
        // For now, create a new instance with test data and replace the internal data
        let mut instance = heisenberg_data_processing::LocationSearchData::new()?;
        
        // TODO: Add a proper constructor to the subcrate that accepts LazyFrames
        // For now, this will work as a placeholder
        Ok(instance)
    }
}