pub mod embedded;

use heisenberg_data_processing::{get_admin_data, raw::DataSource};
use once_cell::sync::OnceCell;
use polars::prelude::*;
use tracing::warn;

use crate::{data::embedded::METADATA, error::Result};

/// Location search data with fallback loading strategies
#[derive(Clone)]
pub struct LocationSearchData {
    data_source: DataSource,
    admin_search_df: OnceCell<LazyFrame>,
    place_search_df: OnceCell<LazyFrame>,
}

impl LocationSearchData {
    /// Create new LocationSearchData with specified data source
    pub fn new(data_source: DataSource) -> Result<Self> {
        Ok(Self {
            data_source,
            admin_search_df: OnceCell::new(),
            place_search_df: OnceCell::new(),
        })
    }

    /// Create new LocationSearchData using default embedded data
    pub fn new_embedded() -> Result<Self> {
        Self::new(DataSource::default())
    }

    /// Get admin search data as LazyFrame with fallback loading
    pub fn admin_search_df(&self) -> Result<&LazyFrame> {
        // Use OnceCell to cache the result after first load
        self.admin_search_df.get_or_try_init(|| {
            self.load_admin_data()
                .and_then(|lf| lf.collect().map(|df| df.lazy()).map_err(Into::into))
        })
    }

    /// Get place search data as LazyFrame with fallback loading
    pub fn place_search_df(&self) -> Result<LazyFrame> {
        self.load_place_data()
            .and_then(|lf| lf.collect().map(|df| df.lazy()).map_err(Into::into))
    }

    fn load_admin_data(&self) -> Result<LazyFrame> {
        if METADATA.source == self.data_source {
            // If the data source matches the embedded metadata, we can load the embedded data directly
            return embedded::load_embedded_admin_search_data();
        }

        match get_admin_data(&self.data_source) {
            // If the data source is the same as the embedded one, we can load the embedded data directly.
            Ok(admin_lf) => Ok(admin_lf),
            Err(e) => {
                // If we can't get the admin data, we can try to load the embedded data
                warn!(
                    "Failed to load admin data from source: {}, falling back to embedded data. Error: {}",
                    self.data_source, e
                );
                embedded::load_embedded_admin_search_data()
            }
        }
    }

    fn load_place_data(&self) -> Result<LazyFrame> {
        if METADATA.source == self.data_source {
            // If the data source matches the embedded metadata, we can load the embedded data directly
            return embedded::load_embedded_place_search_data();
        }

        match get_admin_data(&self.data_source) {
            // If the data source is the same as the embedded one, we can load the embedded data directly.
            Ok(place_lf) => Ok(place_lf),
            Err(e) => {
                // If we can't get the place data, we can try to load the embedded data
                warn!(
                    "Failed to load place data from source: {}, falling back to embedded data. Error: {}",
                    self.data_source, e
                );
                embedded::load_embedded_place_search_data()
            }
        }
    }

    /// Get the current data source
    pub fn data_source(&self) -> &DataSource {
        &self.data_source
    }
}
