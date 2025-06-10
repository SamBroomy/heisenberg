pub mod embedded;

use anyhow::Context;
use error::Result;
use heisenberg_data_processing::{DataSource, get_admin_data};
use once_cell::sync::OnceCell;
use polars::prelude::*;
use tracing::warn;

use crate::data::embedded::METADATA;

/// Location search data with fallback loading strategies
#[derive(Clone)]
pub struct LocationSearchData {
    data_source: DataSource,
    admin_search_df: OnceCell<LazyFrame>,
    place_search_df: OnceCell<LazyFrame>,
}

impl LocationSearchData {
    /// Create new LocationSearchData with specified data source
    pub fn new(data_source: DataSource) -> Self {
        Self {
            data_source,
            admin_search_df: OnceCell::new(),
            place_search_df: OnceCell::new(),
        }
    }

    /// Create new LocationSearchData using default embedded data
    pub fn new_embedded() -> Self {
        Self::new(METADATA.source.clone())
    }

    /// Get admin search data as LazyFrame with fallback loading
    pub fn admin_search_df(&self) -> LazyFrame {
        // Use OnceCell to cache the result after first load
        self.admin_search_df
            .get_or_init(|| {
                self.load_admin_data()
                    .and_then(|lf| {
                        // Collect the LazyFrame into a DataFrame and then convert it back to LazyFrame
                        lf.collect().map(|df| df.lazy()).map_err(From::from)
                    })
                    .expect("Failed to load admin search data")
            })
            .clone()
    }

    /// Get place search data as LazyFrame with fallback loading
    pub fn place_search_df(&self) -> LazyFrame {
        self.place_search_df
            .get_or_init(|| {
                self.load_place_data()
                    .and_then(|lf| {
                        // Collect the LazyFrame into a DataFrame and then convert it back to LazyFrame
                        lf.collect().map(|df| df.lazy()).map_err(From::from)
                    })
                    .expect("Failed to load admin search data")
            })
            .clone()
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

impl Default for LocationSearchData {
    fn default() -> Self {
        Self::new_embedded()
    }
}

pub mod error {
    use heisenberg_data_processing::error::DataError;
    use polars::error::PolarsError;
    use thiserror::Error;

    /// Custom error type for data loading operations
    #[derive(Error, Debug)]
    pub enum HeisenbergDataError {
        #[error("Data loading error: {0}")]
        DataError(#[from] DataError),
        #[error("Polars error: {0}")]
        Polars(#[from] PolarsError),
        #[error("Data source error: {0}")]
        DataSourceError(String),
    }

    pub type Result<T> = std::result::Result<T, HeisenbergDataError>;
}
