pub mod embedded;

use once_cell::sync::OnceCell;
use polars::prelude::*;
use crate::error::Result;

// Public for potential future use
pub use embedded::embedded_data_exists;

/// Location search data loaded from embedded parquet files
#[derive(Clone)]
pub struct LocationSearchData {
    admin_search_df: OnceCell<LazyFrame>,
    place_search_df: OnceCell<LazyFrame>,
}

impl LocationSearchData {
    /// Create new LocationSearchData using embedded parquet files
    pub fn new() -> Result<Self> {
        Ok(Self {
            admin_search_df: OnceCell::new(),
            place_search_df: OnceCell::new(),
        })
    }

    /// Get admin search data as LazyFrame
    pub fn admin_search_df(&self) -> Result<&LazyFrame> {
        self.admin_search_df.get_or_try_init(|| {
            embedded::load_admin_search_data()
        })
    }

    /// Get place search data as LazyFrame
    pub fn place_search_df(&self) -> Result<&LazyFrame> {
        self.place_search_df.get_or_try_init(|| {
            embedded::load_place_search_data()
        })
    }
}