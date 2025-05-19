pub(crate) use error::SearchError;
pub mod location_search;
use error::Result;
pub use location_search::{
    AdminSearchParams, PlaceSearchParams, SearchScoreAdminParams, SearchScorePlaceParams,
    SmartFlexibleSearchConfig, admin_search_inner, bulk_location_search_inner,
    location_search_inner, place_search_inner,
};

mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum SearchError {
        #[error("DataFrame error: {0}")]
        DataFrame(#[from] polars::prelude::PolarsError),
        #[error("Index error: {0}")]
        IndexError(#[from] crate::index::IndexError),
        #[error(transparent)]
        Other(#[from] anyhow::Error),
    }
    pub type Result<T> = std::result::Result<T, SearchError>;
}
