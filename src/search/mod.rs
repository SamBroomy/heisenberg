pub(crate) use error::SearchError;
mod admin_search;
mod common;
mod place_search;
mod smart_flexible_search;

pub use admin_search::{AdminFrame, AdminSearchParams, SearchScoreAdminParams, admin_search_inner};
pub use place_search::{PlaceFrame, PlaceSearchParams, SearchScorePlaceParams, place_search_inner};
pub use smart_flexible_search::{
    SearchConfig, SearchResult, bulk_location_search_inner, location_search_inner,
};

use error::Result;

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
