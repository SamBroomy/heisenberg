//! Search functionality for location matching.
//!
//! This module contains the core search implementations that power location finding.
//! It includes administrative search, place search, and intelligent search orchestration.

pub use error::SearchError;
mod admin_search;
mod common;
mod place_search;
mod search_orchestration;

pub use admin_search::{AdminFrame, AdminSearchParams, SearchScoreAdminParams, admin_search_inner};
use error::Result;
pub use place_search::{PlaceFrame, PlaceSearchParams, SearchScorePlaceParams, place_search_inner};
pub use search_orchestration::{
    SearchConfig, SearchResult, bulk_location_search_inner, location_search_inner,
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
