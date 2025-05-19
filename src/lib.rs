pub mod backfill;
pub mod data;
pub mod index;
pub mod search;
pub mod service;

pub use backfill::{GenericEntry, GeonameEntry, LocationEntry};
pub use error::HeisenbergError;
use once_cell::sync::OnceCell;
pub use search::{AdminSearchParams, PlaceSearchParams, SmartFlexibleSearchConfig};
pub use service::Heisenberg;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

static PLACES_DF_CACHE: OnceCell<()> = OnceCell::new();

pub fn init_logging(level: impl Into<LevelFilter>) -> Result<&'static (), HeisenbergError> {
    PLACES_DF_CACHE.get_or_try_init(|| {
        let filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(level.into().to_string()))?
            .add_directive("tantivy=warn".parse().unwrap())
            .add_directive("hyper_util=warn".parse().unwrap());

        tracing_subscriber::fmt::fmt()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::CLOSE)
            .init();
        Ok(())
    })
}

mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum HeisenbergError {
        #[error("Backfill error: {0}")]
        BackfillError(#[from] crate::backfill::BackfillError),
        #[error("Data error: {0}")]
        DataError(#[from] crate::data::DataError),
        #[error("Index error: {0}")]
        IndexError(#[from] crate::index::IndexError),
        #[error("Search error: {0}")]
        SearchError(#[from] crate::search::SearchError),
        #[error("DataFrame error: {0}")]
        DataFrame(#[from] polars::prelude::PolarsError),
        #[error(transparent)]
        Other(#[from] anyhow::Error),
        #[error("Init Logging error: {0}")]
        InitLoggingError(#[from] tracing_subscriber::filter::ParseError),
    }
}
