use thiserror::Error;

#[derive(Error, Debug)]
pub enum HeisenbergError {
    #[error("Backfill error: {0}")]
    BackfillError(#[from] crate::backfill::BackfillError),
    #[error("Search error: {0}")]
    SearchError(#[from] crate::search::SearchError),
    #[error("Index error: {0}")]
    IndexError(#[from] crate::index::IndexError),
    #[error("Data processing error: {0}")]
    DataProcessing(#[from] heisenberg_data_processing::DataError),
    #[error("DataFrame error: {0}")]
    DataFrame(#[from] polars::prelude::PolarsError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Init Logging error: {0}")]
    InitLoggingError(#[from] tracing_subscriber::filter::ParseError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, HeisenbergError>;
