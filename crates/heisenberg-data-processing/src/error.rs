use polars::prelude::PolarsError;
use thiserror::Error;
pub type Result<T> = std::result::Result<T, DataError>;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),
    #[cfg(feature = "download_data")]
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[cfg(feature = "download_data")]
    #[error("Join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
    #[cfg(feature = "download_data")]
    #[error("Zip error: {0}")]
    ZipError(#[from] zip::result::ZipError),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("No data directory provided and download_data feature is disabled")]
    NoDataDirProvided,
    #[error("Required data files not found in the provided directory")]
    RequiredFilesNotFound,
    #[error("Metadata file not found in the provided directory")]
    MetadataFileNotFound,
    #[error("Embedded Admin data not found in the provided directory")]
    EmbeddedAdminDataNotFound,
    #[error("Embedded data not found in the provided directory")]
    EmbeddedDataNotFound,
}
