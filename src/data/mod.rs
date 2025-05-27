mod processed;
mod raw;
pub(crate) use error::DataError;

pub const DATA_DIR_DEFAULT: &str = "./hberg_data";
static DATA_DIR: OnceCell<String> = OnceCell::new();

pub fn get_data_dir() -> &'static Path {
    DATA_DIR
        .get_or_init(|| std::env::var("DATA_DIR").unwrap_or_else(|_| DATA_DIR_DEFAULT.to_string()))
        .as_ref()
}

use std::path::Path;

use once_cell::sync::OnceCell;
pub use processed::get_data;

mod error {
    use polars::prelude::PolarsError;
    use thiserror::Error;

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
        #[error("No data directory provided and download_data feature is disabled")]
        NoDataDirProvided,
        #[error("Required data files not found in the provided directory")]
        RequiredFilesNotFound,
    }

    pub type Result<T> = std::result::Result<T, DataError>;
}
