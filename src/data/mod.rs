mod processed;
mod raw;

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
        #[error("HTTP error: {0}")]
        Http(#[from] reqwest::Error),
        #[error("Join error: {0}")]
        JoinError(#[from] tokio::task::JoinError),
        #[error("Zip error: {0}")]
        ZipError(#[from] zip::result::ZipError),
    }

    pub type Result<T> = std::result::Result<T, DataError>;
}
