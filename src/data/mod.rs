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
