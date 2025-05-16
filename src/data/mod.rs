use std::path::Path;

use anyhow::Result;
use once_cell::sync::OnceCell;
use polars::prelude::*;

mod create_admin_search;
mod create_place_search;
mod raw;

use tracing::{info, warn};

const DATA_DIR: &str = "./hberg_data";
const ADMIN_SEARCH_PARQUET: &str = "admin_search.parquet";
const PLACE_SEARCH_PARQUET: &str = "place_search.parquet";

static ADMIN_DF_CACHE: OnceCell<LazyFrame> = OnceCell::new();
static PLACE_DF_CACHE: OnceCell<LazyFrame> = OnceCell::new();

fn get_df(path: &Path) -> Result<LazyFrame> {
    info!(
        path = ?path,
        "Loading and collecting into memory for the first time..."
    );
    let t_load = std::time::Instant::now();
    let df = LazyFrame::scan_parquet(path, Default::default())?
        .collect()
        .map(|df| df.lazy())
        .map_err(anyhow::Error::from);
    info!(
        time_collected = ?t_load.elapsed(),
        "Collected into memory"
    );
    df
}

fn get_admin_df(admin_search_path: &Path) -> Result<&'static LazyFrame> {
    ADMIN_DF_CACHE.get_or_try_init(|| get_df(admin_search_path))
}

fn get_place_df(place_search_path: &Path) -> Result<&'static LazyFrame> {
    PLACE_DF_CACHE.get_or_try_init(|| get_df(place_search_path))
}

fn load_parquet_files(
    admin_search_path: &Path,
    place_search_path: &Path,
) -> Result<(LazyFrame, LazyFrame)> {
    let admin_search_df = get_admin_df(admin_search_path)?.clone();
    let place_search_df = get_place_df(place_search_path)?.clone();
    Ok((admin_search_df, place_search_df))
}

/// Load the data from the parquet files or create them if they don't exist
/// Returns a tuple of LazyFrames: (admin_search_df, place_search_df)
/// These LazyFrames are collected into memory for faster access
/// and are cached for future use.
/// Calling this function multiple times will not reload the data.
pub fn get_data() -> Result<(LazyFrame, LazyFrame)> {
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| DATA_DIR.to_string());
    let data_dir = Path::new(&data_dir);
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir)?;
    }
    let admin_search_path = data_dir.join(ADMIN_SEARCH_PARQUET);
    let place_search_path = data_dir.join(PLACE_SEARCH_PARQUET);

    match (admin_search_path.exists(), place_search_path.exists()) {
        (true, true) => {
            // Both files exist, load them
            return load_parquet_files(&admin_search_path, &place_search_path);
        }
        (false, true) => {
            warn!("Admin search data file not found. ");
        }
        (true, false) => {
            warn!("Place search data file not found. ");
        }
        (false, false) => {
            warn!("Both admin and place search data files not found. ");
        }
    }

    let (all_countries_lf, country_info_lf, feature_codes_lf) = raw::get_raw_data()?;

    let admin_search_lf =
        create_admin_search::get_admin_search_lf(all_countries_lf.clone(), country_info_lf)?;

    let place_search_lf = create_place_search::get_place_search_lf(
        all_countries_lf,
        feature_codes_lf,
        admin_search_lf.clone(),
    )?;

    let mut dfs = collect_all([admin_search_lf, place_search_lf])?;

    let place_search_df = dfs.pop().expect("Place search should be last");
    let admin_search_df = dfs.pop().expect("Admin search should be first");

    let save_exprs = |lf: DataFrame, path: &Path| -> Result<()> {
        Ok(lf
            .lazy()
            .drop_nulls(Some(vec!["geonameId".into()]))
            .sort(["geonameId"], SortMultipleOptions::default())
            .sink_parquet(&path, ParquetWriteOptions::default(), None)?)
    };

    save_exprs(admin_search_df, &admin_search_path)?;
    save_exprs(place_search_df, &place_search_path)?;

    load_parquet_files(&admin_search_path, &place_search_path)
}
