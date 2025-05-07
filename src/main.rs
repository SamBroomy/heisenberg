pub mod search;
use search::fts_search::{FTSIndex, FTSIndexes};
use search::{admin_search, place_search};

use anyhow::Result;

use polars::prelude::*;
use tracing::{info, Level};
use tracing_subscriber::fmt::format::FmtSpan;

fn main() -> Result<()> {
    tracing_subscriber::fmt::fmt()
        // filter spans/events with level TRACE or higher.
        .with_max_level(Level::INFO)
        .with_span_events(FmtSpan::CLOSE)
        // build but do not install the subscriber.
        .init();

    let admin_index = FTSIndex::new(FTSIndexes::AdminSearch)?;

    let lf = LazyFrame::scan_parquet(
        "./data/processed/geonames/admin_search.parquet",
        Default::default(),
    )?
    .collect()?
    .lazy();

    let t0 = std::time::Instant::now();

    let admins = admin_search(
        "The united states of america",
        &[0, 1],
        lf.clone(),
        &admin_index,
        None,
        Some(20),
        true,
    )?
    .unwrap()
    .collect()?;

    info!("Admin search results: {:?}", admins);

    let admins1 = admin_search(
        "California",
        &[1, 2],
        lf.clone(),
        &admin_index,
        Some(admins.lazy()),
        Some(20),
        true,
    )?
    .unwrap()
    .collect()?;
    info!("Admin1 search results: {:?}", admins1);

    let admins2 = admin_search(
        "Los Angeles County",
        &[2, 3],
        lf.clone(),
        &admin_index,
        Some(admins1.lazy()),
        Some(20),
        false,
    )?
    .unwrap()
    .collect()?;
    info!("Admin2 search results: {:?}", admins2);

    info!("Admin search took {} seconds", t0.elapsed().as_secs_f32());

    Ok(())
}
