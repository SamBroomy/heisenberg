pub mod search;
use search::fts_search::{AdminIndexDef, FTSIndex, PlacesIndexDef};
use search::{admin_search, get_admin_df, get_places_df, place_search};

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

    let t0 = std::time::Instant::now();
    // To create/load an admin search index:
    let admin_fts_index = FTSIndex::new(AdminIndexDef, /* overwrite= */ false)?;

    info!(
        "FTSIndex<AdminIndexDef> took {} seconds to load/create",
        t0.elapsed().as_secs_f32()
    );
    let t1 = std::time::Instant::now();
    let places_fts_index = FTSIndex::new(PlacesIndexDef, /* overwrite= */ false)?;
    info!(
        "FTSIndex<PlacesIndexDef> took {} seconds to load/create",
        t1.elapsed().as_secs_f32()
    );
    info!(
        "FTSIndex took {} seconds to load/create",
        t0.elapsed().as_secs_f32()
    );

    // let results = admin_fts_index.search("Kenya", 10, true)?;
    // let results = places_fts_index.search("Nairobi", 10, true)?;

    let t1 = std::time::Instant::now();
    let admin_lf = get_admin_df()?;
    info!(
        "Admin search took {} seconds to load",
        t1.elapsed().as_secs_f32()
    );
    let t2 = std::time::Instant::now();

    let place_lf = get_places_df()?;
    info!(
        "Place search took {} seconds to load",
        t2.elapsed().as_secs_f32()
    );
    info!(
        "Total load time took {} seconds",
        t0.elapsed().as_secs_f32()
    );

    let t0 = std::time::Instant::now();

    let admins = admin_search(
        "The united states of america",
        &[0, 1],
        admin_lf.clone(),
        &admin_fts_index,
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
        admin_lf.clone(),
        &admin_fts_index,
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
        admin_lf.clone(),
        &admin_fts_index,
        Some(admins1.lazy()),
        Some(20),
        false,
    )?
    .unwrap()
    .collect()?;
    info!("Admin2 search results: {:?}", admins2);

    let place = place_search(
        "Los Angeles",
        place_lf.clone(),
        &places_fts_index,
        Some(admins2.lazy()),
        Some(20),
        true,
        None,
        None,
        None,
    )?
    .unwrap()
    .collect()?;
    info!("Place search results: {:?}", place);

    info!("Admin search took {} seconds", t0.elapsed().as_secs_f32());

    Ok(())
}
