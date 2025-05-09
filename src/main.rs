pub mod search;
use search::fts_search::{AdminIndexDef, FTSIndex, PlacesIndexDef};
use search::{
    admin_search, backfill_hierarchy_from_codes, get_admin_df, get_places_df, place_search,
    smart_flexible_search, TargetLocationAdminCodes,
};

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
        &admin_fts_index,
        admin_lf.clone(),
        None::<DataFrame>,
        &Default::default(),
    )?
    .unwrap();
    info!("Admin search results: {:?}", admins);

    let admins1 = admin_search(
        "California",
        &[1, 2],
        &admin_fts_index,
        admin_lf.clone(),
        Some(admins),
        &Default::default(),
    )?
    .unwrap();
    info!("Admin1 search results: {:?}", admins1);

    let admins2 = admin_search(
        "Los Angeles County",
        &[2, 3],
        &admin_fts_index,
        admin_lf.clone(),
        Some(admins1),
        &Default::default(),
    )?
    .unwrap();
    info!("Admin2 search results: {:?}", admins2);

    let admin3 = admin_search(
        "Beverly Hills",
        &[3, 4],
        &admin_fts_index,
        admin_lf.clone(),
        Some(admins2.clone()),
        &Default::default(),
    )?
    .unwrap();
    info!("Admin3 search results: {:?}", admin3);

    let places_input = concat(
        &[admins2.lazy(), admin3.lazy()],
        UnionArgs {
            diagonal: true,
            ..Default::default()
        },
    )?;
    info!("Places input {:?}", places_input.clone().collect()?);

    let place = place_search(
        "Beverly Hills",
        &places_fts_index,
        place_lf.clone(),
        Some(places_input),
        &Default::default(),
    )?
    .unwrap();
    info!("Place search results: {:?}", place);

    info!("Admin search took {} seconds", t0.elapsed().as_secs_f32());

    let t0 = std::time::Instant::now();

    let examples = [
        vec![
            "The united states of america",
            "California",
            "Los Angeles County",
            "Beverly Hills",
        ],
        vec!["UK", "London", "Camden", "British Museum"],
        vec!["United Kingdom", "London", "Westminster", "Parlement"],
        vec!["FR", "Provence-Alpes-Côte d'Azur", "Le Lavandou"],
        vec!["England", "Dover", "Dover Ferry Terminal"],
        vec![
            "FR",
            "Provence-Alpes-Côte d'Azur",
            "Var",
            "Arrondissement de Toulon",
            "Le Lavandou",
        ],
    ];

    for input in examples {
        let output = smart_flexible_search(
            &input,
            &admin_fts_index,
            admin_lf.clone(),
            &places_fts_index,
            place_lf.clone(),
            &Default::default(),
        )?;
        info!(
            "Smart flexible search took {} seconds",
            t0.elapsed().as_secs_f32()
        );

        for (i, df) in output.iter().enumerate() {
            info!("Smart flexible search results {}: {:?}", i, df);
        }
        let chosen_df = output.last().unwrap();
        info!("Chosen DataFrame: {:?}", chosen_df);

        if !chosen_df.is_empty() {
            let target_codes = TargetLocationAdminCodes::from_dataframe_row(chosen_df, 0)?;
            info!("Target Location Admin Codes: {:#?}", target_codes);
            let enriched_hierarchy =
                backfill_hierarchy_from_codes(&target_codes, admin_lf.clone())?;
            // Now `enriched_hierarchy` contains the full path
            println!("Enriched Hierarchy: {:#?}", enriched_hierarchy);
        }
    }

    Ok(())
}
