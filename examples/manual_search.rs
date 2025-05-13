use anyhow::Result;
use hbergv4::{AdminSearchParams, LocationSearchService}; // Use your library name
use polars::prelude::*;
use tracing::{debug, info_span, Level};

fn main() -> Result<()> {
    hbergv4::init_logging(Level::DEBUG)?;

    let t_total_setup = std::time::Instant::now();
    let search_service = LocationSearchService::new(false)?; // false = don't overwrite indexes

    debug!(
        elapsed_seconds = t_total_setup.elapsed().as_secs_f32(),
        "LocationSearchService setup complete"
    );
    let _example_search_span = info_span!("manual_search_example").entered();

    let admin_params = AdminSearchParams::default();

    let admins0 = search_service
        .admin_search(
            "United Kingdom",
            &[0, 1], // Search for level 0 (country) and expect results that could be parents for level 1
            None::<DataFrame>,
            &admin_params,
        )?
        .unwrap_or_default();
    debug!(admins0_df = ?admins0, "Admin0 (Country) search results");

    let admins1 = search_service
        .admin_search(
            "London",
            &[1, 2], // Search for level 1 (state), using previous results as context
            Some(admins0),
            &admin_params,
        )?
        .unwrap_or_default();
    debug!(admins1_df = ?admins1, "Admin1 (State) search results");

    let admins2 = search_service
        .admin_search(
            "Westminster",
            &[2, 3], // Search for level 2 (county)
            Some(admins1),
            &admin_params,
        )?
        .unwrap_or_default();
    debug!(admins2_df = ?admins2, "Admin2 (County) search results");

    let admin3 = search_service
        .admin_search(
            "Parliament",
            &[3, 4], // Search for level 3 (city/admin3)
            Some(admins2.clone()),
            &admin_params,
        )?
        .unwrap_or_default();
    debug!(admin3_df = ?admin3, "Admin3 (City) search results");

    let places_input_df = concat(
        &[admins2.lazy(), admin3.lazy()],
        UnionArgs {
            diagonal: true,
            ..Default::default()
        },
    )?
    .collect()?; // Collect to DataFrame for place_search if it expects DataFrame

    debug!(places_input_df = ?places_input_df, "Places input DataFrame for place search");
    let place = search_service
        .place_search("Parliament", Some(places_input_df), &Default::default())?
        .unwrap_or_default();
    debug!(place_df = ?place, "Place search results for 'Parliament'");
    let t_total = t_total_setup.elapsed();
    debug!(
        total_elapsed_seconds = t_total.as_secs_f32(),
        "Total elapsed time for manual search example"
    );

    Ok(())
}
