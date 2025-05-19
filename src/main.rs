use anyhow::Result;
use heisenberg::backfill::ResolvedSearchResult;
use heisenberg::{GeonameEntry, Heisenberg, SmartFlexibleSearchConfig};
use polars::prelude::*;
use tracing::{debug, info, info_span, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("debug"))?
        .add_directive("tantivy=info".parse()?);
    tracing_subscriber::fmt::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    let t_total_setup = std::time::Instant::now();
    let search_service = Heisenberg::new(false)?;

    debug!(
        elapsed_seconds = t_total_setup.elapsed().as_secs_f32(),
        "LocationSearchService setup complete"
    );

    let _example_search_span = info_span!("manual_search_example").entered();

    // Example using the service
    let admins = search_service
        .admin_search(
            "The united states of america",
            &[0, 1],
            None::<DataFrame>,
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admins = ?admins, "Admin search results");

    let admins1 = search_service
        .admin_search(
            "California",
            &[1, 2],
            Some(admins), // Pass the DataFrame directly
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admins1 = ?admins1, "Admin1 search results");
    let admins2 = search_service
        .admin_search(
            "Los Angeles County",
            &[2, 3],
            Some(admins1),
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admins2 = ?admins2, "Admin2 search results");

    let admin3 = search_service
        .admin_search(
            "Beverly Hills",
            &[3, 4],
            Some(admins2.clone()), // Clone if admins2 is used again
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admin3 = ?admin3, "Admin3 search results");

    if !admins2.is_empty() && !admin3.is_empty() {
        let places_input_df = concat(
            &[admins2.lazy(), admin3.lazy()],
            UnionArgs {
                diagonal: true,
                ..Default::default()
            },
        )?
        .collect()?; // Collect to DataFrame for place_search if it expects DataFrame

        let place = search_service
            .place_search("Beverly Hills", Some(places_input_df), &Default::default())?
            .unwrap_or_default();
        debug!(place = ?place, "Place search results");
    }

    drop(_example_search_span);

    let examples = vec![
        vec!["US", "CA", "SF", "Golden Gate Bridge"],
        vec!["FL", "Lakeland"],
        vec![
            "The united states of america",
            "California",
            "Los Angeles County",
            "Beverly Hills",
        ],
        vec!["UK", "London", "Camden", "British Museum"],
        vec!["United Kingdom", "London", "Westminster", "Parliament"],
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
    let mut times = vec![];
    let smart_search_config = SmartFlexibleSearchConfig::default();

    for input in &examples {
        let t0 = std::time::Instant::now();
        let output = search_service.search_smart(input, &smart_search_config)?;
        let elapsed = t0.elapsed().as_secs_f32();

        warn!(
            ave_time = elapsed / input.len() as f32,
            "Smart flexible search seconds per example"
        );
        times.push(elapsed);

        for (i, df) in output.iter().enumerate() {
            info!(i=i,df=?df, "Smart flexible search results");
        }
    }
    let avg_time = times.iter().sum::<f32>() / times.len() as f32;
    warn!(avg_time = avg_time, "Average smart flexible search time");
    warn!(
        total_time = times.iter().sum::<f32>(),
        "Total smart flexible search time"
    );

    let t_bulk = std::time::Instant::now();
    let examples_refs: Vec<&[&str]> = examples.iter().map(|v| v.as_slice()).collect();
    let out_bulk: Vec<Vec<ResolvedSearchResult<GeonameEntry>>> =
        search_service.resolve_batch(&examples_refs, &smart_search_config, 20)?;

    warn!(t_bulk = ?t_bulk.elapsed(), "Bulk smart flexible search took");
    warn!(t_avg_per_example = ?t_bulk.elapsed().as_secs_f32() / examples.len() as f32, "Average time per example");

    for (i, vec) in out_bulk.iter().enumerate() {
        if vec.is_empty() {
            warn!(i, "Bulk smart flexible search results {}: empty", i);
            continue;
        }

        warn!(i, "Bulk smart flexible search results");
        if let Some((j, res)) = vec.iter().enumerate().next() {
            info!(i = i, j = j, "Bulk smart flexible search result");
            info!("{:#?}", res);
        }
    }

    Ok(())
}
