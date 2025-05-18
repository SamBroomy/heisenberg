use anyhow::Result;
use hbergv4::{
    FullAdminHierarchy, GeonameEntry, GeonameFullEntry, LocationSearchService,
    SmartFlexibleSearchConfig,
}; // Use your library name
use tracing::{info, info_span, warn, Level};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    hbergv4::init_logging(Level::INFO)?;

    let search_service = LocationSearchService::new(false)?;
    let smart_config = SmartFlexibleSearchConfig::default();

    info!("--- Running Single Structured Location Search ---");
    let single_query = vec!["UK", "London", "Camden", "British Museum"];
    let _single_span = info_span!("structured_search_single", query = ?single_query).entered();

    match search_service.find_structured_location::<GeonameFullEntry>(&single_query, &smart_config)
    {
        Ok(Some(hierarchy)) => {
            info!("Found structured location: {:#?}", hierarchy);
            if let Some(level0) = &hierarchy.level0 {
                info!("L0 (Country): {} ({})", level0.name, level0.geoname_id);
            }
            if let Some(level1) = &hierarchy.level1 {
                info!("L1: {} ({})", level1.name, level1.geoname_id);
            }
            if let Some(level2) = &hierarchy.level2 {
                info!("L2: {} ({})", level2.name, level2.geoname_id);
            }
            if let Some(level3) = &hierarchy.level3 {
                info!("L3: {} ({})", level3.name, level3.geoname_id);
            }
            if let Some(level4) = &hierarchy.level4 {
                info!("L4: {} ({})", level4.name, level4.geoname_id);
            }
        }
        Ok(None) => {
            warn!(
                "Could not find a structured location for: {:?}",
                single_query
            );
        }
        Err(e) => {
            warn!(
                "Error during structured location search for {:?}: {:?}",
                single_query, e
            );
        }
    }
    drop(_single_span);

    info!("--- Running Bulk Structured Location Search ---");
    let bulk_queries: Vec<Vec<&str>> = vec![
        vec!["FR", "Paris", "Louvre Museum"],
        vec!["USA", "California", "Los Angeles", "Hollywood Sign"],
        vec!["Germany", "Bavaria", "Munich", "Marienplatz"],
        vec!["Invalid", "Nonsense", "Query"], // Example of a query likely to fail
    ];
    let _bulk_span =
        info_span!("structured_search_bulk", num_queries = bulk_queries.len()).entered();

    // Convert Vec<Vec<&str>> to Vec<&[&str]> for the service method if it expects that,
    // or adjust the service method signature.
    // The current service method `bulk_find_structured_locations` is generic over Batch: AsRef<[Term]>.
    // So `Vec<Vec<&str>>` should work directly if `Vec<&str>` implements `AsRef<[&str]>`.
    // Let's assume it works or adjust if needed.
    // For clarity, if the inner type is `Vec<&str>`, it can be passed as `&[Vec<&str>]`.

    match search_service.bulk_find_structured_locations::<Vec<&str>, &str, GeonameEntry>(
        &bulk_queries,
        &smart_config,
    ) {
        Ok(results) => {
            for (i, opt_hierarchy) in results.iter().enumerate() {
                let _query_span =
                    info_span!("bulk_result_item", query_idx = i, query = ?bulk_queries[i])
                        .entered();
                match opt_hierarchy {
                    Some(hierarchy) => {
                        info!("Found structured location\n{:#?}", hierarchy);
                    }
                    None => {
                        warn!("No structured location found.");
                    }
                }
            }
        }
        Err(e) => {
            warn!("Error during bulk structured location search: {:?}", e);
        }
    }

    Ok(())
}
