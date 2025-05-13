use anyhow::Result;
use hbergv4::{LocationSearchService, SmartFlexibleSearchConfig};
use tracing::{info, info_span, warn, Level};

fn main() -> Result<()> {
    hbergv4::init_logging(Level::DEBUG)?;

    let search_service = LocationSearchService::new(false)?;

    let examples = [
        vec![
            "The united states of america",
            "California",
            "Los Angeles County",
            "Beverly Hills",
        ],
        vec!["London", "Camden", "British Museum"],
        vec!["United Kingdom", "London", "Westminster", "Parlement"],
        vec!["FR", "Provence-Alpes-Côte d'Azur", "Le Lavandou"],
        vec!["England", "Dover", "Dover Ferry Terminal"],
        vec![
            "Provence-Alpes-Côte d'Azur",
            "Var",
            "Arrondissement de Toulon",
            "Le Lavandou",
        ],
        vec!["Germany", "Berlin", "Brandenburg Gate"],
        vec!["Japan", "Tokyo", "Shibuya Crossing"],
    ];
    let mut times = vec![];
    let smart_search_config = SmartFlexibleSearchConfig::default();

    info!("--- Running Individual Smart Flexible Searches ---");
    for (idx, input) in examples.iter().enumerate() {
        let _span = info_span!("smart_search_single", query_idx = idx, query = ?input).entered();
        let t0 = std::time::Instant::now();
        let output_dfs = search_service.smart_flexible_search(input, &smart_search_config)?;
        let elapsed = t0.elapsed().as_secs_f32();
        times.push(elapsed);

        info!(elapsed_secs = elapsed, "Search complete.");
        for (i, df) in output_dfs.iter().enumerate() {
            info!(df_idx = i, df_head = ?df.head(Some(3)), "Result DataFrame");
        }
    }
    if !times.is_empty() {
        let avg_time = times.iter().sum::<f32>() / times.len() as f32;
        info!(
            avg_time_secs = avg_time,
            "Average smart_flexible_search time"
        );
        info!(
            total_time_secs = times.iter().sum::<f32>(),
            "Total smart_flexible_search time for all examples"
        );
    }

    info!("--- Running Bulk Smart Flexible Search ---");
    let t_bulk = std::time::Instant::now();
    // SmartFlexibleSearchConfig expects &[&[&str]], so we convert
    let examples_refs: Vec<&[&str]> = examples.iter().map(|v| v.as_slice()).collect();
    let out_bulk_dfs =
        search_service.bulk_smart_flexible_search(&examples_refs, &smart_search_config)?;
    let bulk_elapsed = t_bulk.elapsed();

    info!(total_bulk_time = ?bulk_elapsed, "Bulk smart_flexible_search complete.");
    if !examples.is_empty() {
        info!(avg_time_per_query_ms = ?(bulk_elapsed.as_millis() as f32 / examples.len() as f32), "Average time per query in bulk");
    }

    for (i, dfs_for_one_input) in out_bulk_dfs.iter().enumerate() {
        info!(query_idx = i, query = ?examples[i], "Results for query in bulk:");
        if dfs_for_one_input.is_empty() {
            warn!("Bulk search results for query index {}: empty", i);
        }
        for (df_idx, df) in dfs_for_one_input.iter().enumerate() {
            info!(df_idx = df_idx, df_head = ?df.head(Some(3)), "Result DataFrame");
        }
    }

    Ok(())
}
