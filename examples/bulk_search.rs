// examples/bulk_search.rs
use anyhow::Result;
use heisenberg::{Heisenberg, HeisenbergError, SearchConfig};
use tracing::Level;

fn main() -> Result<(), HeisenbergError> {
    // Initialize logging
    heisenberg::init_logging(Level::INFO)?;

    // Create search service
    let search_service = Heisenberg::new(false)?;

    // Create configuration with higher limits for bulk processing
    let config = SearchConfig {
        limit: 10,
        all_cols: true, // Include all columns in results
        ..Default::default()
    };

    // Define batch of queries to process
    let batch_queries = [
        vec!["US", "CA", "SF", "Golden Gate Bridge"],
        vec!["UK", "London", "Camden", "British Museum"],
        vec!["FR", "Provence-Alpes-Côte d'Azur", "Le Lavandou"],
        vec!["DE", "Berlin", "Brandenburg Gate"],
        vec!["JP", "Tokyo", "Shibuya Crossing"],
        vec!["IT", "Rome", "Colosseum"],
        vec!["US", "CA", "SF", "Golden Gate Bridge"],
        vec!["FL", "Lakeland"],
        vec![
            "The united states of america",
            "California",
            "Los Angeles County",
            "Beverly Hills",
        ],
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
        vec!["CA", "Toronto", "CN Tower"],
        vec!["AU", "Sydney", "Sydney Opera House"],
        vec!["BR", "Rio de Janeiro", "Christ the Redeemer"],
        vec!["IN", "Agra", "Taj Mahal"],
        vec!["ZA", "Cape Town", "Table Mountain"],
        vec!["Moscow", "Red Square"],
        vec!["RU", "Saint Petersburg", "Hermitage Museum"],
        vec!["UK", "London", "Camden", "British Museum"], // Duplicate queries are treated internally as one query so wont run the same query multiple times
        vec!["UK", "London", "Camden", "British Museum"],
        vec!["UK", "London", "Camden", "British Museum"],
        vec!["UK", "London", "Camden", "British Museum"],
    ];

    println!(
        "Starting bulk search for {} queries...",
        batch_queries.len()
    );

    // Convert to slice references for bulk processing
    let batch_refs: Vec<&[&str]> = batch_queries.iter().map(|v| v.as_slice()).collect();

    // Start timing
    let start_time = std::time::Instant::now();

    // Execute bulk search
    let bulk_results = search_service.search_bulk_with_config(&batch_refs, &config)?;

    // Calculate elapsed time
    let elapsed = start_time.elapsed();
    let avg_time_per_query = elapsed.as_secs_f32() / batch_queries.len() as f32;

    println!(
        "✅ Bulk search completed in {:.3} seconds",
        elapsed.as_secs_f32()
    );
    println!(
        "   Average time per query: {:.3} seconds",
        avg_time_per_query
    );
    println!("   Processed {} queries", bulk_results.len());

    // Print summary of results
    for (i, query_results) in bulk_results.iter().enumerate() {
        println!("\n[Query {}] {:?}", i + 1, batch_queries[i]);
        println!("   Found {} result DataFrames", query_results.len());

        if !query_results.is_empty() {
            // Count total rows across all DataFrames
            let total_rows = query_results.iter().map(|df| df.height()).sum::<usize>();
            println!("   Total rows: {}", total_rows);

            // Show top result if available
            for df in query_results.iter() {
                if !df.is_empty() {
                    let top_results = df.head(Some(10));
                    println!("   Top results: {:?}", top_results);
                } else {
                    println!("   Empty DataFrame");
                }
            }
        } else {
            println!("   No results found");
        }
    }

    Ok(())
}
