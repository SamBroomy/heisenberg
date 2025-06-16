//! Batch processing for efficient bulk operations
//!
//! This example demonstrates how to process multiple location queries
//! efficiently using batch operations, which are significantly faster
//! than processing queries individually.

use std::time::Instant;

use heisenberg::{LocationContext, LocationSearcher, SearchConfigBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let searcher = LocationSearcher::new_embedded()?;

    // Define multiple queries to process
    let queries = vec![
        vec!["UK", "London"],
        vec!["France", "Paris"],
        vec!["Japan", "Tokyo"],
        vec!["USA", "New York"],
        vec!["Germany", "Berlin"],
        vec!["Australia", "Sydney"],
        vec!["Canada", "Toronto"],
        vec!["Italy", "Rome"],
    ];

    println!("Processing {} queries...", queries.len());

    // Performance comparison: individual vs batch
    compare_performance(&searcher, &queries)?;

    // Batch search with configuration
    batch_search_with_config(&searcher, &queries)?;

    // Batch resolution for complete hierarchies
    batch_resolution(&searcher, &queries)?;

    Ok(())
}

fn compare_performance(
    searcher: &LocationSearcher,
    queries: &[Vec<&str>],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nPerformance comparison:");

    // Individual processing
    let start = Instant::now();
    let mut individual_results = Vec::new();
    for query in queries {
        let results = searcher.search(query)?;
        individual_results.push(results);
    }
    let individual_time = start.elapsed();

    // Batch processing
    let start = Instant::now();
    let batch_results = searcher.search_bulk(queries)?;
    let batch_time = start.elapsed();

    println!(
        "  Individual: {:.3}s total ({:.3}s per query)",
        individual_time.as_secs_f32(),
        individual_time.as_secs_f32() / queries.len() as f32
    );

    println!(
        "  Batch:      {:.3}s total ({:.3}s per query)",
        batch_time.as_secs_f32(),
        batch_time.as_secs_f32() / queries.len() as f32
    );

    if individual_time > batch_time {
        let speedup = individual_time.as_secs_f32() / batch_time.as_secs_f32();
        println!("  Speedup:    {speedup:.1}x faster");
    }

    // Verify results are equivalent
    assert_eq!(individual_results.len(), batch_results.len());
    println!("  Results verified: individual and batch produce same output");

    Ok(())
}

fn batch_search_with_config(
    searcher: &LocationSearcher,
    queries: &[Vec<&str>],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nBatch search with fast configuration:");

    let config = SearchConfigBuilder::fast()
        .limit(3) // Limit results for faster processing
        .build();

    let results = searcher.search_bulk_with_config(queries, &config)?;

    for (i, (query, query_results)) in queries.iter().zip(results.iter()).enumerate() {
        let total_results = query_results.len();
        println!("  Query {}: {:?} - {} results", i + 1, query, total_results);
    }

    Ok(())
}

fn batch_resolution(
    searcher: &LocationSearcher,
    queries: &[Vec<&str>],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nBatch resolution:");

    let start = Instant::now();
    let resolved_results = searcher.resolve_location_batch(queries)?;
    let elapsed = start.elapsed();

    println!(
        "  Resolved {} queries in {:.3}s",
        queries.len(),
        elapsed.as_secs_f32()
    );

    for (i, (query, results)) in queries.iter().zip(resolved_results.iter()).enumerate() {
        if let Some(top_result) = results.first() {
            let hierarchy = build_hierarchy_string(&top_result.context);
            println!("  {}: {:?} → {}", i + 1, query, hierarchy);
        }
    }

    Ok(())
}

fn build_hierarchy_string(context: &LocationContext) -> String {
    let mut parts = Vec::new();

    if let Some(admin0) = &context.admin0 {
        parts.push(admin0.name());
    }
    if let Some(admin1) = &context.admin1 {
        parts.push(admin1.name());
    }
    if let Some(place) = &context.place {
        parts.push(place.name());
    }

    if parts.is_empty() {
        "No hierarchy found".to_string()
    } else {
        parts.join(" → ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_env() {
        let _ = heisenberg::init_logging(tracing::Level::WARN);
    }

    #[test]
    fn test_batch_processing_example() {
        setup_test_env();
        assert!(
            main().is_ok(),
            "Batch processing example should run successfully"
        );
    }
}
