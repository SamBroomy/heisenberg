//! Basic location search functionality
//!
//! This example demonstrates the fundamental search operations:
//! - Creating a searcher instance
//! - Simple location searches
//! - Working with search results

use heisenberg::{LocationSearcher, SearchConfigBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create searcher instance (downloads/processes data on first run)
    let searcher = LocationSearcher::new(false)?;

    // Simple single-term search
    println!("Searching for 'London':");
    let results = searcher.search(&["London"])?;
    print_search_results(&results, 3);

    // Multi-term search for better precision
    println!("\nSearching for ['Paris', 'France']:");
    let results = searcher.search(&["Paris", "France"])?;
    print_search_results(&results, 3);

    // Search with configuration
    println!("\nFast search for 'New York' (limited results):");
    let config = SearchConfigBuilder::fast().limit(5).build();
    let results = searcher.search_with_config(&["New York"], &config)?;
    print_search_results(&results, 3);

    Ok(())
}

fn print_search_results(results: &[heisenberg::SearchResult], limit: usize) {
    for (i, result) in results.iter().take(limit).enumerate() {
        let result_type = match result {
            heisenberg::SearchResult::Admin(_) => "Admin",
            heisenberg::SearchResult::Place(_) => "Place",
        };

        println!(
            "  {}. {} ({}) - Score: {:.3}, Feature: {}",
            i + 1,
            result.name().unwrap_or("Unknown"),
            result_type,
            result.score().unwrap_or(0.0),
            result.feature_code().unwrap_or("Unknown")
        );
    }

    if results.len() > limit {
        println!("  ... and {} more results", results.len() - limit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn setup_test_env() {
        unsafe {
            env::set_var("USE_TEST_DATA", "true");
        }
    }

    #[test]
    fn test_basic_search_example() {
        setup_test_env();
        assert!(main().is_ok(), "Basic search example should run successfully");
    }
}
