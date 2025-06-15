//! Data source and initialization patterns
//!
//! This example demonstrates the different ways to initialize the `LocationSearcher`
//! with various data sources and loading strategies.

use heisenberg::{DataSource, LocationSearcher};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Heisenberg LocationSearcher - Data Source Examples\n");

    // Method 1: Use embedded data (fastest startup, no downloads)
    println!("1. Using embedded data (recommended for most use cases):");
    let embedded_searcher = LocationSearcher::new_embedded()?;
    let results = embedded_searcher.search(&["London"])?;
    println!("   Found {} results for 'London'\n", results.len());

    // Method 2: Smart initialization with fallback (tries embedded first, then downloads)
    println!("2. Smart initialization with Cities15000 dataset:");
    let smart_searcher = LocationSearcher::initialize(DataSource::Cities15000)?;
    let results = smart_searcher.search(&["Paris"])?;
    println!("   Found {} results for 'Paris'\n", results.len());

    // Method 3: Force fresh indexes (rebuilds everything)
    println!("3. Force fresh indexes with Cities5000 dataset:");
    let fresh_searcher = LocationSearcher::new_with_fresh_indexes(DataSource::Cities5000)?;
    let results = fresh_searcher.search(&["Tokyo"])?;
    println!("   Found {} results for 'Tokyo'\n", results.len());

    // Method 4: Try to load existing (returns None if doesn't exist)
    println!("4. Try to load existing indexes:");
    match LocationSearcher::load_existing(DataSource::Cities1000)? {
        Some(existing_searcher) => {
            let results = existing_searcher.search(&["Berlin"])?;
            println!(
                "   Loaded existing, found {} results for 'Berlin'\n",
                results.len()
            );
        }
        None => {
            println!("   No existing indexes found for Cities1000\n");
        }
    }

    // Show available data sources
    println!("Available data sources:");
    let data_sources = [
        DataSource::Cities15000,
        DataSource::Cities5000,
        DataSource::Cities1000,
        DataSource::Cities500,
        DataSource::AllCountries,
        DataSource::TestData,
    ];

    for source in data_sources {
        println!(
            "  - {}: {}",
            source,
            match source {
                DataSource::Cities15000 => "Cities with population > 15,000 (recommended)",
                DataSource::Cities5000 => "Cities with population > 5,000",
                DataSource::Cities1000 => "Cities with population > 1,000",
                DataSource::Cities500 => "Cities with population > 500",
                DataSource::AllCountries => "All GeoNames data (very large)",
                DataSource::TestData => "Test data for development",
            }
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    fn setup_test_env() {
        let _ = heisenberg::init_logging(tracing::Level::WARN);
    }

    #[test]
    fn test_data_sources_example() {
        setup_test_env();
        assert!(
            main().is_ok(),
            "Data sources example should run successfully"
        );
    }
}
