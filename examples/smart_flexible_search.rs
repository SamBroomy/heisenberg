// examples/smart_flexible_search.rs
use anyhow::Result;
use heisenberg::{Heisenberg, HeisenbergError, SmartFlexibleSearchConfig};
use tracing::Level;

fn main() -> Result<(), HeisenbergError> {
    // Initialize logging
    heisenberg::init_logging(Level::INFO)?;

    // Create search service
    let search_service = Heisenberg::new(false)?;

    // Create a default configuration
    let config = SmartFlexibleSearchConfig::default();

    // Define some example location queries
    let examples = [
        vec![
            // "United States", Is able to still work with missing context
            "CA", // Can use abbreviations
            "San Francisco",
            "Golden Gate Bridge",
        ],
        vec![
            "United States",
            // "CA", // Can use abbreviations
            "SF",
            "Golden Gate Bridge",
        ],
        vec!["UK", "London", "Camden", "British Museum"],
        vec!["France", "Provence-Alpes-Côte d'Azur", "Le Lavandou"],
        vec!["Germany", "Berlin", "Brandenburg Gate"],
    ];

    // Process each example query
    for (i, query) in examples.iter().enumerate() {
        println!("\n[Example {}] Searching for: {:?}", i + 1, query);

        // Start timing
        let start_time = std::time::Instant::now();

        // Perform smart flexible search
        let results = search_service.search_smart(query, &config)?;

        // Calculate elapsed time
        let elapsed = start_time.elapsed();

        println!(
            "✅ Search completed in {:.3} seconds",
            elapsed.as_secs_f32()
        );
        println!("   Found {} result DataFrames", results.len());

        // Print summary of each result DataFrame
        for (j, df) in results.iter().enumerate() {
            println!("   - Result DataFrame {}: {} rows", j, df.height());
            if !df.is_empty() {
                // Get the first row of the DataFrame to see what was found
                let df = df.head(Some(10));

                println!("     Top results: {:?}", df);

                // Get the first row
                let first_row = df.head(Some(1));

                // Try to extract some meaningful information to display
                if let Ok(name_col) = first_row.column("name") {
                    if let Ok(name_str) = name_col.str() {
                        if let Some(name) = name_str.get(0) {
                            println!("     Name: {}", name);
                        }
                    }
                }

                // Extract feature code if available
                if let Ok(feature_col) = first_row.column("feature_code") {
                    if let Ok(feature_str) = feature_col.str() {
                        if let Some(feature) = feature_str.get(0) {
                            println!("     Type: {}", feature);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
