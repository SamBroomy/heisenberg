use anyhow::Result;
use heisenberg::{GeonameEntry, Heisenberg, HeisenbergError, SmartFlexibleSearchConfig};
use tracing::Level;

fn main() -> Result<(), HeisenbergError> {
    // Initialize logging
    heisenberg::init_logging(Level::INFO)?;

    // Create search service
    let search_service = Heisenberg::new(false)?;

    // Define search configuration
    let config = SmartFlexibleSearchConfig::default();

    // Define example queries
    let examples = [
        vec!["California", "Golden Gate Bridge"],
        vec!["London", "British Museum"],
        vec!["Paris", "Louvre Museum"],
    ];

    // Process each query individually with full resolution
    for (i, query) in examples.iter().enumerate() {
        println!(
            "\n[Example {}] Full location resolution for: {:?}",
            i + 1,
            query
        );

        // Start timing
        let start_time = std::time::Instant::now();

        // Resolve locations with GeonameEntry type and limit of 3 results per query
        let resolved_results =
            search_service.resolve_location::<_, GeonameEntry>(query, &config, 3)?;

        // Calculate elapsed time
        let elapsed = start_time.elapsed();

        println!(
            "✅ Resolution completed in {:.3} seconds",
            elapsed.as_secs_f32()
        );
        println!("   Found {} resolved locations", resolved_results.len());

        // Print detailed information about each resolved location
        for (j, result) in resolved_results.iter().enumerate() {
            println!("\n   [Result {}] Score: {:.4}", j + 1, result.score);

            // Show full location hierarchy
            println!("   📍 Location Hierarchy:");

            if let Some(ref country) = result.context.admin0 {
                println!(
                    "      Country: {} (ID: {})",
                    country.name, country.geoname_id
                );
            }

            if let Some(ref admin1) = result.context.admin1 {
                println!("      Admin1:  {} (ID: {})", admin1.name, admin1.geoname_id);
            }

            if let Some(ref admin2) = result.context.admin2 {
                println!("      Admin2:  {} (ID: {})", admin2.name, admin2.geoname_id);
            }

            if let Some(ref admin3) = result.context.admin3 {
                println!("      Admin3:  {} (ID: {})", admin3.name, admin3.geoname_id);
            }

            if let Some(ref admin4) = result.context.admin4 {
                println!("      Admin4:  {} (ID: {})", admin4.name, admin4.geoname_id);
            }

            if let Some(ref place) = result.context.place {
                println!("      Place:   {} (ID: {})", place.name, place.geoname_id);
            }

            // Print simple representation
            println!("   🔍 Simple representation: {:?}", result.simple());
        }
    }

    // Now demonstrate bulk resolution
    println!("\n===== Bulk Location Resolution =====");

    // Convert queries to required format
    let examples_refs: Vec<&[&str]> = examples.iter().map(|v| v.as_slice()).collect();

    // Start timing
    let bulk_start_time = std::time::Instant::now();

    // Resolve locations in bulk
    let bulk_resolved_results =
        search_service.resolve_batch::<GeonameEntry, _, _>(&examples_refs, &config, 3)?;

    // Calculate elapsed time
    let bulk_elapsed = bulk_start_time.elapsed();

    println!(
        "✅ Bulk resolution completed in {:.3} seconds",
        bulk_elapsed.as_secs_f32()
    );
    println!(
        "   Average time per query: {:.3} seconds",
        bulk_elapsed.as_secs_f32() / examples.len() as f32
    );
    println!("   Processed {} queries", bulk_resolved_results.len());

    // Print summary of bulk results
    for (i, results) in bulk_resolved_results.iter().enumerate() {
        println!("\n[Bulk Query {}] {:?}", i + 1, examples[i]);
        println!("   Found {} resolved locations", results.len());

        if !results.is_empty() {
            // Get top result
            let top_result = &results[0];

            // Print simple representation of top result
            println!(
                "   Top result simple representation: {:?}",
                top_result.simple()
            );
        }
    }

    Ok(())
}
