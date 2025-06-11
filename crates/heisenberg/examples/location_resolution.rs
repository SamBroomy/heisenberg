//! Location resolution and administrative hierarchy
//!
//! This example demonstrates how to resolve locations into complete
//! administrative hierarchies, from country level down to specific places.

use heisenberg::{BasicEntry, GenericEntry, LocationEntryCore, LocationSearcher};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let searcher = LocationSearcher::new_embedded()?;

    // Resolve location with minimal data (BasicEntry)
    println!("Basic resolution for 'Tokyo':");
    let basic_results = searcher.resolve_location::<_, BasicEntry>(&["Tokyo"])?;
    for (i, result) in basic_results.iter().take(3).enumerate() {
        println!("  {}. Score: {:.3}", i + 1, result.score);
        print_basic_context(&result.context);
    }

    // Resolve location with full data (GenericEntry)
    println!("\nFull resolution for ['California','San Francisco']:");

    let full_results =
        searcher.resolve_location::<_, GenericEntry>(&["California", "San Francisco"])?;
    for (i, result) in full_results.iter().take(2).enumerate() {
        println!("  {}. Score: {:.3}", i + 1, result.score);
        print_full_context(&result.context);
    }

    // Batch resolution for multiple locations
    println!("\nBatch resolution:");
    let queries = vec![
        vec!["Germany", "Berlin"],
        vec!["Canada", "Toronto"],
        vec!["Australia", "Sydney"],
    ];

    let batch_results = searcher.resolve_location_batch::<BasicEntry, _, _>(&queries)?;
    for (i, results) in batch_results.iter().enumerate() {
        if let Some(top_result) = results.first() {
            println!("  Query {}: Score {:.3}", i + 1, top_result.score);
            print_basic_context(&top_result.context);
        }
    }

    Ok(())
}

fn print_basic_context(context: &heisenberg::LocationContext<BasicEntry>) {
    let mut hierarchy = Vec::new();

    if let Some(admin0) = &context.admin0 {
        hierarchy.push(format!("Country: {}", admin0.name()));
    }
    if let Some(admin1) = &context.admin1 {
        hierarchy.push(format!("State: {}", admin1.name()));
    }
    if let Some(admin2) = &context.admin2 {
        hierarchy.push(format!("County: {}", admin2.name()));
    }
    if let Some(place) = &context.place {
        hierarchy.push(format!("Place: {}", place.name()));
    }

    if !hierarchy.is_empty() {
        println!("     {}", hierarchy.join(" → "));
    }
}

fn print_full_context(context: &heisenberg::LocationContext<GenericEntry>) {
    let mut hierarchy = Vec::new();

    if let Some(admin0) = &context.admin0 {
        hierarchy.push(admin0.name().to_string());
    }
    if let Some(admin1) = &context.admin1 {
        hierarchy.push(admin1.name().to_string());
    }
    if let Some(admin2) = &context.admin2 {
        hierarchy.push(admin2.name().to_string());
    }
    if let Some(place) = &context.place {
        let coords = if let (Some(lat), Some(lon)) = (place.latitude, place.longitude) {
            format!(" ({lat:.3}, {lon:.3})")
        } else {
            String::new()
        };
        hierarchy.push(format!("{}{}", place.name(), coords));
    }

    if !hierarchy.is_empty() {
        println!("     {}", hierarchy.join(" → "));
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    fn setup_test_env() {
        let _ = heisenberg::init_logging(tracing::Level::WARN);
    }

    #[test]
    fn test_location_resolution_example() {
        setup_test_env();
        assert!(
            main().is_ok(),
            "Location resolution example should run successfully"
        );
    }
}
