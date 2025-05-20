use anyhow::Result;
use heisenberg::{Heisenberg, PlaceSearchParams, error::HeisenbergError};
use polars::prelude::*;
use tracing::Level;

fn main() -> Result<(), HeisenbergError> {
    // Initialize logging
    heisenberg::init_logging(Level::INFO)?;

    // Create search service
    let search_service = Heisenberg::new(false)?;

    println!("Building administrative context for place search...");

    // First build up administrative context
    let country_results = search_service
        .admin_search("England", &[0, 1], None::<DataFrame>, &Default::default())?
        .unwrap_or_default();

    println!(
        "✅ Found country context: {} results",
        country_results.height()
    );
    println!("   Top results: {:?}", country_results.head(Some(10)));

    let city_results = search_service
        .admin_search(
            "London",
            &[1, 2],
            Some(country_results.clone()),
            &Default::default(),
        )?
        .unwrap_or_default();

    println!("✅ Found city context: {} results", city_results.height());
    println!("   Top results: {:?}", city_results.head(Some(10)));

    // Combine the administrative contexts into a single context
    let combined_context = if !country_results.is_empty() && !city_results.is_empty() {
        println!("Creating combined administrative context from country and city...");
        let combined_context = concat(
            &[country_results.lazy(), city_results.lazy()],
            UnionArgs {
                diagonal: true,
                ..Default::default()
            },
        )?
        .collect()?;

        println!(
            "✅ Combined context created with {} rows",
            combined_context.height()
        );
        println!("   Top results: {:?}", combined_context.head(Some(10)));
        Some(combined_context)
    } else {
        println!("❌ Insufficient administrative context found");
        None
    };

    // Now search for a specific place using the administrative context
    if let Some(context) = combined_context {
        // Create custom place search parameters
        let place_params = PlaceSearchParams {
            limit: 5,                   // Return top 5 results
            min_importance_tier: 3,     // Focus on more important places
            center_lat: Some(51.50722), // London coordinates as center
            center_lon: Some(-0.1275),
            ..Default::default()
        };

        println!("🔍 Searching for place: British Museum (with administrative context)");
        let place_results =
            search_service.place_search("British Museum", Some(context), &place_params)?;

        match place_results {
            Some(results) if !results.is_empty() => {
                println!("✅ Found place: {} results", results.height());
                println!("   Top results: {:?}", results.head(Some(10)));
            }
            _ => println!("❌ No results found for British Museum"),
        }
    }

    Ok(())
}
