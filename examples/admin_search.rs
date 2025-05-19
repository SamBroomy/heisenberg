// examples/manual_admin_search.rs
use anyhow::Result;
use heisenberg::{AdminSearchParams, Heisenberg, HeisenbergError};
use polars::prelude::*;
use tracing::Level;

fn main() -> Result<(), HeisenbergError> {
    // Initialize logging
    heisenberg::init_logging(Level::INFO)?;

    // Create search service
    let search_service = Heisenberg::new(false)?;

    // Start with a country search (level 0)
    let admin0 = "The united states of america";
    println!("🔍 Searching for country: {}", admin0);
    let country_results = search_service
        .admin_search(
            admin0,
            &[0, 1], // Search for level 0 (country) and expect results that could be parents for level 1
            None::<DataFrame>, // No previous context
            &AdminSearchParams::default(),
        )?
        .unwrap_or_default();

    println!("✅ Found country: {} results", country_results.height());
    if !country_results.is_empty() {
        println!("   First result: {:?}\n", country_results.head(Some(10)));
    }

    // Search for state (level 1) using country as context
    println!("🔍 Searching for state: California");
    let state_results = search_service
        .admin_search(
            "California",
            &[1, 2], // Search for level 1 (state) and expect results that could be parents for level 2
            Some(country_results), // Pass country results as context
            &AdminSearchParams::default(),
        )?
        .unwrap_or_default();

    println!("✅ Found state: {} results", state_results.height());
    if !state_results.is_empty() {
        println!("   Result: {:?}\n", state_results.head(Some(10)));
    }

    // Search for county (level 2) using state as context
    println!("🔍 Searching for county: Los Angeles County");
    let county_results = search_service
        .admin_search(
            "Los Angeles County",
            &[2, 3], // Search for level 2 (county) and expect results that could be parents for level 3
            Some(state_results), // Pass state results as context
            &AdminSearchParams::default(),
        )?
        .unwrap_or_default();

    println!("✅ Found county: {} results", county_results.height());
    if !county_results.is_empty() {
        println!("   Result: {:?}\n", county_results.head(Some(10)));
    }

    // Search for city (level 3) using county as context
    println!("🔍 Searching for city: Beverly Hills");
    let city_results = search_service
        .admin_search(
            "Beverly Hills",
            &[3, 4], // Search for level 3 (city) and expect results that could be parents for level 4
            Some(county_results), // Pass county results as context
            &AdminSearchParams::default(),
        )?
        .unwrap_or_default();

    println!("✅ Found city: {} results", city_results.height());
    if !city_results.is_empty() {
        println!("   Result: {:?}\n", city_results.head(Some(10)));
    }

    Ok(())
}
