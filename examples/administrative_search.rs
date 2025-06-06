//! Low-level administrative search
//!
//! This example demonstrates the low-level administrative search API,
//! showing how to search at specific administrative levels and build
//! hierarchical context manually.

use heisenberg::{AdminSearchParams, LocationSearcher};
use polars::prelude::DataFrame;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let searcher = LocationSearcher::new(false)?;

    // Search for a country (admin level 0)
    println!("Searching for country 'United States':");
    let country_results = searcher.admin_search(
        "United States",
        &[0], // Search at country level
        None, // No previous context
        &AdminSearchParams::default(),
    )?;

    if let Some(countries) = country_results {
        println!("  Found {} countries", countries.height());

        if countries.height() > 0 {
            // Search for state (admin level 1) within the country
            println!("  Searching for state 'California' within found country:");
            let state_results = searcher.admin_search(
                "California",
                &[1],            // Search at state level
                Some(countries), // Use country as context
                &AdminSearchParams::default(),
            )?;

            if let Some(states) = state_results {
                println!("    Found {} states", states.height());

                if states.height() > 0 {
                    // Search for county (admin level 2) within the state
                    println!("    Searching for 'Los Angeles County' within found state:");
                    let county_results = searcher.admin_search(
                        "Los Angeles County",
                        &[2],         // Search at county level
                        Some(states), // Use state as context
                        &AdminSearchParams::default(),
                    )?;

                    if let Some(counties) = county_results {
                        println!("      Found {} counties", counties.height());

                        // Display some information about the final result
                        if counties.height() > 0 {
                            print_admin_result(&counties)?;
                        }
                    }
                }
            }
        }
    }

    // Demonstrate parallel search at multiple levels
    println!("\nSearching 'Berlin' at multiple admin levels:");
    let berlin_results = searcher.admin_search(
        "Berlin",
        &[0, 1, 2, 3], // Search across multiple levels
        None,
        &AdminSearchParams::default(),
    )?;

    if let Some(results) = berlin_results {
        println!(
            "  Found {} administrative entities named 'Berlin'",
            results.height()
        );
        print_admin_result(&results)?;
    }

    Ok(())
}

fn print_admin_result(df: &DataFrame) -> Result<(), Box<dyn std::error::Error>> {
    if df.height() == 0 {
        return Ok(());
    }

    // Try to extract some meaningful columns
    let columns_to_show = ["name", "admin_level", "feature_code", "population"];
    let mut available_columns = Vec::new();
    let column_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    for col_name in &columns_to_show {
        if column_names.contains(&col_name.to_string()) {
            available_columns.push(*col_name);
        }
    }

    if !available_columns.is_empty() {
        let subset = df.select(available_columns)?;
        let preview = subset.head(Some(3));
        println!("      Preview: {}", preview);
    }

    Ok(())
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
    fn test_administrative_search_example() {
        setup_test_env();
        assert!(main().is_ok(), "Administrative search example should run successfully");
    }
}
