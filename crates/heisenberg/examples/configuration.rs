//! Search configuration and customization
//!
//! This example demonstrates how to customize search behavior using
//! different configurations for various use cases and requirements.

use heisenberg::{LocationSearcher, SearchConfig, SearchConfigBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use embedded data for fast startup
    let searcher = LocationSearcher::new_embedded()?;
    let query = &["Cambridge"];

    println!("Comparing different search configurations for 'Cambridge':\n");

    // Preset configurations
    test_preset_configs(&searcher, query)?;

    // Custom configurations
    test_custom_configs(&searcher, query)?;

    // Configuration for specific use cases
    test_use_case_configs(&searcher, query)?;

    Ok(())
}

fn test_preset_configs(
    searcher: &LocationSearcher,
    query: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Preset configurations:");

    // Fast configuration - optimized for speed
    let fast_config = SearchConfigBuilder::fast().build();
    let fast_results = searcher.search_with_config(query, &fast_config)?;
    println!(
        "  Fast:          {} results (limit: {})",
        fast_results.len(),
        fast_config.limit
    );

    // Comprehensive configuration - optimized for accuracy
    let comprehensive_config = SearchConfigBuilder::comprehensive().build();
    let comprehensive_results = searcher.search_with_config(query, &comprehensive_config)?;
    println!(
        "  Comprehensive: {} results (limit: {})",
        comprehensive_results.len(),
        comprehensive_config.limit
    );

    // Quality places configuration - focus on important places
    let quality_config = SearchConfigBuilder::quality_places().build();
    let quality_results = searcher.search_with_config(query, &quality_config)?;
    println!(
        "  Quality:       {} results (importance tier: {})\n",
        quality_results.len(),
        quality_config.place_min_importance_tier
    );

    Ok(())
}

fn test_custom_configs(
    searcher: &LocationSearcher,
    query: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Custom configurations:");

    // Minimal configuration - very fast, few results
    let minimal_config = SearchConfigBuilder::new()
        .limit(3)
        .max_admin_terms(2)
        .place_importance_threshold(1) // Only most important places
        .build();

    let minimal_results = searcher.search_with_config(query, &minimal_config)?;
    println!("  Minimal:       {} results", minimal_results.len());

    // Detailed configuration - include all data columns
    let detailed_config = SearchConfigBuilder::new()
        .limit(10)
        .include_all_columns()
        .build();

    let detailed_results = searcher.search_with_config(query, &detailed_config)?;
    println!(
        "  Detailed:      {} results (all columns included)",
        detailed_results.len()
    );

    // Location-biased configuration
    let london_biased_config = SearchConfigBuilder::new().limit(5).build(); // Note: location bias would need additional API support

    let biased_results = searcher.search_with_config(query, &london_biased_config)?;
    println!("  London-biased: {} results\n", biased_results.len());

    Ok(())
}

fn test_use_case_configs(
    searcher: &LocationSearcher,
    query: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Use case specific configurations:");

    // API backend - fast response, reasonable results
    let api_config = SearchConfigBuilder::fast().limit(10).build();

    let api_results = searcher.search_with_config(query, &api_config)?;
    println!("  API backend:   {} results", api_results.len());

    // Data processing - comprehensive, all data
    let processing_config = SearchConfigBuilder::comprehensive()
        .include_all_columns()
        .build();

    let processing_results = searcher.search_with_config(query, &processing_config)?;
    println!("  Data proc:     {} results", processing_results.len());

    // Autocomplete - very fast, top results only
    let autocomplete_config = SearchConfigBuilder::fast()
        .limit(5)
        .place_importance_threshold(2)
        .build();

    let autocomplete_results = searcher.search_with_config(query, &autocomplete_config)?;
    println!("  Autocomplete:  {} results", autocomplete_results.len());

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_config() -> SearchConfig {
    // Example of building a completely custom configuration
    SearchConfigBuilder::new()
        .limit(20)
        .place_importance_threshold(3)
        .max_admin_terms(5)
        .include_all_columns()
        .proactive_admin_search(true)
        .text_search(true, 2) // Enable fuzzy search with 2x limit multiplier
        .build()
}

#[cfg(test)]
mod tests {

    use super::*;

    fn setup_test_env() {
        let _ = heisenberg::init_logging(tracing::Level::WARN);
    }

    #[test]
    fn test_configuration_example() {
        setup_test_env();
        assert!(
            main().is_ok(),
            "Configuration example should run successfully"
        );
    }
}
