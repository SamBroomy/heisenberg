// examples/custom_configuration.rs
use anyhow::Result;
use heisenberg::{
    Heisenberg, HeisenbergError, SearchScoreAdminParams, SearchScorePlaceParams,
    SearchConfig,
};
use tracing::Level;

fn main() -> Result<(), HeisenbergError> {
    // Initialize logging
    heisenberg::init_logging(Level::INFO)?;

    // Create search service
    let search_service = Heisenberg::new(false)?;

    // 1. Create a custom SmartFlexibleSearchConfig
    println!("Creating custom search configuration...");

    let custom_config = SearchConfig {
        limit: 15,                     // Return more results
        all_cols: true,                // Include all columns in results
        max_sequential_admin_terms: 4, // Maximum admin terms to process sequentially
        place_min_importance_tier: 3,  // Only consider more important places (1-3)
        admin_search_score_params: SearchScoreAdminParams {
            text_weight: 0.5,    // Increase weight for text relevance
            pop_weight: 0.3,     // Increase weight for population
            parent_weight: 0.1,  // Decrease parent context influence
            feature_weight: 0.1, // Decrease feature type importance
        },
        place_search_score_params: SearchScorePlaceParams {
            text_weight: 0.5,               // Prioritize text matching
            importance_weight: 0.25,        // Importance matters more
            feature_weight: 0.1,            // Feature type matters less
            parent_admin_score_weight: 0.1, // Parent admin context matters less
            distance_weight: 0.05,          // Distance matters less
        },
        ..Default::default()
    };

    // 2. Test the custom configuration with a search query
    let query = vec!["United Kingdom", "London", "Westminster", "Big Ben"];

    println!("🔍 Searching for: {:?} with custom configuration", query);

    // Perform search with custom configuration
    let results = search_service.search_with_config(&query, &custom_config)?;

    println!(
        "✅ Search completed with {} result DataFrames",
        results.len()
    );

    // 3. Compare with default configuration
    println!("\nComparing with default configuration...");

    let default_results =
        search_service.search_with_config(&query, &SearchConfig::default())?;

    println!(
        "✅ Default search completed with {} result DataFrames",
        default_results.len()
    );

    // 4. Compare results
    println!("\nResults comparison:");
    println!(
        "   Custom config: {} total rows",
        results.iter().map(|df| df.height()).sum::<usize>()
    );
    println!(
        "   Default config: {} total rows",
        default_results.iter().map(|df| df.height()).sum::<usize>()
    );

    // Show top custom result
    if !results.is_empty() && !results.last().unwrap().is_empty() {
        let top_result = results.last().unwrap().head(Some(1));

        println!("\nTop custom config result:");

        if let (Ok(name_col), Ok(feature_col), Ok(score_col)) = (
            top_result.column("name"),
            top_result.column("feature_code"),
            top_result.column("score_place"), // Assuming this is the score column name
        ) {
            if let (Ok(name_str), Ok(feature_str), Ok(score_f64)) =
                (name_col.str(), feature_col.str(), score_col.f64())
            {
                if let (Some(name), Some(feature), Some(score)) =
                    (name_str.get(0), feature_str.get(0), score_f64.get(0))
                {
                    println!("   Name: {}", name);
                    println!("   Type: {}", feature);
                    println!("   Score: {:.4}", score);
                }
            }
        }
    }

    Ok(())
}
