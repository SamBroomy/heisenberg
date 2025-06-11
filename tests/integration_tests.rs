//! Integration tests for Heisenberg location search
//!
//! These tests run against the full public API and verify that core functionality
//! works correctly. They use test data (controlled by USE_TEST_DATA environment variable)
//! for faster execution.

use heisenberg::{
    BasicEntry, GenericEntry, LocationEntryCore, LocationSearcher, SearchConfigBuilder, DataSource,
};

fn setup_test_env() {
    let _ = heisenberg::init_logging(tracing::Level::WARN);
}

#[test]
fn test_full_workflow() {
    setup_test_env();

    // Test the complete workflow from search to resolution
    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");

    // 1. Basic search
    let search_results = searcher
        .search(&["United States"])
        .expect("Search should work");
    assert!(
        !search_results.is_empty(),
        "Should find results for United States"
    );

    // 2. Search with configuration
    let config = SearchConfigBuilder::fast().limit(3).build();

    let limited_results = searcher
        .search_with_config(&["California"], &config)
        .expect("Configured search should work");
    assert!(limited_results.len() <= 3, "Should respect limit");

    // 3. Multi-term search
    let multi_results = searcher
        .search(&["San Francisco", "California"])
        .expect("Multi-term search should work");
    assert!(
        !multi_results.is_empty(),
        "Should find results for San Francisco, California"
    );

    // 4. Resolution with GenericEntry
    let resolved_generic = searcher
        .resolve_location::<_, GenericEntry>(&["San Francisco"])
        .expect("Resolution should work");
    assert!(!resolved_generic.is_empty(), "Should resolve San Francisco");

    let context = &resolved_generic[0].context;
    assert!(
        context.admin0.is_some() || context.place.is_some(),
        "Should have some administrative context"
    );

    // 5. Resolution with BasicEntry
    let resolved_basic = searcher
        .resolve_location::<_, BasicEntry>(&["California"])
        .expect("Basic resolution should work");
    assert!(!resolved_basic.is_empty(), "Should resolve California");
}

#[test]
fn test_batch_operations() {
    setup_test_env();

    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");

    // Test batch search
    let queries = vec![
        vec!["United States"],
        vec!["California"],
        vec!["San Francisco"],
    ];

    let batch_results = searcher
        .search_bulk(&queries)
        .expect("Batch search should work");

    assert_eq!(
        batch_results.len(),
        queries.len(),
        "Should have results for all queries"
    );

    // Check that each query has some results (in test data)
    for (i, results) in batch_results.iter().enumerate() {
        // Note: Some queries might not have results in test data, which is okay
        println!("Query {}: found {} results", i, results.len());
    }

    // Test batch resolution
    let simple_queries = vec![
        vec!["United States"],
        vec!["California"],
        vec!["San Francisco"],
    ];

    let batch_resolved = searcher
        .resolve_location_batch::<GenericEntry, _, _>(&simple_queries)
        .expect("Batch resolution should work");

    assert_eq!(
        batch_resolved.len(),
        simple_queries.len(),
        "Should have resolved results for all queries"
    );
}

#[test]
fn test_configuration_presets() {
    setup_test_env();

    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");

    // Test all preset configurations
    let configs = vec![
        ("fast", SearchConfigBuilder::fast().build()),
        (
            "comprehensive",
            SearchConfigBuilder::comprehensive().build(),
        ),
        (
            "quality_places",
            SearchConfigBuilder::quality_places().build(),
        ),
    ];

    for (name, config) in configs {
        let results = searcher
            .search_with_config(&["United States"], &config)
            .expect(&format!("{} config should work", name));

        println!("{} config: found {} results", name, results.len());
        assert!(
            results.len() <= config.limit,
            "Should respect limit for {}",
            name
        );
    }
}

#[test]
fn test_error_handling() {
    setup_test_env();

    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");

    // Test various edge cases that should not panic
    let long_string = "a".repeat(1000);
    let edge_cases = vec![
        vec![""],                  // Empty string
        vec!["   "],               // Whitespace only
        vec!["XYZ123NONEXISTENT"], // Non-existent location
        vec![&long_string],        // Very long string
        vec![],                    // Empty vector
    ];

    for case in edge_cases {
        let result = searcher.search(&case);
        assert!(
            result.is_ok(),
            "Search should not error for edge case: {:?}",
            case
        );

        // Resolution should also not error
        let resolved = searcher.resolve_location::<_, BasicEntry>(&case);
        assert!(
            resolved.is_ok(),
            "Resolution should not error for edge case: {:?}",
            case
        );
    }
}

#[test]
fn test_search_result_properties() {
    setup_test_env();

    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");
    let results = searcher
        .search(&["United States"])
        .expect("Search should work");

    if let Some(result) = results.first() {
        // Test basic properties
        assert!(
            !result.name().unwrap_or("").is_empty(),
            "Result should have a name"
        );
        assert!(
            result.geoname_id().unwrap_or(0) > 0,
            "Result should have a valid geoname_id"
        );
        assert!(
            result.score().unwrap_or(0.0) >= 0.0,
            "Score should be non-negative"
        );
        assert!(
            !result.feature_code().unwrap_or("").is_empty(),
            "Result should have a feature code"
        );

        // Test that we can convert to different types
        match result {
            heisenberg::SearchResult::Admin(_) => {
                println!(
                    "Found administrative entity: {}",
                    result.name().unwrap_or("Unknown")
                );
            }
            heisenberg::SearchResult::Place(_) => {
                println!("Found place: {}", result.name().unwrap_or("Unknown"));
            }
        }
    }
}

#[test]
fn test_resolution_context() {
    setup_test_env();

    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");

    // Test resolution with multi-term query for better context
    let resolved = searcher
        .resolve_location::<_, GenericEntry>(&["San Francisco", "California"])
        .expect("Resolution should work");

    if let Some(result) = resolved.first() {
        let context = &result.context;

        // Check that we have some level of administrative hierarchy
        let levels = [
            ("admin0", context.admin0.as_ref()),
            ("admin1", context.admin1.as_ref()),
            ("admin2", context.admin2.as_ref()),
            ("admin3", context.admin3.as_ref()),
            ("admin4", context.admin4.as_ref()),
            ("place", context.place.as_ref()),
        ];

        let populated_levels: Vec<_> = levels.iter().filter(|(_, entry)| entry.is_some()).collect();

        assert!(
            !populated_levels.is_empty(),
            "Should have at least one level populated"
        );

        println!("Populated levels for San Francisco, California:");
        for (level_name, entry) in populated_levels {
            if let Some(entry) = entry {
                println!(
                    "  {}: {} (ID: {})",
                    level_name,
                    entry.name(),
                    entry.geoname_id()
                );
            }
        }

        // Test score
        assert!(
            result.score >= 0.0,
            "Resolution score should be non-negative"
        );
        assert!(result.score <= 1.0, "Resolution score should be <= 1.0");
    }
}

#[test]
fn test_custom_configuration() {
    setup_test_env();

    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");

    // Test custom configuration building
    let custom_config = SearchConfigBuilder::new()
        .limit(7)
        .place_importance_threshold(3)
        .max_admin_terms(4)
        .include_all_columns()
        .build();

    assert_eq!(custom_config.limit, 7);
    assert_eq!(custom_config.place_min_importance_tier, 3);
    assert_eq!(custom_config.max_sequential_admin_terms, 4);
    assert!(custom_config.all_cols);

    // Test that the configuration works in practice
    let results = searcher
        .search_with_config(&["California"], &custom_config)
        .expect("Custom config search should work");

    assert!(
        results.len() <= custom_config.limit,
        "Should respect custom limit"
    );
}

#[test]
fn test_concurrent_access() {
    setup_test_env();

    let searcher = LocationSearcher::new_embedded().expect("Should create searcher");

    // Test that the searcher can be used concurrently
    use std::sync::Arc;
    use std::thread;

    let searcher = Arc::new(searcher);
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let searcher_clone = Arc::clone(&searcher);
            thread::spawn(move || {
                let query = match i {
                    0 => vec!["United States"],
                    1 => vec!["California"],
                    _ => vec!["San Francisco"],
                };

                let results = searcher_clone.search(&query);
                assert!(results.is_ok(), "Concurrent search {} should work", i);
                results.unwrap()
            })
        })
        .collect();

    // Wait for all threads and collect results
    let all_results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    assert_eq!(all_results.len(), 3, "Should have results from all threads");
}

#[test]
fn test_constructor_patterns() {
    setup_test_env();

    // Test 1: new_embedded (should always work with test data)
    let embedded_searcher = LocationSearcher::new_embedded().expect("Embedded searcher should work");
    let results = embedded_searcher.search(&["United States"]).expect("Search should work");
    assert!(!results.is_empty(), "Embedded searcher should find results");

    // Test 2: initialize with test data source
    let smart_searcher = LocationSearcher::initialize(DataSource::TestData).expect("Smart initialization should work");
    let results = smart_searcher.search(&["California"]).expect("Search should work");
    // Note: results might be empty for test data, but the call should succeed

    // Test 3: load_existing (might return None, but should not error)
    let existing_result = LocationSearcher::load_existing(DataSource::TestData).expect("Load existing should not error");
    match existing_result {
        Some(existing_searcher) => {
            let results = existing_searcher.search(&["San Francisco"]).expect("Search should work");
            println!("Found existing searcher with {} results for San Francisco", results.len());
        }
        None => {
            println!("No existing searcher found for TestData (expected)");
        }
    }

    // Test 4: new_with_fresh_indexes (should work but might take longer)
    let fresh_searcher = LocationSearcher::new_with_fresh_indexes(DataSource::TestData).expect("Fresh searcher should work");
    let results = fresh_searcher.search(&["United States"]).expect("Search should work");
    // Note: results might be empty for test data, but the call should succeed
}

#[test]
fn test_data_source_enum() {
    // Test DataSource functionality
    assert_eq!(DataSource::default(), DataSource::Cities15000);

    // Test string conversion
    assert_eq!(DataSource::Cities15000.to_string(), "cities15000");
    assert_eq!(DataSource::TestData.to_string(), "test_data");

    // Test parsing from string
    use std::str::FromStr;
    assert_eq!(DataSource::from_str("cities15000").unwrap(), DataSource::Cities15000);
    assert_eq!(DataSource::from_str("test_data").unwrap(), DataSource::TestData);
    assert!(DataSource::from_str("invalid").is_err());
}
