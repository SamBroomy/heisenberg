#!/usr/bin/env python3
"""
Configuration and Search Options Example for Heisenberg

This example demonstrates how to customize search behavior using different
configurations and search options. It shows how to tune the search for
different use cases like speed vs. accuracy.
"""

import heisenberg


def main() -> None:
    print("=== Heisenberg Configuration Examples ===\n")

    # Basic searcher with embedded data
    searcher = heisenberg.LocationSearcher()

    # Show data source usage
    print("0. Data Source Configuration")
    print("-" * 30)

    # Use different data sources
    embedded_ds = heisenberg.DataSource.embedded()
    print(f"Using embedded data source: {embedded_ds}")

    # Example of using a different data source
    cities5k_ds = heisenberg.DataSource.cities5000()
    print(f"Alternative data source: {cities5k_ds}")
    print("Note: This example uses embedded data for speed\n")

    # 1. Using preset configurations
    print("1. Preset Configurations")
    print("-" * 30)

    # Fast configuration - fewer results, optimized for speed
    print("Fast configuration (optimized for speed):")
    fast_config = heisenberg.SearchConfigBuilder.fast().build()
    results = searcher.find("New York", fast_config)
    print(f"  Found {len(results)} results with fast config")
    if results:
        print(f"  Top result: {results[0].name} (Score: {results[0].score:.3f})")
    print()

    # Comprehensive configuration - more results, higher accuracy
    print("Comprehensive configuration (optimized for accuracy):")
    comprehensive_config = heisenberg.SearchConfigBuilder.comprehensive().build()
    results = searcher.find("New York", comprehensive_config)
    print(f"  Found {len(results)} results with comprehensive config")
    if results:
        print(f"  Top result: {results[0].name} (Score: {results[0].score:.3f})")
    print()

    # Quality places configuration - focuses on important places
    print("Quality places configuration (major cities/landmarks):")
    quality_config = heisenberg.SearchConfigBuilder.quality_places().build()
    results = searcher.find("Cambridge", quality_config)
    print(f"  Found {len(results)} results with quality places config")
    if results:
        print(f"  Top result: {results[0].name} (Score: {results[0].score:.3f})")
    print()

    # 2. Custom configurations
    print("2. Custom Configurations")
    print("-" * 30)

    # Build a custom configuration
    custom_config = (
        heisenberg.SearchConfigBuilder()
        .limit(15)  # Return up to 15 results
        .place_importance(3)  # Focus on moderately important places
        .admin_search(enabled=True)  # Enable administrative search
        .fuzzy_search(enabled=False)  # Disable fuzzy matching for exact matches
        .build()
    )

    print("Custom configuration with specific settings:")
    results = searcher.find("Los Angeles", custom_config)
    print(f"  Found {len(results)} results with custom config")
    print(f"  Configuration - Limit: {custom_config.limit}, Importance: {custom_config.place_importance_threshold}")
    if results:
        print(f"  Top result: {results[0].name} (Score: {results[0].score:.3f})")
    print()

    # 3. Location bias (geographical preference)
    print("3. Location Bias")
    print("-" * 30)

    # Search near London coordinates
    london_config = (
        heisenberg.SearchConfigBuilder()
        .limit(5)
        .location_bias(51.5074, -0.1278)  # London coordinates
        .build()
    )

    print("Searching for 'Cambridge' with London bias:")
    results = searcher.find("Cambridge", london_config)
    if results:
        print(f"  Top result: {results[0].name} (Score: {results[0].score:.3f})")
        print("  This should prefer Cambridge, UK over Cambridge, MA")
    print()

    # Search near Boston coordinates
    boston_config = (
        heisenberg.SearchConfigBuilder()
        .limit(5)
        .location_bias(42.3601, -71.0589)  # Boston coordinates
        .build()
    )

    print("Searching for 'Cambridge' with Boston bias:")
    results = searcher.find("Cambridge", boston_config)
    if results:
        print(f"  Top result: {results[0].name} (Score: {results[0].score:.3f})")
        print("  This should prefer Cambridge, MA over Cambridge, UK")
    print()

    # 4. Advanced configuration with custom weights
    print("4. Advanced Configuration with Custom Weights")
    print("-" * 30)

    # Create configuration that prioritizes text matching
    text_focused_config = (
        heisenberg.SearchConfigBuilder()
        .limit(10)
        .admin_weights(text=0.7, population=0.1, parent=0.1, feature=0.1)
        .place_weights(text=0.6, importance=0.2, feature=0.1, parent=0.05, distance=0.05)
        .build()
    )

    print("Configuration optimized for text matching:")
    results = searcher.find("Springfield", text_focused_config)
    print(f"  Found {len(results)} results")
    if results:
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result.name} (Score: {result.score:.3f})")
    print()

    # 5. SearchOptions validation
    print("5. SearchOptions Validation")
    print("-" * 30)

    try:
        # Valid options
        valid_options = heisenberg.SearchOptions(limit=20, place_importance_threshold=3, max_admin_terms=5)
        print(f"✓ Valid options created: limit={valid_options.limit}")
    except ValueError as e:
        print(f"✗ Error creating valid options: {e}")

    try:
        # Invalid options - should raise error
        heisenberg.SearchOptions(
            place_importance_threshold=10  # Out of range (1-5)
        )
    except ValueError as e:
        print(f"✓ Correctly caught invalid option: {e}")

    print("\n=== Configuration examples completed! ===")


if __name__ == "__main__":
    main()
