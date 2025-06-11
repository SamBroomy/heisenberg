#!/usr/bin/env python3
"""
Advanced Features Example for Heisenberg

This example demonstrates advanced features of the Heisenberg library including
direct Rust API access, low-level search methods, DataFrame operations,
and advanced configuration patterns.
"""

import heisenberg
from heisenberg._internal import RustLocationSearcher, RustSearchConfigBuilder


def main():
    print("=== Heisenberg Advanced Features Examples ===\n")

    # Create both high-level and low-level searchers
    searcher = heisenberg.LocationSearcher()
    rust_searcher = RustLocationSearcher()

    # 1. Direct Rust API access
    print("1. Direct Rust API Access")
    print("-" * 30)

    print("Using direct Rust searcher for low-level operations:")

    # Basic search with Rust API
    rust_results = rust_searcher.search(["Tokyo"])
    print(f"  Direct search for 'Tokyo': {len(rust_results)} results")
    if rust_results:
        result = rust_results[0]
        name = result.name() if callable(result.name) else result.name
        score = result.score() if callable(result.score) else result.score
        if name and score is not None:
            print(f"    Top result: {name} (Score: {score:.3f})")
        else:
            print(f"    Top result: {name or 'Unknown'} (Score: {score or 0.0:.3f})")

    # Search with Rust configuration
    rust_config = RustSearchConfigBuilder().limit(3).build()
    config_results = rust_searcher.search_with_config(["London"], rust_config)
    print(f"  Configured search for 'London': {len(config_results)} results")
    print()

    # 2. Low-level search methods
    print("2. Low-level Search Methods")
    print("-" * 30)

    # Administrative search at specific levels
    print("Administrative search for 'France' at country level (admin level 0):")
    admin_results = rust_searcher.admin_search("France", [0], None)
    if admin_results is not None:
        print(f"  Found administrative entity: shape {admin_results.shape}")
        if admin_results.height > 0:
            # Access DataFrame data
            names = admin_results.get_column("name").to_list()
            print(f"  Names: {names[:3]}")  # Show first 3 names
    else:
        print("  No administrative results found")
    print()

    # Place search
    print("Place search for 'Paris':")
    place_results = rust_searcher.place_search("Paris", None)
    if place_results is not None:
        print(f"  Found places: shape {place_results.shape}")
        if place_results.height > 0:
            names = place_results.get_column("name").to_list()
            print(f"  Place names: {names[:3]}")
    else:
        print("  No place results found")
    print()

    # 3. Advanced configuration patterns
    print("3. Advanced Configuration Patterns")
    print("-" * 30)

    # Complex configuration with chained builders
    advanced_config = (
        heisenberg.SearchConfigBuilder()
        .comprehensive()  # Start with comprehensive preset
        .limit(8)  # Override limit
        .place_importance(2)  # Only high-importance places
        .admin_search(True)  # Enable admin search
        .location_bias(40.7128, -74.0060)  # NYC coordinates
        .admin_weights(text=0.5, population=0.3, parent=0.1, feature=0.1)
        .place_weights(
            text=0.4, importance=0.3, feature=0.15, parent=0.1, distance=0.05
        )
        .build()
    )

    print("Advanced configuration with multiple customizations:")
    results = searcher.find("Manhattan", advanced_config)
    print(f"  Found {len(results)} results for 'Manhattan'")
    if results:
        for i, result in enumerate(results[:3], 1):
            print(f"    {i}. {result.name} (Score: {result.score:.3f})")
    print()

    # 4. Preset configuration comparison
    print("4. Preset Configuration Comparison")
    print("-" * 30)

    query = "Cambridge"
    configs = {
        "Fast": heisenberg.SearchConfigBuilder.fast().build(),
        "Comprehensive": heisenberg.SearchConfigBuilder.comprehensive().build(),
        "Quality Places": heisenberg.SearchConfigBuilder.quality_places().build(),
    }

    print(f"Comparing presets for query '{query}':")
    for name, config in configs.items():
        results = searcher.find(query, config)
        print(f"  {name:15} - {len(results):2d} results", end="")
        if results:
            print(f" - Top: {results[0].name} ({results[0].score:.3f})")
        else:
            print(" - No results")
    print()

    # 5. SearchResult methods and attributes
    print("5. SearchResult Methods and Attributes")
    print("-" * 30)

    results = searcher.find("New York")
    if results:
        result = results[0]
        print(f"Exploring SearchResult for '{result.name}':")

        # Basic attributes
        print(f"  Geoname ID: {result.geoname_id}")
        print(f"  Feature Code: {result.feature_code}")
        print(f"  Score: {result.score:.4f}")

        # Optional geographic data
        if hasattr(result, "latitude") and result.latitude is not None:
            print(f"  Coordinates: ({result.latitude:.4f}, {result.longitude:.4f})")

        if hasattr(result, "population") and result.population is not None:
            print(f"  Population: {result.population:,}")

        # Methods
        if hasattr(result, "admin_hierarchy"):
            hierarchy = result.admin_hierarchy()
            print(f"  Admin hierarchy: {hierarchy}")

        if hasattr(result, "full_name"):
            full_name = result.full_name()
            print(f"  Full name: {full_name}")

        # Dictionary conversion
        result_dict = result.to_dict()
        print(f"  Available fields: {sorted(result_dict.keys())}")
    print()

    # 6. Error handling and edge cases
    print("6. Error Handling and Edge Cases")
    print("-" * 30)

    # Test with various edge cases
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "XYZ123NONEXISTENT",  # Non-existent location
        "a" * 1000,  # Very long string
    ]

    print("Testing edge cases:")
    for case in edge_cases:
        try:
            results = searcher.find(case)
            print(
                f"  '{case[:20]}{'...' if len(case) > 20 else ''}': {len(results)} results"
            )
        except Exception as e:
            print(
                f"  '{case[:20]}{'...' if len(case) > 20 else ''}': Error - {type(e).__name__}"
            )
    print()

    # 7. Different search methods comparison
    print("7. Different Search Methods Comparison")
    print("-" * 30)

    if hasattr(searcher, "find_quick"):
        quick_results = searcher.find_quick("Berlin")
        print(f"Quick search for 'Berlin': {len(quick_results)} results")

    if hasattr(searcher, "find_comprehensive"):
        comprehensive_results = searcher.find_comprehensive("Berlin")
        print(
            f"Comprehensive search for 'Berlin': {len(comprehensive_results)} results"
        )

    if hasattr(searcher, "find_important_places"):
        important_results = searcher.find_important_places("Berlin")
        print(f"Important places search for 'Berlin': {len(important_results)} results")

    # Regular search for comparison
    regular_results = searcher.find("Berlin")
    print(f"Regular search for 'Berlin': {len(regular_results)} results")
    print()

    # 8. Working with multiple entry types
    print("8. Working with Multiple Entry Types")
    print("-" * 30)

    print("Available entry types:")
    entry_types = [
        "BasicEntry",
        "GenericEntry",
        "LocationContextBasic",
        "LocationContextGeneric",
        "ResolvedBasicSearchResult",
        "ResolvedGenericSearchResult",
    ]

    for entry_type in entry_types:
        if hasattr(heisenberg, entry_type):
            print(f"  ✓ {entry_type} available")
        else:
            print(f"  ✗ {entry_type} not available")

    print("\n=== Advanced features examples completed! ===")


if __name__ == "__main__":
    main()
