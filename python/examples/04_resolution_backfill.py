#!/usr/bin/env python3
"""
Resolution and Backfill Example for Heisenberg

This example demonstrates the resolution and backfill functionality, which
enriches location data by finding the complete administrative hierarchy
for a given location (e.g., finding the country, state, county for a city).
"""

import heisenberg


def main():
    print("=== Heisenberg Resolution and Backfill Examples ===\n")

    # Create the Rust searcher for direct access to resolution methods
    rust_searcher = heisenberg.Heisenberg()

    # 1. Basic location resolution
    print("1. Basic Location Resolution")
    print("-" * 30)

    print("Resolving 'Paris':")
    resolved_results = rust_searcher.resolve_location(["Paris"])

    if resolved_results:
        result = resolved_results[0]
        print(f"  Score: {result.score:.3f}")
        print("  Administrative Hierarchy:")

        context = result.context
        if context.admin0:
            print(f"    Country (Admin0): {context.admin0.name}")
        if context.admin1:
            print(f"    State/Region (Admin1): {context.admin1.name}")
        if context.admin2:
            print(f"    County/Department (Admin2): {context.admin2.name}")
        if context.admin3:
            print(f"    Local Admin (Admin3): {context.admin3.name}")
        if context.admin4:
            print(f"    Sub-local Admin (Admin4): {context.admin4.name}")
        if context.place:
            print(f"    Place: {context.place.name}")
    print()

    # 2. Multi-term resolution for better accuracy
    print("2. Multi-term Resolution")
    print("-" * 30)

    print("Resolving ['London', 'England']:")
    resolved_results = rust_searcher.resolve_location(["London", "England"])

    if resolved_results:
        result = resolved_results[0]
        print(f"  Score: {result.score:.3f}")
        print("  Complete hierarchy:")

        context = result.context
        levels = [
            ("Country", context.admin0),
            ("State/Region", context.admin1),
            ("County", context.admin2),
            ("Local Admin", context.admin3),
            ("Sub-local Admin", context.admin4),
            ("Place", context.place),
        ]

        for level_name, entry in levels:
            if entry:
                print(f"    {level_name}: {entry.name} (ID: {entry.geoname_id})")
    print()

    # 3. Batch resolution
    print("3. Batch Resolution")
    print("-" * 30)

    location_queries = [["New York"], ["Berlin", "Germany"], ["Tokyo", "Japan"]]

    print("Resolving multiple locations in batch:")
    batch_resolved = rust_searcher.resolve_location_batch(location_queries)

    for query, results in zip(location_queries, batch_resolved):
        query_str = " + ".join(query)
        print(f"\n  Query: {query_str}")

        if results:
            result = results[0]
            print(f"    Score: {result.score:.3f}")

            # Show the resolved hierarchy
            context = result.context
            if context.place:
                print(f"    Place: {context.place.name}")
            if context.admin2:
                print(f"    County/Region: {context.admin2.name}")
            if context.admin1:
                print(f"    State/Province: {context.admin1.name}")
            if context.admin0:
                print(f"    Country: {context.admin0.name}")
        else:
            print("    No results found")
    print()

    # 4. Using different entry types
    print("4. Different Entry Types")
    print("-" * 30)

    # BasicEntry provides minimal information
    print("Using BasicEntry (minimal data):")
    config = heisenberg.RustSearchConfigBuilder.fast().build()
    basic_results = rust_searcher.resolve_location_with_config(["Paris"], config)

    if basic_results:
        result = basic_results[0]
        context = result.context
        if context.place:
            print(f"  Place: {context.place.name} (ID: {context.place.geoname_id})")
        if context.admin0:
            print(f"  Country: {context.admin0.name} (ID: {context.admin0.geoname_id})")
    print()

    # 5. Resolution with custom configuration
    print("5. Resolution with Custom Configuration")
    print("-" * 30)

    # Create configuration optimized for quality
    quality_config = (
        heisenberg.RustSearchConfigBuilder()
        .limit(5)
        .place_importance_threshold(2)  # Higher quality places only
        .comprehensive()  # Use comprehensive preset as base
        .build()
    )

    print("Resolving with quality-focused configuration:")
    quality_results = rust_searcher.resolve_location_with_config(
        ["Cambridge"], quality_config
    )

    for i, result in enumerate(quality_results[:3], 1):
        print(f"\n  Result {i} (Score: {result.score:.3f}):")
        context = result.context

        if context.place:
            print(f"    Place: {context.place.name}")
        if context.admin1:
            print(f"    State/Region: {context.admin1.name}")
        if context.admin0:
            print(f"    Country: {context.admin0.name}")
    print()

    # 6. Working with resolved results
    print("6. Working with Resolved Results")
    print("-" * 30)

    resolved_results = rust_searcher.resolve_location(["San Francisco", "California"])

    if resolved_results:
        result = resolved_results[0]
        print("Working with resolved result for San Francisco:")
        print(f"  Overall score: {result.score:.3f}")

        # Check if we can get simple names
        if hasattr(result, "simple"):
            simple_names = result.simple()
            print(f"  Simple names: {simple_names}")

        # Check if we can get full names
        if hasattr(result, "full"):
            full_names = result.full()
            print(f"  Full names: {full_names}")

        # Access individual components
        context = result.context
        components = []

        if context.place:
            components.append(f"Place: {context.place.name}")
        if context.admin2:
            components.append(f"County: {context.admin2.name}")
        if context.admin1:
            components.append(f"State: {context.admin1.name}")
        if context.admin0:
            components.append(f"Country: {context.admin0.name}")

        print(f"  Complete path: {' → '.join(components)}")

    print("\n=== Resolution and backfill examples completed! ===")


if __name__ == "__main__":
    main()
