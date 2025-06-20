#!/usr/bin/env python3
"""
Basic Usage Example for Heisenberg Location Search

This example demonstrates the most common usage patterns for the Heisenberg
location search library. It shows how to perform simple searches and access
search results.
"""

import heisenberg


def main() -> None:
    print("=== Heisenberg Location Search - Basic Usage ===\n")

    # Create a location searcher
    print("1. Creating LocationSearcher...")
    searcher = heisenberg.LocationSearcher()
    print("✓ LocationSearcher created successfully\n")

    # Simple search examples
    print("2. Simple Location Searches")
    print("-" * 30)

    # Search for a city
    print("Searching for 'London':")
    results = searcher.find("London")
    for i, result in enumerate(results[:3], 1):  # Show top 3 results
        print(f"  {i}. {result.name} (ID: {result.geoname_id}, Score: {result.score:.3f})")
    print()

    # Search for a country
    print("Searching for 'France':")
    results = searcher.find("France")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result.name} (ID: {result.geoname_id}, Score: {result.score:.3f})")
    print()

    # Multi-term search (largest to smallest location for optimal results)
    print("Searching for ['France', 'Paris']:")
    results = searcher.find(["France", "Paris"])
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result.name} (ID: {result.geoname_id}, Score: {result.score:.3f})")
    print()

    # Working with search results
    print("3. Working with Search Results")
    print("-" * 30)

    if results:
        result = results[0]
        print("Top result for Paris, France:")
        print(f"  Name: {result.name}")
        print(f"  Geoname ID: {result.geoname_id}")
        print(f"  Feature Code: {result.feature_code}")
        print(f"  Score: {result.score:.3f}")

        # Check for optional attributes
        if hasattr(result, "latitude") and result.latitude is not None:
            print(f"  Coordinates: {result.latitude}, {result.longitude}")
        if hasattr(result, "population") and result.population is not None:
            print(f"  Population: {result.population:,}")

        # Convert to dictionary
        result_dict = result.to_dict()
        print(f"  Keys in result: {list(result_dict.keys())}")
        print()

    # Convenience functions
    print("4. Convenience Functions")
    print("-" * 30)

    # Use convenience function for single searches
    print("Using convenience function find_location:")
    results = heisenberg.find_location("Tokyo")
    if results:
        print(f"  Found: {results[0].name} (Score: {results[0].score:.3f})")
    print()

    # Batch searches with convenience function
    print("Using convenience function find_locations_batch:")
    batch_queries = [["Germany", "Berlin"], ["Spain", "Madrid"], ["Italy", "Rome"]]
    batch_results = heisenberg.find_locations_batch(batch_queries)
    for query, results in zip(batch_queries, batch_results, strict=False):
        if results:
            print(f"  Query {query}: {results[0].name} (Score: {results[0].score:.3f})")

    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
