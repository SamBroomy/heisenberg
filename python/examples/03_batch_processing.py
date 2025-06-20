#!/usr/bin/env python3
"""
Batch Processing Example for Heisenberg

This example demonstrates how to efficiently process multiple location queries
in batches, which is much faster than processing them individually when you
have many locations to resolve.
"""

import time

import heisenberg


def main() -> None:
    print("=== Heisenberg Batch Processing Examples ===\n")

    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

    # 1. Basic batch processing
    print("1. Basic Batch Processing")
    print("-" * 30)

    # Prepare multiple queries
    city_queries = [["London"], ["Paris"], ["Tokyo"], ["New York"], ["Berlin"]]

    print("Processing multiple cities in batch:")
    batch_results = searcher.find_batch(city_queries)

    for query, results in zip(city_queries, batch_results, strict=False):
        query_str = " + ".join(query)
        if results:
            print(f"  {query_str}: {results[0].name} (Score: {results[0].score:.3f})")
        else:
            print(f"  {query_str}: No results found")
    print()

    # 2. Multi-term batch queries
    print("2. Multi-term Batch Queries")
    print("-" * 30)

    # Note: Use largest to smallest location order for optimal results
    multi_term_queries = [
        ["France", "Paris"],
        ["England", "London"],
        ["Japan", "Tokyo"],
        ["USA", "New York"],
        ["Germany", "Berlin"],
    ]

    print("Processing city + country queries in batch:")
    batch_results = searcher.find_batch(multi_term_queries)

    for query, results in zip(multi_term_queries, batch_results, strict=False):
        query_str = " + ".join(query)
        if results:
            print(f"  {query_str}: {results[0].name} (Score: {results[0].score:.3f})")
        else:
            print(f"  {query_str}: No results found")
    print()

    # 3. Performance comparison: individual vs batch
    print("3. Performance Comparison")
    print("-" * 30)

    test_queries = [["Madrid"], ["Rome"], ["Vienna"], ["Amsterdam"], ["Brussels"]]

    # Time individual processing
    start_time = time.time()
    individual_results = []
    for query in test_queries:
        results = searcher.find(query)
        individual_results.append(results)
    individual_time = time.time() - start_time

    # Time batch processing
    start_time = time.time()
    batch_results = searcher.find_batch(test_queries)
    batch_time = time.time() - start_time

    print(f"Individual processing time: {individual_time:.3f} seconds")
    print(f"Batch processing time: {batch_time:.3f} seconds")
    print(f"Speedup: {individual_time / batch_time:.1f}x faster")
    print()

    # 4. Batch processing with configuration
    print("4. Batch Processing with Configuration")
    print("-" * 30)

    # Use fast configuration for batch processing
    fast_config = heisenberg.SearchConfigBuilder.fast().build()

    diverse_queries = [
        ["California"],  # State
        ["Los Angeles"],  # City
        ["USA"],  # Country
        ["Silicon Valley"],  # Region/Area
        ["Golden Gate Bridge"],  # Landmark
    ]

    print("Batch processing with fast configuration:")
    batch_results = searcher.find_batch(diverse_queries, fast_config)

    for query, results in zip(diverse_queries, batch_results, strict=False):
        query_str = " + ".join(query)
        if results:
            top_result = results[0]
            print(f"  {query_str}: {top_result.name}")
            print(f"    Score: {top_result.score:.3f}, Feature: {top_result.feature_code}")
        else:
            print(f"  {query_str}: No results found")
    print()

    # 5. Handling mixed quality queries
    print("5. Handling Mixed Quality Queries")
    print("-" * 30)

    mixed_queries = [
        ["NY", "New York"],  # Good query (largest to smallest)
        [""],  # Empty query
        ["XYZ123"],  # Invalid/nonsense query
        ["France", "Paris"],  # Good query (largest to smallest)
        ["   "],  # Whitespace only
        ["UK", "London"],  # Good query (largest to smallest)
    ]

    print("Processing queries with mixed quality:")
    batch_results = searcher.find_batch(mixed_queries)

    for i, (query, results) in enumerate(zip(mixed_queries, batch_results, strict=False)):
        query_str = " + ".join([q.strip() for q in query if q.strip()])
        if not query_str:
            query_str = "[empty/invalid]"

        if results:
            print(f"  Query {i + 1} ({query_str}): {results[0].name} (Score: {results[0].score:.3f})")
        else:
            print(f"  Query {i + 1} ({query_str}): No results found")
    print()

    # 6. Convenience function for batch processing
    print("6. Convenience Function for Batches")
    print("-" * 30)

    simple_queries = [["Chicago"], ["Miami"], ["Seattle"]]

    print("Using convenience function find_locations_batch:")
    convenience_results = heisenberg.find_locations_batch(simple_queries)

    for query, results in zip(simple_queries, convenience_results, strict=False):
        query_str = " + ".join(query)
        if results:
            print(f"  {query_str}: {results[0].name} (Score: {results[0].score:.3f})")

    print("\n=== Batch processing examples completed! ===")


if __name__ == "__main__":
    main()
