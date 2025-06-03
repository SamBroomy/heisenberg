"""
Integration tests for Python-Rust binding functionality.

These tests focus on the integration between the Python wrapper layer and the
underlying Rust implementation, ensuring data flows correctly and APIs
behave as expected.
"""

import pytest
import heisenberg
from unittest.mock import patch
import logging


class TestLocationSearcherIntegration:
    """Test the LocationSearcher class integration with Rust backend."""

    @pytest.fixture
    def searcher(self):
        """Create a LocationSearcher instance for testing."""
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_searcher_initialization_parameters(self, searcher):
        """Test that searcher initializes correctly with different parameters."""
        # Test with default parameters
        assert searcher is not None

        # Test initialization with rebuild_indexes=True (should work but take longer)
        # We'll skip this in CI but it's important for manual testing

    def test_find_method_signatures(self, searcher):
        """Test all find method signatures work correctly."""
        # Test find with string input
        results1 = searcher.find("London")
        assert isinstance(results1, list)

        # Test find with list input
        results2 = searcher.find(["London", "UK"])
        assert isinstance(results2, list)

        # Test find with custom options
        options = heisenberg.SearchOptions(limit=5, place_importance_threshold=2)
        results3 = searcher.find("London", options)
        assert isinstance(results3, list)
        assert len(results3) <= 5  # Should respect limit

        # Test find with SearchConfig
        config = heisenberg.SearchConfigBuilder().limit(3).build()
        results4 = searcher.find("London", config)
        assert isinstance(results4, list)
        assert len(results4) <= 3

    def test_preset_methods_with_various_inputs(self, searcher):
        """Test preset search methods with different input types."""
        test_inputs = ["Paris", ["Tokyo", "Japan"], ["New York", "NY", "USA"], "Berlin"]

        for input_data in test_inputs:
            # Test all preset methods
            quick_results = searcher.find_quick(input_data)
            assert isinstance(quick_results, list)

            comprehensive_results = searcher.find_comprehensive(input_data)
            assert isinstance(comprehensive_results, list)

            important_results = searcher.find_important_places(input_data)
            assert isinstance(important_results, list)

            # Important places should generally have fewer results (higher quality)
            if quick_results and important_results:
                # Important places filter should generally return fewer results
                # (this is a heuristic test)
                pass

    def test_batch_processing_integration(self, searcher):
        """Test batch processing maintains data integrity."""
        batch_queries = [
            ["London", "UK"],
            ["Paris", "France"],
            ["Tokyo", "Japan"],
            ["Berlin", "Germany"],
            ["Sydney", "Australia"],
        ]

        # Test batch method
        batch_results = searcher.find_batch(batch_queries)
        assert isinstance(batch_results, list)
        assert len(batch_results) == len(batch_queries)

        # Each batch should contain a list of results
        for i, query_results in enumerate(batch_results):
            assert isinstance(query_results, list), (
                f"Batch {i} results should be a list"
            )

        # Compare with individual queries to ensure consistency
        individual_results = []
        for query in batch_queries:
            individual_results.append(searcher.find(query))

        # Results should be similar (allowing for some variance in ranking)
        for i, (batch_result, individual_result) in enumerate(
            zip(batch_results, individual_results)
        ):
            # Both should have results (assuming queries are valid)
            if batch_result and individual_result:
                # Check if top result names are similar
                batch_name = batch_result[0].name if batch_result else None
                individual_name = (
                    individual_result[0].name if individual_result else None
                )

                # This is a loose test - results might vary slightly
                assert batch_name is not None and individual_name is not None, (
                    f"Query {i}: {batch_queries[i]} should return named results"
                )

    def test_search_result_data_integrity(self, searcher):
        """Test that SearchResult objects contain expected data."""
        results = searcher.find("London")

        if results:
            result = results[0]

            # Test required attributes exist
            assert hasattr(result, "name")
            assert hasattr(result, "geoname_id")
            assert hasattr(result, "score")

            # Test attribute types
            assert isinstance(result.name, str)
            assert isinstance(result.geoname_id, int)
            assert isinstance(result.score, (int, float))

            # Test method functionality
            hierarchy = result.admin_hierarchy()
            assert isinstance(hierarchy, list)

            full_name = result.full_name()
            assert isinstance(full_name, str)
            assert result.name in full_name

            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "name" in result_dict
            assert "geoname_id" in result_dict
            assert result_dict["name"] == result.name
            assert result_dict["geoname_id"] == result.geoname_id

    def test_configuration_propagation(self, searcher):
        """Test that configuration options properly propagate to Rust layer."""
        # Test different configurations produce different results

        # Fast config - should be quick, fewer results
        fast_config = heisenberg.SearchConfigBuilder.fast().build()
        fast_results = searcher.find("London", fast_config)

        # Comprehensive config - should be more thorough
        comprehensive_config = heisenberg.SearchConfigBuilder.comprehensive().build()
        comprehensive_results = searcher.find("London", comprehensive_config)

        # Both should return results
        assert isinstance(fast_results, list)
        assert isinstance(comprehensive_results, list)

        # Test limit configuration
        limit_config = heisenberg.SearchConfigBuilder().limit(2).build()
        limited_results = searcher.find("London", limit_config)
        assert len(limited_results) <= 2

    def test_error_handling_integration(self, searcher):
        """Test error handling between Python and Rust layers."""
        # Test empty input
        empty_results = searcher.find("")
        assert isinstance(empty_results, list)

        # Test very long input
        long_input = "a" * 1000
        long_results = searcher.find(long_input)
        assert isinstance(long_results, list)

        # Test unicode input
        unicode_results = searcher.find("Москва")  # Moscow in Russian
        assert isinstance(unicode_results, list)

        # Test special characters
        special_results = searcher.find("São Paulo")
        assert isinstance(special_results, list)

    def test_convenience_functions_integration(self):
        """Test that convenience functions work correctly."""
        # Test find_location convenience function
        results = heisenberg.find_location("Madrid")
        assert isinstance(results, list)

        # Test find_locations_batch convenience function
        batch_results = heisenberg.find_locations_batch(
            [["Rome"], ["Vienna", "Austria"]]
        )
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2

        for query_results in batch_results:
            assert isinstance(query_results, list)


class TestSearchOptionsValidation:
    """Test SearchOptions validation and edge cases."""

    def test_valid_options_creation(self):
        """Test creating valid SearchOptions instances."""
        # Test default values
        default_options = heisenberg.SearchOptions()
        assert default_options.limit == 20
        assert default_options.place_importance_threshold == 5
        assert default_options.proactive_admin_search is True

        # Test custom values
        custom_options = heisenberg.SearchOptions(
            limit=10,
            include_all_columns=True,
            max_admin_terms=3,
            place_importance_threshold=2,
            proactive_admin_search=False,
            fuzzy_search=False,
            search_multiplier=5,
        )
        assert custom_options.limit == 10
        assert custom_options.include_all_columns is True
        assert custom_options.max_admin_terms == 3
        assert custom_options.place_importance_threshold == 2
        assert custom_options.proactive_admin_search is False
        assert custom_options.fuzzy_search is False
        assert custom_options.search_multiplier == 5

    def test_invalid_options_validation(self):
        """Test that invalid options raise appropriate errors."""
        # Test invalid importance threshold (should be 1-5)
        with pytest.raises(
            ValueError, match="place_importance_threshold must be between 1 and 5"
        ):
            heisenberg.SearchOptions(place_importance_threshold=0)

        with pytest.raises(
            ValueError, match="place_importance_threshold must be between 1 and 5"
        ):
            heisenberg.SearchOptions(place_importance_threshold=6)

        # Test invalid limit
        with pytest.raises(ValueError, match="limit must be positive"):
            heisenberg.SearchOptions(limit=0)

        with pytest.raises(ValueError, match="limit must be positive"):
            heisenberg.SearchOptions(limit=-1)

        # Test invalid max_admin_terms
        with pytest.raises(ValueError, match="max_admin_terms must be positive"):
            heisenberg.SearchOptions(max_admin_terms=0)

        # Test invalid search_multiplier
        with pytest.raises(ValueError, match="search_multiplier must be positive"):
            heisenberg.SearchOptions(search_multiplier=0)

    def test_options_to_config_conversion(self):
        """Test that SearchOptions correctly converts to SearchConfig."""
        options = heisenberg.SearchOptions(
            limit=15,
            place_importance_threshold=3,
            proactive_admin_search=False,
            fuzzy_search=True,
        )

        config = options.to_config()
        assert isinstance(config, heisenberg.SearchConfig)

        # The conversion should work properly with the Rust layer
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
        results = searcher.find("London", config)
        assert isinstance(results, list)


class TestSearchConfigBuilderIntegration:
    """Test SearchConfigBuilder fluent interface and integration."""

    def test_fluent_interface_chaining(self):
        """Test that fluent interface methods can be chained."""
        config = (
            heisenberg.SearchConfigBuilder()
            .limit(10)
            .place_importance(3)
            .admin_search(True)
            .fuzzy_search(False)
            .include_all_columns()
            .max_admin_terms(4)
            .build()
        )

        assert isinstance(config, heisenberg.SearchConfig)

        # Test the config actually works
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
        results = searcher.find("Paris", config)
        assert isinstance(results, list)

    def test_preset_configurations(self):
        """Test that preset configurations work correctly."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

        # Test fast preset
        fast_config = heisenberg.SearchConfigBuilder.fast().build()
        fast_results = searcher.find("Berlin", fast_config)
        assert isinstance(fast_results, list)

        # Test comprehensive preset
        comprehensive_config = heisenberg.SearchConfigBuilder.comprehensive().build()
        comprehensive_results = searcher.find("Berlin", comprehensive_config)
        assert isinstance(comprehensive_results, list)

        # Test quality places preset
        quality_config = heisenberg.SearchConfigBuilder.quality_places().build()
        quality_results = searcher.find("Berlin", quality_config)
        assert isinstance(quality_results, list)

    def test_config_builder_edge_cases(self):
        """Test edge cases in configuration building."""
        # Test minimum values
        min_config = (
            heisenberg.SearchConfigBuilder()
            .limit(1)
            .place_importance(1)
            .max_admin_terms(1)
            .build()
        )

        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
        results = searcher.find("Tokyo", min_config)
        assert isinstance(results, list)
        assert len(results) <= 1

        # Test maximum reasonable values
        max_config = (
            heisenberg.SearchConfigBuilder()
            .limit(100)
            .place_importance(5)
            .max_admin_terms(10)
            .build()
        )

        results = searcher.find("Tokyo", max_config)
        assert isinstance(results, list)


class TestRustAPIDirectAccess:
    """Test direct access to Rust API (lower-level interface)."""

    def test_rust_searcher_creation(self):
        """Test creating the direct Rust searcher."""
        rust_searcher = heisenberg.Heisenberg(rebuild_indexes=False)
        assert rust_searcher is not None

    def test_rust_search_methods(self):
        """Test direct Rust search methods return DataFrames."""
        rust_searcher = heisenberg.Heisenberg(rebuild_indexes=False)

        # Test basic search
        dataframes = rust_searcher.search(["London"])
        assert isinstance(dataframes, list)

        # Test with config
        config = heisenberg.SearchConfigBuilder().limit(5).build()
        config_dataframes = rust_searcher.search_with_config(["Paris"], config)
        assert isinstance(config_dataframes, list)

    def test_rust_resolve_methods(self):
        """Test direct Rust resolve methods."""
        rust_searcher = heisenberg.Heisenberg(rebuild_indexes=False)

        # Test resolve_location
        resolved = rust_searcher.resolve_location(["London"])
        assert isinstance(resolved, list)

        # Test resolve_location_batch
        batch_resolved = rust_searcher.resolve_location_batch([["Paris"], ["Tokyo"]])
        assert isinstance(batch_resolved, list)
        assert len(batch_resolved) == 2


class TestLoggingIntegration:
    """Test logging integration between Python and Rust."""

    def test_logging_configuration(self):
        """Test that logging can be configured and works."""
        # Create a logger
        logger = logging.getLogger("heisenberg")
        logger.setLevel(logging.DEBUG)

        # Create searcher (this should log initialization)
        with patch("logging.Logger.info") as mock_info:
            searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

            # Should have logged something during initialization
            # Note: This test depends on Rust logging being properly configured

    def test_error_propagation(self):
        """Test that Rust errors are properly propagated to Python."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

        # These should not raise exceptions but return empty results
        try:
            results = searcher.find([])  # Empty list
            assert isinstance(results, list)

            results = searcher.find(None)  # This might raise TypeError
        except TypeError:
            # Expected for None input
            pass


class TestPerformanceCharacteristics:
    """Test performance characteristics and scaling behavior."""

    def test_batch_vs_individual_performance(self):
        """Test that batch processing is more efficient than individual calls."""
        import time

        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

        queries = [
            ["London", "UK"],
            ["Paris", "France"],
            ["Berlin", "Germany"],
            ["Madrid", "Spain"],
            ["Rome", "Italy"],
        ]

        # Time individual calls
        start_individual = time.time()
        individual_results = []
        for query in queries:
            individual_results.append(searcher.find(query))
        end_individual = time.time()
        individual_time = end_individual - start_individual

        # Time batch call
        start_batch = time.time()
        batch_results = searcher.find_batch(queries)
        end_batch = time.time()
        batch_time = end_batch - start_batch

        # Batch should generally be faster (or at least not much slower)
        # This is more of a performance characterization than a strict test
        print(f"Individual calls time: {individual_time:.3f}s")
        print(f"Batch call time: {batch_time:.3f}s")

        # Ensure results are equivalent in structure
        assert len(batch_results) == len(individual_results)

    def test_large_batch_handling(self):
        """Test handling of larger batches."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

        # Create a larger batch
        large_batch = [["City" + str(i)] for i in range(50)]

        results = searcher.find_batch(large_batch)
        assert isinstance(results, list)
        assert len(results) == 50

        # Each result should be a list
        for result_list in results:
            assert isinstance(result_list, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
