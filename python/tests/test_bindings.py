#!/usr/bin/env python3
"""
Consolidated tests for Heisenberg Python bindings.

Simple tests to validate that the Python bindings work correctly
and catch any breaking changes in the API.
"""

import pytest
import heisenberg

import os

os.environ["USE_TEST_DATA"] = "true"


class TestBasicFunctionality:
    """Test basic search functionality works."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=True)

    def test_searcher_creation(self):
        """Test that LocationSearcher can be created."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=True)
        assert searcher is not None

    def test_simple_search(self, searcher):
        """Test basic search functionality."""
        results = searcher.find("London")
        assert isinstance(results, list)
        if results:
            result = results[0]
            assert isinstance(result, heisenberg.SearchResult)
            assert hasattr(result, "name")
            assert hasattr(result, "geoname_id")
            assert hasattr(result, "score")
            assert isinstance(result.name, str)
            assert isinstance(result.geoname_id, int)
            assert isinstance(result.score, float)

    def test_empty_input_handling(self, searcher):
        """Test handling of empty inputs."""
        results = searcher.find("")
        assert isinstance(results, list)

        results = searcher.find([])
        assert isinstance(results, list)

    def test_unicode_input(self, searcher):
        """Test unicode inputs work."""
        results = searcher.find("北京")  # Beijing in Chinese
        assert isinstance(results, list)


class TestSearchMethods:
    """Test different search methods work."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_quick_search(self, searcher):
        """Test quick search method."""
        results = searcher.find_quick("Paris")
        assert isinstance(results, list)

    def test_comprehensive_search(self, searcher):
        """Test comprehensive search method."""
        results = searcher.find_comprehensive("Tokyo")
        assert isinstance(results, list)

    def test_important_places_search(self, searcher):
        """Test important places search method."""
        results = searcher.find_important_places("Berlin")
        assert isinstance(results, list)

    def test_batch_search(self, searcher):
        """Test batch search functionality."""
        queries = [["London"], ["Paris"], ["Tokyo"]]
        results = searcher.find_batch(queries)
        assert isinstance(results, list)
        assert len(results) == 3
        for query_results in results:
            assert isinstance(query_results, list)


class TestConfiguration:
    """Test configuration and options work."""

    def test_search_options_validation(self):
        """Test SearchOptions validation."""
        # Valid options should work
        options = heisenberg.SearchOptions(limit=10, place_importance_threshold=3)
        assert options.limit == 10
        assert options.place_importance_threshold == 3

        # Invalid options should raise errors
        with pytest.raises(ValueError):
            heisenberg.SearchOptions(place_importance_threshold=10)  # Out of range

        with pytest.raises(ValueError):
            heisenberg.SearchOptions(limit=0)  # Invalid limit

    def test_config_builder(self):
        """Test SearchConfigBuilder fluent interface."""
        config = (
            heisenberg.SearchConfigBuilder()
            .limit(5)
            .place_importance(2)
            .admin_search(True)
            .fuzzy_search(False)
            .build()
        )
        assert config.limit == 5
        assert config.place_importance_threshold == 2

    def test_preset_configurations(self):
        """Test preset configurations."""
        fast_config = heisenberg.SearchConfigBuilder.fast().build()
        comprehensive_config = heisenberg.SearchConfigBuilder.comprehensive().build()
        quality_config = heisenberg.SearchConfigBuilder.quality_places().build()

        assert fast_config.limit <= comprehensive_config.limit
        assert isinstance(fast_config, heisenberg.SearchOptions)
        assert isinstance(comprehensive_config, heisenberg.SearchOptions)
        assert isinstance(quality_config, heisenberg.SearchOptions)

    def test_search_with_config(self):
        """Test search with custom configuration."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
        config = heisenberg.SearchConfigBuilder().limit(5).build()
        results = searcher.find("Madrid", config)
        assert isinstance(results, list)
        assert len(results) <= 5


class TestSearchResult:
    """Test SearchResult object functionality."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_search_result_methods(self, searcher):
        """Test SearchResult object methods."""
        results = searcher.find("New York")
        if results:
            result = results[0]

            # Test admin hierarchy
            hierarchy = result.admin_hierarchy()
            assert isinstance(hierarchy, list)

            # Test full name
            full_name = result.full_name()
            assert isinstance(full_name, str)
            assert result.name in full_name

            # Test dictionary conversion
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "name" in result_dict
            assert "geoname_id" in result_dict
            assert result_dict["name"] == result.name

    def test_search_result_attributes(self, searcher):
        """Test SearchResult has expected attributes."""
        results = searcher.find("London")
        if results:
            result = results[0]

            # Required attributes
            assert hasattr(result, "geoname_id")
            assert hasattr(result, "name")
            assert hasattr(result, "feature_code")
            assert hasattr(result, "score")

            # Optional attributes (should exist even if None)
            assert hasattr(result, "latitude")
            assert hasattr(result, "longitude")
            assert hasattr(result, "population")


class TestConvenienceFunctions:
    """Test convenience functions work."""

    def test_find_location_function(self):
        """Test find_location convenience function."""
        results = heisenberg.find_location("Rome")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], heisenberg.SearchResult)

    def test_find_locations_batch_function(self):
        """Test find_locations_batch convenience function."""
        queries = [["Vienna"], ["Prague"]]
        results = heisenberg.find_locations_batch(queries)
        assert isinstance(results, list)
        assert len(results) == 2
        for query_results in results:
            assert isinstance(query_results, list)


class TestRustAPIAccess:
    """Test direct Rust API access works."""

    def test_rust_searcher_creation(self):
        """Test creating the direct Rust searcher."""
        rust_searcher = heisenberg.Heisenberg(overwrite_indexes=False)
        assert rust_searcher is not None

    def test_rust_search_methods(self):
        """Test direct Rust search methods."""
        rust_searcher = heisenberg.Heisenberg(overwrite_indexes=False)

        # Test basic search
        results = rust_searcher.search(["London"])
        assert isinstance(results, list)

        # Test search_with_config
        config = heisenberg.RustSearchConfigBuilder().limit(5).build()
        config_results = rust_searcher.search_with_config(["Paris"], config)
        assert isinstance(config_results, list)

        # Test search_bulk
        bulk_results = rust_searcher.search_bulk([["Tokyo"], ["Berlin"]])
        assert isinstance(bulk_results, list)
        assert len(bulk_results) == 2

        # Test search_bulk_with_config
        bulk_config_results = rust_searcher.search_bulk_with_config(
            [["Madrid"], ["Rome"]], config
        )
        assert isinstance(bulk_config_results, list)
        assert len(bulk_config_results) == 2

    def test_rust_resolve_methods(self):
        """Test direct Rust resolve methods."""
        rust_searcher = heisenberg.Heisenberg(overwrite_indexes=False)

        # Test resolve_location
        resolved = rust_searcher.resolve_location(["Paris"])
        assert isinstance(resolved, list)

        # Test resolve_location_with_config
        config = heisenberg.RustSearchConfigBuilder().limit(3).build()
        resolved_config = rust_searcher.resolve_location_with_config(["London"], config)
        assert isinstance(resolved_config, list)

        # Test resolve_location_batch
        batch_resolved = rust_searcher.resolve_location_batch([["Berlin"], ["Madrid"]])
        assert isinstance(batch_resolved, list)
        assert len(batch_resolved) == 2

        # Test resolve_location_batch_with_config
        batch_config_resolved = rust_searcher.resolve_location_batch_with_config(
            [["Vienna"], ["Prague"]], config
        )
        assert isinstance(batch_config_resolved, list)
        assert len(batch_config_resolved) == 2

    def test_rust_admin_and_place_search(self):
        """Test direct Rust admin and place search methods."""
        rust_searcher = heisenberg.Heisenberg(overwrite_indexes=False)

        # Test admin_search
        admin_results = rust_searcher.admin_search("France", [0], None)
        # admin_search can return None or a DataFrame
        assert admin_results is None or hasattr(admin_results, "shape")

        # Test place_search
        place_results = rust_searcher.place_search("Paris", None)
        # place_search can return None or a DataFrame
        assert place_results is None or hasattr(place_results, "shape")


class TestSearchOptionsValidation:
    """Test comprehensive SearchOptions validation."""

    def test_search_options_boundary_values(self):
        """Test SearchOptions with boundary values."""
        # Test minimum valid values
        min_options = heisenberg.SearchOptions(
            limit=1,
            place_importance_threshold=1,
            max_admin_terms=1,
            search_multiplier=1,
        )
        assert min_options.limit == 1
        assert min_options.place_importance_threshold == 1

        # Test maximum reasonable values
        max_options = heisenberg.SearchOptions(
            limit=1000,
            place_importance_threshold=5,
            max_admin_terms=100,
            search_multiplier=10,
        )
        assert max_options.limit == 1000
        assert max_options.place_importance_threshold == 5

    def test_search_options_validation_errors(self):
        """Test all SearchOptions validation error cases."""
        # Test place_importance_threshold validation
        with pytest.raises(
            ValueError, match="place_importance_threshold must be between 1 and 5"
        ):
            heisenberg.SearchOptions(place_importance_threshold=0)

        with pytest.raises(
            ValueError, match="place_importance_threshold must be between 1 and 5"
        ):
            heisenberg.SearchOptions(place_importance_threshold=6)

        # Test limit validation
        with pytest.raises(ValueError, match="limit must be positive"):
            heisenberg.SearchOptions(limit=0)

        with pytest.raises(ValueError, match="limit must be positive"):
            heisenberg.SearchOptions(limit=-1)

    def test_search_options_weight_validation(self):
        """Test weight validation warnings."""
        import warnings

        # Test admin weights that don't sum to 1.0
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            options = heisenberg.SearchOptions(
                admin_text_weight=0.1,
                admin_population_weight=0.1,
                admin_parent_weight=0.1,
                admin_feature_weight=0.1,
            )
            # Should create successfully but may log warning
            assert options is not None

    def test_search_options_location_bias(self):
        """Test location bias functionality."""
        options = heisenberg.SearchOptions(
            center_latitude=40.7128,
            center_longitude=-74.0060,
        )
        assert options.center_latitude == 40.7128
        assert options.center_longitude == -74.0060


class TestRustConfigBuilder:
    """Test RustSearchConfigBuilder functionality."""

    def test_rust_config_builder_methods(self):
        """Test all RustSearchConfigBuilder methods."""
        config = (
            heisenberg.RustSearchConfigBuilder()
            .limit(10)
            .place_importance_threshold(3)
            .proactive_admin_search(False)
            .max_admin_terms(5)
            .include_all_columns()
            .text_search(True, 2)
            .build()
        )

        assert isinstance(config, heisenberg.RustSearchConfig)

    def test_rust_config_builder_presets(self):
        """Test RustSearchConfigBuilder preset methods."""
        fast_config = heisenberg.RustSearchConfigBuilder.fast().build()
        comprehensive_config = (
            heisenberg.RustSearchConfigBuilder.comprehensive().build()
        )
        quality_config = heisenberg.RustSearchConfigBuilder.quality_places().build()

        assert isinstance(fast_config, heisenberg.RustSearchConfig)
        assert isinstance(comprehensive_config, heisenberg.RustSearchConfig)
        assert isinstance(quality_config, heisenberg.RustSearchConfig)


class TestEntryTypes:
    """Test BasicEntry and GenericEntry functionality."""

    def test_basic_entry_functionality(self):
        """Test BasicEntry methods and attributes."""
        # We can't easily create these directly, but we can test they exist
        assert hasattr(heisenberg, "BasicEntry")

        # Test that BasicEntry has expected attributes
        basic_entry_attrs = ["geoname_id", "name"]
        for attr in basic_entry_attrs:
            assert hasattr(heisenberg.BasicEntry, attr)

    def test_generic_entry_functionality(self):
        """Test GenericEntry methods and attributes."""
        assert hasattr(heisenberg, "GenericEntry")

        # Test that GenericEntry has expected attributes
        generic_entry_attrs = [
            "geoname_id",
            "name",
            "admin0_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "feature_code",
            "latitude",
            "longitude",
            "population",
        ]
        for attr in generic_entry_attrs:
            assert hasattr(heisenberg.GenericEntry, attr)

    def test_location_context_types(self):
        """Test LocationContext types exist."""
        assert hasattr(heisenberg, "LocationContextBasic")
        assert hasattr(heisenberg, "LocationContextGeneric")

    def test_resolved_search_result_types(self):
        """Test ResolvedSearchResult types exist."""
        assert hasattr(heisenberg, "ResolvedBasicSearchResult")
        assert hasattr(heisenberg, "ResolvedGenericSearchResult")


class TestResolvedSearchResultProperties:
    """Test ResolvedSearchResult properties and methods."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_resolved_search_result_properties(self, searcher):
        """Test ResolvedSearchResult context and score properties."""
        # Get raw resolved results to test properties
        rust_searcher = heisenberg.Heisenberg(overwrite_indexes=False)
        resolved_results = rust_searcher.resolve_location(["London"])

        if resolved_results:
            result = resolved_results[0]

            # Test that result has context and score properties
            assert hasattr(result, "context")
            assert hasattr(result, "score")

            # Test context access
            context = result.context
            assert context is not None

            # Test score access
            score = result.score
            assert isinstance(score, float)
            assert score >= 0.0

    def test_location_context_admin_properties(self, searcher):
        """Test LocationContext admin level properties."""
        rust_searcher = heisenberg.Heisenberg(overwrite_indexes=False)
        resolved_results = rust_searcher.resolve_location(["London"])

        if resolved_results:
            result = resolved_results[0]
            context = result.context

            # Test that context has admin properties
            admin_attrs = ["admin0", "admin1", "admin2", "admin3", "admin4", "place"]
            for attr in admin_attrs:
                assert hasattr(context, attr)

    def test_resolved_result_methods(self, searcher):
        """Test ResolvedSearchResult methods."""
        rust_searcher = heisenberg.Heisenberg(overwrite_indexes=False)
        resolved_results = rust_searcher.resolve_location(["Paris"])

        if resolved_results:
            result = resolved_results[0]

            # Test simple method
            if hasattr(result, "simple"):
                simple_names = result.simple()
                assert isinstance(simple_names, list)

            # Test full method
            if hasattr(result, "full"):
                full_names = result.full()
                assert isinstance(full_names, list)


class TestAdvancedSearchFeatures:
    """Test advanced search configuration features."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_config_builder_chaining(self):
        """Test that config builder methods can be chained."""
        config = (
            heisenberg.SearchConfigBuilder()
            .limit(15)
            .place_importance(3)
            .admin_search(True)
            .fuzzy_search(False)
            .location_bias(51.5074, -0.1278)  # London coordinates
            .admin_weights(text=0.5, population=0.3, parent=0.1, feature=0.1)
            .place_weights(
                text=0.6, importance=0.2, feature=0.1, parent=0.05, distance=0.05
            )
            .build()
        )

        assert config.limit == 15
        assert config.place_importance_threshold == 3
        assert config.center_latitude == 51.5074
        assert config.center_longitude == -0.1278

    def test_search_with_location_bias(self, searcher):
        """Test search with location bias."""
        # Test search near London
        london_config = (
            heisenberg.SearchConfigBuilder()
            .limit(5)
            .location_bias(51.5074, -0.1278)
            .build()
        )

        results = searcher.find("Cambridge", london_config)
        assert isinstance(results, list)

        # Test search near Boston
        boston_config = (
            heisenberg.SearchConfigBuilder()
            .limit(5)
            .location_bias(42.3601, -71.0589)
            .build()
        )

        results = searcher.find("Cambridge", boston_config)
        assert isinstance(results, list)

    def test_search_with_custom_weights(self, searcher):
        """Test search with custom scoring weights."""
        custom_config = (
            heisenberg.SearchConfigBuilder()
            .limit(10)
            .admin_weights(text=0.6, population=0.2, parent=0.1, feature=0.1)
            .place_weights(
                text=0.5, importance=0.3, feature=0.1, parent=0.05, distance=0.05
            )
            .build()
        )

        results = searcher.find("New York", custom_config)
        assert isinstance(results, list)


class TestLowLevelAPIAccess:
    """Test low-level API methods in LocationSearcher."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_raw_search_methods(self, searcher):
        """Test raw search methods that return DataFrames."""
        # Test search_raw
        if hasattr(searcher, "search_raw"):
            raw_results = searcher.search_raw(["London"])
            assert isinstance(raw_results, list)

        # Test search_raw_with_config
        if hasattr(searcher, "search_raw_with_config"):
            config = heisenberg.SearchConfigBuilder().limit(5).build()
            raw_config_results = searcher.search_raw_with_config(["Paris"], config)
            assert isinstance(raw_config_results, list)

    def test_admin_and_place_search_methods(self, searcher):
        """Test admin_search and place_search methods."""
        # Test admin_search
        if hasattr(searcher, "admin_search"):
            admin_results = searcher.admin_search("France", [0])
            # Can return None or DataFrame
            assert admin_results is None or hasattr(admin_results, "shape")

        # Test place_search
        if hasattr(searcher, "place_search"):
            place_results = searcher.place_search("Paris")
            # Can return None or DataFrame
            assert place_results is None or hasattr(place_results, "shape")

    def test_resolve_methods(self, searcher):
        """Test resolve_location methods."""
        # Test resolve_location
        if hasattr(searcher, "resolve_location"):
            resolved = searcher.resolve_location(["Berlin"])
            assert isinstance(resolved, list)

        # Test resolve_location_batch
        if hasattr(searcher, "resolve_location_batch"):
            batch_resolved = searcher.resolve_location_batch([["Madrid"], ["Vienna"]])
            assert isinstance(batch_resolved, list)
            assert len(batch_resolved) == 2


class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling and edge cases."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_malformed_inputs(self, searcher):
        """Test handling of malformed inputs."""
        malformed_inputs = [
            None,  # This might raise TypeError, which is acceptable
            123,  # This might raise TypeError, which is acceptable
            {"invalid": "input"},  # This might raise TypeError, which is acceptable
        ]

        for malformed_input in malformed_inputs:
            try:
                # These should either work or raise TypeError/ValueError
                results = searcher.find(malformed_input)
                assert isinstance(results, list)  # If it works, should return list
            except (TypeError, ValueError):
                # These are acceptable error types for malformed input
                pass

    def test_extremely_long_inputs(self, searcher):
        """Test handling of extremely long inputs."""
        # Very long string
        very_long_string = "city " * 10000
        results = searcher.find(very_long_string)
        assert isinstance(results, list)

        # Very long list
        very_long_list = ["term"] * 1000
        results = searcher.find(very_long_list)
        assert isinstance(results, list)

    def test_batch_with_empty_and_invalid_queries(self, searcher):
        """Test batch processing with mixed valid/invalid queries."""
        mixed_batch = [
            ["London"],  # Valid
            [],  # Empty
            [""],  # Empty string
            ["Valid", "Query"],  # Valid multi-term
            ["   "],  # Whitespace only
        ]

        results = searcher.find_batch(mixed_batch)
        assert isinstance(results, list)
        assert len(results) == 5  # Should have results for all queries

    def test_configuration_validation_edge_cases(self):
        """Test edge cases in configuration validation."""
        # Test floating point precision in weights
        try:
            options = heisenberg.SearchOptions(
                admin_text_weight=0.33333333,
                admin_population_weight=0.33333333,
                admin_parent_weight=0.33333334,  # Slightly different due to precision
                admin_feature_weight=0.0,
            )
            assert options is not None
        except ValueError:
            # Weight validation errors are acceptable
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
