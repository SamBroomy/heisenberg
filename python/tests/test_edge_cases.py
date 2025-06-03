"""
Edge case and error handling tests for the Python bindings.

These tests focus on boundary conditions, error scenarios, and
ensuring the system behaves gracefully under stress.
"""

import pytest
import heisenberg
import logging
from unittest.mock import patch


class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_none_input_handling(self, searcher):
        """Test handling of None inputs."""
        with pytest.raises(TypeError):
            searcher.find(None)

    def test_empty_input_handling(self, searcher):
        """Test handling of empty inputs."""
        # Empty string should not crash
        results = searcher.find("")
        assert isinstance(results, list)

        # Empty list should not crash
        results = searcher.find([])
        assert isinstance(results, list)

    def test_whitespace_input_handling(self, searcher):
        """Test handling of whitespace-only inputs."""
        whitespace_inputs = [
            " ",
            "\t",
            "\n",
            "   ",
            "\t\n ",
            ["  ", "  "],
            ["\t\t", "\n\n"],
        ]

        for input_data in whitespace_inputs:
            results = searcher.find(input_data)
            assert isinstance(results, list), f"Failed for input: {repr(input_data)}"

    def test_very_long_input_handling(self, searcher):
        """Test handling of extremely long inputs."""
        # Single very long string
        long_string = "a" * 10000
        results = searcher.find(long_string)
        assert isinstance(results, list)

        # Many long strings
        long_terms = ["b" * 1000 for _ in range(100)]
        results = searcher.find(long_terms)
        assert isinstance(results, list)

    def test_unicode_edge_cases(self, searcher):
        """Test various unicode edge cases."""
        unicode_inputs = [
            "🏙️ New York",  # Emoji
            "Москва",  # Cyrillic
            "東京",  # Japanese
            "北京",  # Chinese
            "عمان",  # Arabic
            "Θεσσαλονίκη",  # Greek
            "İstanbul",  # Turkish
            "café",  # French accents
            "Zürich",  # German umlauts
            "Kraków",  # Polish
            "🌍🗺️🏛️",  # Multiple emojis
        ]

        for unicode_input in unicode_inputs:
            try:
                results = searcher.find(unicode_input)
                assert isinstance(results, list), f"Failed for: {unicode_input}"
            except Exception as e:
                pytest.fail(f"Unicode input '{unicode_input}' caused exception: {e}")

    def test_special_characters(self, searcher):
        """Test handling of special characters."""
        special_inputs = [
            "St. John's",
            "O'Connor",
            "Jean-Luc",
            "México D.F.",
            "São Paulo",
            "Côte d'Ivoire",
            "name@domain.com",  # Email-like
            "123-456-7890",  # Phone-like
            "$%^&*()",  # Special symbols
            "<script>alert('xss')</script>",  # XSS-like
            "'; DROP TABLE locations; --",  # SQL injection-like
        ]

        for special_input in special_inputs:
            try:
                results = searcher.find(special_input)
                assert isinstance(results, list), f"Failed for: {special_input}"
            except Exception as e:
                pytest.fail(f"Special input '{special_input}' caused exception: {e}")

    def test_numeric_inputs(self, searcher):
        """Test handling of numeric inputs as strings."""
        numeric_inputs = [
            "123",
            "12345",
            "0",
            "-123",
            "3.14159",
            "1e10",
            "123.456.789",  # Multiple dots
            "12,345",  # Comma separator
        ]

        for numeric_input in numeric_inputs:
            results = searcher.find(numeric_input)
            assert isinstance(results, list), f"Failed for: {numeric_input}"

    def test_mixed_type_lists(self, searcher):
        """Test lists with mixed content types."""
        # This should work as all items get converted to strings
        mixed_list = ["London", "123", "", "  ", "café"]
        results = searcher.find(mixed_list)
        assert isinstance(results, list)


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""

    def test_extreme_limit_values(self):
        """Test extreme values for result limits."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

        # Very small limit
        small_config = heisenberg.SearchConfigBuilder().limit(1).build()
        results = searcher.find("London", small_config)
        assert isinstance(results, list)
        assert len(results) <= 1

        # Very large limit (should not crash)
        large_config = heisenberg.SearchConfigBuilder().limit(10000).build()
        results = searcher.find("London", large_config)
        assert isinstance(results, list)

    def test_invalid_search_options(self):
        """Test SearchOptions with invalid values."""
        # Test all validation cases
        with pytest.raises(ValueError):
            heisenberg.SearchOptions(place_importance_threshold=0)

        with pytest.raises(ValueError):
            heisenberg.SearchOptions(place_importance_threshold=6)

        with pytest.raises(ValueError):
            heisenberg.SearchOptions(limit=0)

        with pytest.raises(ValueError):
            heisenberg.SearchOptions(limit=-1)

        with pytest.raises(ValueError):
            heisenberg.SearchOptions(max_admin_terms=0)

        with pytest.raises(ValueError):
            heisenberg.SearchOptions(search_multiplier=0)

    def test_boundary_values_search_options(self):
        """Test boundary values that should be valid."""
        # Minimum valid values
        min_options = heisenberg.SearchOptions(
            limit=1,
            place_importance_threshold=1,
            max_admin_terms=1,
            search_multiplier=1,
        )
        assert min_options.limit == 1

        # Maximum valid values
        max_options = heisenberg.SearchOptions(
            limit=999999,
            place_importance_threshold=5,
            max_admin_terms=999,
            search_multiplier=999,
        )
        assert max_options.limit == 999999


class TestBatchProcessingEdgeCases:
    """Test edge cases in batch processing."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_empty_batch(self, searcher):
        """Test processing empty batch."""
        results = searcher.find_batch([])
        assert isinstance(results, list)
        assert len(results) == 0

    def test_batch_with_empty_queries(self, searcher):
        """Test batch containing empty queries."""
        batch = [
            [],  # Empty query
            ["London"],  # Normal query
            [""],  # Single empty string
            ["  "],  # Whitespace only
        ]

        results = searcher.find_batch(batch)
        assert isinstance(results, list)
        assert len(results) == 4  # Should have results for all 4 queries

    def test_very_large_batch(self, searcher):
        """Test processing a very large batch."""
        large_batch = [["City" + str(i)] for i in range(1000)]

        try:
            results = searcher.find_batch(large_batch)
            assert isinstance(results, list)
            assert len(results) == 1000
        except Exception as e:
            # If it fails, it should be a resource limitation, not a crash
            assert "memory" in str(e).lower() or "resource" in str(e).lower()

    def test_batch_with_mixed_query_sizes(self, searcher):
        """Test batch with queries of different sizes."""
        mixed_batch = [
            ["A"],  # Single term
            ["New", "York"],  # Two terms
            ["Los", "Angeles", "California"],  # Three terms
            ["São", "Paulo", "SP", "Brazil"],  # Four terms
            [],  # Empty
            ["London", "UK", "England", "Great", "Britain"],  # Five terms
        ]

        results = searcher.find_batch(mixed_batch)
        assert isinstance(results, list)
        assert len(results) == 6


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.fixture
    def searcher(self):
        return heisenberg.LocationSearcher(rebuild_indexes=False)

    def test_malformed_config_handling(self, searcher):
        """Test handling of malformed configurations."""
        # This tests that the system gracefully handles unexpected config states
        try:
            # Create config with extreme values that might cause issues
            extreme_config = heisenberg.SearchConfigBuilder().limit(0).build()
            results = searcher.find("London", extreme_config)
            # Should either work or raise a sensible error
            assert isinstance(results, list)
        except (ValueError, RuntimeError) as e:
            # Acceptable error types for malformed config
            assert len(str(e)) > 0

    def test_memory_pressure_simulation(self, searcher):
        """Test behavior under simulated memory pressure."""
        # This test tries to trigger memory-related edge cases
        try:
            # Large batch with large queries
            memory_stress_batch = [
                ["very long city name " * 100] * 50  # Large query
                for _ in range(100)  # Many queries
            ]

            results = searcher.find_batch(memory_stress_batch)
            assert isinstance(results, list)
        except Exception as e:
            # If it fails, should be a resource error, not a crash
            assert any(
                keyword in str(e).lower()
                for keyword in ["memory", "resource", "limit", "capacity"]
            )

    def test_concurrent_access_safety(self, searcher):
        """Test that the searcher is safe for concurrent access."""
        import threading
        import time

        results = []
        errors = []

        def search_worker(term, worker_id):
            try:
                worker_results = searcher.find(f"City{worker_id}")
                results.append((worker_id, len(worker_results)))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=search_worker, args=(f"Test{i}", i))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Check results
        assert len(errors) == 0, f"Concurrent access caused errors: {errors}"
        assert len(results) == 10, f"Not all threads completed: {len(results)}/10"


class TestConvenienceFunctionEdgeCases:
    """Test edge cases in convenience functions."""

    def test_convenience_function_with_invalid_input(self):
        """Test convenience functions with invalid inputs."""
        # Test find_location with None
        with pytest.raises(TypeError):
            heisenberg.find_location(None)

        # Test find_locations_batch with invalid batch
        with pytest.raises(TypeError):
            heisenberg.find_locations_batch(None)

    def test_convenience_function_empty_inputs(self):
        """Test convenience functions with empty inputs."""
        # Empty string
        results = heisenberg.find_location("")
        assert isinstance(results, list)

        # Empty batch
        batch_results = heisenberg.find_locations_batch([])
        assert isinstance(batch_results, list)
        assert len(batch_results) == 0

    def test_convenience_function_stress(self):
        """Test convenience functions under stress."""
        # Large single query
        large_query = "city " * 1000
        results = heisenberg.find_location(large_query)
        assert isinstance(results, list)

        # Large batch through convenience function
        large_batch = [["city" + str(i)] for i in range(100)]
        batch_results = heisenberg.find_locations_batch(large_batch)
        assert isinstance(batch_results, list)
        assert len(batch_results) == 100


class TestLoggingEdgeCases:
    """Test logging behavior under various conditions."""

    def test_logging_with_unicode(self):
        """Test that logging works with unicode location names."""
        # This test ensures that unicode location names don't break logging
        with patch("heisenberg._internal.logger") as mock_logger:
            searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
            results = searcher.find("Москва")  # Moscow in Russian

            # Should not have caused logging errors
            # (This is more of a smoke test)
            assert isinstance(results, list)

    def test_logging_with_very_long_inputs(self):
        """Test logging behavior with very long inputs."""
        with patch("heisenberg._internal.logger") as mock_logger:
            searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
            long_input = "a" * 10000
            results = searcher.find(long_input)

            # Should not break logging
            assert isinstance(results, list)


class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_multiple_searcher_instances(self):
        """Test creating multiple searcher instances."""
        searchers = []

        try:
            # Create multiple instances
            for i in range(5):
                searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
                searchers.append(searcher)

                # Each should work independently
                results = searcher.find("Test" + str(i))
                assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Multiple searcher instances caused error: {e}")
        finally:
            # Cleanup (Python GC should handle this, but explicit is better)
            del searchers

    def test_searcher_reuse(self):
        """Test that searcher instances can be reused safely."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

        # Use the same instance multiple times
        for i in range(100):
            results = searcher.find(
                f"City{i % 10}"
            )  # Cycle through 10 different queries
            assert isinstance(results, list)

    def test_large_result_handling(self):
        """Test handling of potentially large result sets."""
        searcher = heisenberg.LocationSearcher(rebuild_indexes=False)

        # Query that might return many results
        large_config = heisenberg.SearchConfigBuilder().limit(1000).build()
        results = searcher.find("city", large_config)  # Generic term

        assert isinstance(results, list)
        # Should handle large result sets without crashing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
