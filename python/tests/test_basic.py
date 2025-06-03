import pytest
import heisenberg


def test_location_searcher_creation():
    """Test that LocationSearcher can be created."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    assert searcher is not None


def test_simple_search():
    """Test basic search functionality."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    results = searcher.find("London")
    assert isinstance(results, list)
    # Results should be SearchResult objects
    if results:
        result = results[0]
        assert hasattr(result, "name")
        assert hasattr(result, "score")
        assert hasattr(result, "geoname_id")


def test_quick_search():
    """Test quick search functionality."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    results = searcher.find_quick(["New York"])
    assert isinstance(results, list)


def test_comprehensive_search():
    """Test comprehensive search functionality."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    results = searcher.find_comprehensive(["Paris"])
    assert isinstance(results, list)


def test_important_places_search():
    """Test important places search functionality."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    results = searcher.find_important_places(["Tokyo"])
    assert isinstance(results, list)


def test_batch_search():
    """Test batch search functionality."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    batch_terms = [["London", "UK"], ["Paris", "France"]]
    results = searcher.find_batch(batch_terms)
    assert isinstance(results, list)
    assert len(results) == 2  # Should have results for both queries
    for query_results in results:
        assert isinstance(query_results, list)


def test_search_with_config():
    """Test search with custom configuration."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    config = heisenberg.SearchConfigBuilder.fast().limit(5).build()
    results = searcher.find(["Berlin"], config)
    assert isinstance(results, list)


def test_convenience_functions():
    """Test convenience functions."""
    results = heisenberg.find_location("Madrid")
    assert isinstance(results, list)

    batch_results = heisenberg.find_locations_batch([["Rome"], ["Vienna"]])
    assert isinstance(batch_results, list)
    assert len(batch_results) == 2


def test_search_options_validation():
    """Test SearchOptions validation."""
    # Valid options should work
    options = heisenberg.SearchOptions(
        limit=10,
        place_importance_threshold=3,
    )
    assert options.limit == 10
    assert options.place_importance_threshold == 3

    # Invalid importance threshold should raise error
    with pytest.raises(ValueError):
        heisenberg.SearchOptions(place_importance_threshold=10)  # Should be 1-5


def test_search_config_builder():
    """Test SearchConfigBuilder fluent interface."""
    config = (
        heisenberg.SearchConfigBuilder()
        .limit(15)
        .place_importance(2)
        .admin_search(True)
        .fuzzy_search(False)
        .build()
    )

    assert config.limit == 15
    assert config.place_importance_threshold == 2


def test_search_result_methods():
    """Test SearchResult object methods."""
    searcher = heisenberg.LocationSearcher(rebuild_indexes=False)
    results = searcher.find("London")

    if results:
        result = results[0]

        # Test methods
        hierarchy = result.admin_hierarchy()
        assert isinstance(hierarchy, list)

        full_name = result.full_name()
        assert isinstance(full_name, str)
        assert result.name in full_name

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "name" in result_dict
        assert "geoname_id" in result_dict


@pytest.mark.skip(reason="Requires data download")
def test_direct_rust_api():
    """Test direct Rust API access."""
    # Test the direct Rust binding (exposed as Heisenberg)
    rust_searcher = heisenberg.Heisenberg(rebuild_indexes=False)
    dataframes = rust_searcher.find(["London"])
    assert isinstance(dataframes, list)

    # Test with config
    config = heisenberg.SearchConfigBuilder().limit(5).build()
    dataframes = rust_searcher.find_with_config(["Paris"], config)
    assert isinstance(dataframes, list)
