import pytest
import heisenberg


def test_heisenberg_creation():
    """Test that Heisenberg can be created."""
    hb = heisenberg.Heisenberg(overwrite_indexes=False)
    assert hb is not None


def test_search():
    """Test basic search functionality."""
    hb = heisenberg.Heisenberg(overwrite_indexes=False)
    results = hb.search(["London"])
    assert isinstance(results, list)


def test_admin_search():
    """Test admin search."""
    hb = heisenberg.Heisenberg(overwrite_indexes=False)
    result = hb.admin_search("United Kingdom", levels=[0])
    # Result can be None or a DataFrame
    assert result is None or hasattr(result, "height")


@pytest.mark.skip(reason="Requires data download")
def test_location_resolution():
    """Test location resolution."""
    hb = heisenberg.Heisenberg(overwrite_indexes=False)
    resolved = hb.resolve_location(["London", "UK"])
    assert isinstance(resolved, list)
