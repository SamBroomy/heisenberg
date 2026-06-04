"""
Heisenberg - A powerful location search library.

This module provides comprehensive Python bindings for the Rust-based Heisenberg
location searcher, which can resolve unstructured location inputs into structured
location data based on geonames.

The library is designed for location enrichment, taking input vectors with locations
in descending 'size' order (largest to smallest) and enriching missing or inconsistent
location information using the Geonames dataset.

IMPORTANT INPUT ORDER: For optimal results, provide location terms in descending 'size' order:
- Largest locations first (Country, State/Province)
- Smallest/most specific locations last (County, City, Place)

Example input order: ['Country', 'State', 'County', 'Place']
Example enrichment: ['Florida', 'Lakeland'] → ['United States', 'Florida', 'Polk County', 'Lakeland']
"""

# Import only user-facing wrapper classes
from ._internal import (
    DataSource,
    LocationSearcher,
    LocationSearcherBuilderWrapper,
    SearchConfigBuilder,
    SearchOptions,
    SearchResult,
    __version__,
    find_location,
    find_locations_batch,
)

# Create clean aliases for the main user-facing API
LocationSearcherBuilder = LocationSearcherBuilderWrapper

__all__ = [
    "DataSource",
    # Core search functionality
    "LocationSearcher",
    "LocationSearcherBuilder",
    # Configuration and results
    "SearchConfigBuilder",
    "SearchOptions",
    "SearchResult",
    # Version information
    "__version__",
    # Convenience functions
    "find_location",
    "find_locations_batch",
]
