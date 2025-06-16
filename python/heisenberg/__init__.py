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
    LocationSearcher,
    LocationSearcherBuilderWrapper,
    SearchOptions,
    SearchConfigBuilder,
    SearchResult,
    find_location,
    find_locations_batch,
    DataSource,
    __version__,
)

# Create clean aliases for the main user-facing API
DataSource = DataSource
LocationSearcherBuilder = LocationSearcherBuilderWrapper

__all__ = [
    # Core search functionality
    "LocationSearcher",
    "LocationSearcherBuilder",
    "DataSource",
    # Configuration and results
    "SearchOptions",
    "SearchConfigBuilder",
    "SearchResult",
    # Convenience functions
    "find_location",
    "find_locations_batch",
    # Version information
    "__version__",
]
