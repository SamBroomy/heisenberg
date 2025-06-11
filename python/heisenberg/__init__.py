"""
Heisenberg - A powerful location search library.

This module provides comprehensive Python bindings for the Rust-based Heisenberg
location searcher, which can resolve unstructured location inputs into structured
location data based on geonames.
"""

# Main user-facing API from _internal module
from ._internal import (
    LocationSearcher,
    LocationSearcherBuilderWrapper,
    SearchOptions,
    SearchConfigBuilder,
    SearchResult,
    find_location,
    find_locations_batch,
    DataSourceWrapper,
    RustSearchConfig,
    RustSearchConfigBuilder,
    RustLocationSearcher,
    RustLocationSearcherBuilder,
    BasicEntry,
    GenericEntry,
    LocationContextBasic,
    LocationContextGeneric,
    ResolvedBasicSearchResult,
    ResolvedGenericSearchResult,
    DataSource,
    __version__,
)

# Create aliases for compatibility and convenience
SearchConfig = RustSearchConfig
Heisenberg = RustLocationSearcher  # Alias for direct Rust API access
HeisenbergCore = RustLocationSearcher  # Alternative alias

# Create aliases for the main classes users should use
DataSource = DataSourceWrapper
LocationSearcherBuilder = LocationSearcherBuilderWrapper

__all__ = [
    # Main user-facing API
    "LocationSearcher",
    "LocationSearcherBuilder",
    "SearchOptions",
    "SearchConfigBuilder",
    "SearchResult",
    "find_location",
    "find_locations_batch",
    # Data sources
    "DataSource",  # This is the main one users should use
    "DataSourceWrapper",
    # Alternative names for compatibility
    "Heisenberg",  # Alias for RustLocationSearcher
    "SearchConfig",  # Alias for RustSearchConfig
    "HeisenbergCore",  # Alternative alias for RustLocationSearcher
    # Lower-level Rust API (for advanced users)
    "RustLocationSearcher",
    "RustLocationSearcherBuilder",
    "LocationSearcherBuilderWrapper",
    "RustSearchConfig",
    "RustSearchConfigBuilder",
    # Entry types
    "BasicEntry",
    "GenericEntry",
    "LocationContextBasic",
    "LocationContextGeneric",
    "ResolvedBasicSearchResult",
    "ResolvedGenericSearchResult",
    # Version
    "__version__",
]
