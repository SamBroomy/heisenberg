"""
Heisenberg - A powerful location search library.

This module provides comprehensive Python bindings for the Rust-based Heisenberg
location searcher, which can resolve unstructured location inputs into structured
location data based on geonames.
"""

# Main user-facing API from _internal module
from ._internal import (
    LocationSearcher,
    SearchOptions,
    SearchConfigBuilder,
    SearchResult,
    find_location,
    find_locations_batch,
    RustSearchConfig,
    RustSearchConfigBuilder,
    RustLocationSearcher,
    BasicEntry,
    GenericEntry,
    LocationContextBasic,
    LocationContextGeneric,
    ResolvedBasicSearchResult,
    ResolvedGenericSearchResult,
    __version__,
)

# Create aliases for compatibility and convenience
SearchConfig = RustSearchConfig
Heisenberg = RustLocationSearcher  # Alias for direct Rust API access
HeisenbergCore = RustLocationSearcher  # Alternative alias

__all__ = [
    # Main user-facing API
    "LocationSearcher",
    "SearchOptions",
    "SearchConfigBuilder",
    "SearchResult",
    "find_location",
    "find_locations_batch",
    # Alternative names for compatibility
    "Heisenberg",  # Alias for RustLocationSearcher
    "SearchConfig",  # Alias for RustSearchConfig
    "HeisenbergCore",  # Alternative alias for RustLocationSearcher
    # Lower-level Rust API
    "RustLocationSearcher",
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
