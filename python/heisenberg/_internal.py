"""
Internal interface to the Rust Heisenberg library.

This module provides the main interface to the Rust LocationSearcher and related
functionality, serving as the bridge between the Rust implementation and the
Python user-facing API.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING, Union
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    import polars as pl


# Import the Rust bindings
from .heisenberg import (
    LocationSearcher as RustLocationSearcher,
    SearchConfig as RustSearchConfig,
    SearchConfigBuilder as RustSearchConfigBuilder,
    BasicEntry,
    GenericEntry,
    LocationContextBasic,
    LocationContextGeneric,
    ResolvedBasicSearchResult,
    ResolvedGenericSearchResult,
    __version__,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchOptions:
    """Configuration options for location searches.

    This class provides a pythonic way to configure search parameters with
    sensible defaults and validation.
    """

    # Basic search parameters
    limit: int = 20
    include_all_columns: bool = False
    max_admin_terms: int = 5
    place_importance_threshold: int = 5  # 1=most important, 5=least important

    # Advanced search behavior
    proactive_admin_search: bool = True
    fuzzy_search: bool = True
    search_multiplier: int = 3  # How many more results to fetch for ranking

    # Scoring weights for administrative entities
    admin_text_weight: float = 0.4
    admin_population_weight: float = 0.25
    admin_parent_weight: float = 0.20
    admin_feature_weight: float = 0.15

    # Scoring weights for places
    place_text_weight: float = 0.4
    place_importance_weight: float = 0.20
    place_feature_weight: float = 0.15
    place_parent_weight: float = 0.15
    place_distance_weight: float = 0.05

    # Location bias (for distance scoring)
    center_latitude: Optional[float] = None
    center_longitude: Optional[float] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if not 1 <= self.place_importance_threshold <= 5:
            raise ValueError("place_importance_threshold must be between 1 and 5")

        # Validate weights sum approximately to 1.0 for each category
        admin_weight_sum = (
            self.admin_text_weight
            + self.admin_population_weight
            + self.admin_parent_weight
            + self.admin_feature_weight
        )
        if not 0.95 <= admin_weight_sum <= 1.05:
            logger.warning(
                f"Admin weights sum to {admin_weight_sum:.3f}, should be close to 1.0"
            )

        place_weight_sum = (
            self.place_text_weight
            + self.place_importance_weight
            + self.place_feature_weight
            + self.place_parent_weight
            + self.place_distance_weight
        )
        if not 0.95 <= place_weight_sum <= 1.05:
            logger.warning(
                f"Place weights sum to {place_weight_sum:.3f}, should be close to 1.0"
            )

    def to_rust_config(self) -> RustSearchConfig:
        """Convert to a Rust SearchConfig object."""
        builder = RustSearchConfigBuilder()
        builder = builder.limit(self.limit)
        builder = builder.place_importance_threshold(self.place_importance_threshold)
        if self.include_all_columns:
            builder = builder.include_all_columns()
        builder = builder.max_admin_terms(self.max_admin_terms)
        builder = builder.proactive_admin_search(self.proactive_admin_search)
        builder = builder.text_search(self.fuzzy_search, self.search_multiplier)
        return builder.build()


@dataclass
class SearchResult:
    """A search result with location information and confidence score."""

    # Core location information
    geoname_id: int
    name: str
    feature_code: str

    # Administrative hierarchy
    admin0_name: Optional[str] = None
    admin1_name: Optional[str] = None
    admin2_name: Optional[str] = None
    admin3_name: Optional[str] = None
    admin4_name: Optional[str] = None

    # Geographic information
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    population: Optional[int] = None

    # Search metadata
    score: float = 0.0
    admin_level: int = 0

    @classmethod
    def from_polars_row(cls, row: Dict[str, Any]) -> "SearchResult":
        """Create SearchResult from a Polars DataFrame row."""
        return cls(
            geoname_id=row.get("geoname_id", 0),
            name=row.get("name", ""),
            feature_code=row.get("feature_code", ""),
            admin0_name=row.get("admin0_name"),
            admin1_name=row.get("admin1_name"),
            admin2_name=row.get("admin2_name"),
            admin3_name=row.get("admin3_name"),
            admin4_name=row.get("admin4_name"),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
            population=row.get("population"),
            score=row.get("score", 0.0),
            admin_level=row.get("admin_level", 0),
        )

    @classmethod
    def from_resolved_result(
        cls, resolved: ResolvedGenericSearchResult
    ) -> "SearchResult":
        """Create SearchResult from a ResolvedGenericSearchResult."""
        # Get context??
        context = resolved.get_context()  # Error in find: 'builtins.ResolvedGenericSearchResult' object has no attribute 'get_context'

        # Try to get the best entry (place first, then highest admin level)
        entry = None
        admin_level = 0

        if context.get_place():
            entry = context.get_place()
            admin_level = 0
        else:
            # Find the highest admin level that exists
            for level in range(4, -1, -1):
                admin_entry = getattr(context, f"get_admin{level}")()
                if admin_entry:
                    entry = admin_entry
                    admin_level = level
                    break

        if not entry:
            # Fallback to empty values
            return cls(
                geoname_id=0,
                name="Unknown",
                feature_code="",
                score=resolved.get_score(),
                admin_level=admin_level,
            )

        # Build admin hierarchy
        admin_names = {}
        for i in range(5):
            admin_entry = getattr(context, f"get_admin{i}")()
            if admin_entry:
                admin_names[f"admin{i}_name"] = admin_entry.name

        return cls(
            geoname_id=entry.geoname_id,
            name=entry.name,
            feature_code=entry.feature_code,
            admin0_name=admin_names.get("admin0_name"),
            admin1_name=admin_names.get("admin1_name"),
            admin2_name=admin_names.get("admin2_name"),
            admin3_name=admin_names.get("admin3_name"),
            admin4_name=admin_names.get("admin4_name"),
            latitude=getattr(entry, "latitude", None),
            longitude=getattr(entry, "longitude", None),
            population=getattr(entry, "population", None),
            score=resolved.get_score(),
            admin_level=admin_level,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "geoname_id": self.geoname_id,
            "name": self.name,
            "feature_code": self.feature_code,
            "admin0_name": self.admin0_name,
            "admin1_name": self.admin1_name,
            "admin2_name": self.admin2_name,
            "admin3_name": self.admin3_name,
            "admin4_name": self.admin4_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "population": self.population,
            "score": self.score,
            "admin_level": self.admin_level,
        }

    def admin_hierarchy(self) -> List[str]:
        """Get the administrative hierarchy as a list of names."""
        hierarchy = []
        for admin_name in [
            self.admin0_name,
            self.admin1_name,
            self.admin2_name,
            self.admin3_name,
            self.admin4_name,
        ]:
            if admin_name:
                hierarchy.append(admin_name)
        return hierarchy

    def full_name(self) -> str:
        """Get the full name including administrative hierarchy."""
        parts = [self.name] + self.admin_hierarchy()
        return ", ".join(parts)


class SearchConfigBuilder:
    """Builder for creating search configurations with fluent interface."""

    def __init__(self):
        self.options = SearchOptions()

    def limit(self, limit: int) -> "SearchConfigBuilder":
        """Set the maximum number of results to return."""
        self.options.limit = limit
        return self

    def place_importance(self, threshold: int) -> "SearchConfigBuilder":
        """Set the place importance threshold (1=most important, 5=least)."""
        self.options.place_importance_threshold = threshold
        return self

    def admin_search(self, enabled: bool) -> "SearchConfigBuilder":
        """Enable or disable proactive administrative search."""
        self.options.proactive_admin_search = enabled
        return self

    def fuzzy_search(self, enabled: bool) -> "SearchConfigBuilder":
        """Enable or disable fuzzy string matching."""
        self.options.fuzzy_search = enabled
        return self

    def location_bias(self, latitude: float, longitude: float) -> "SearchConfigBuilder":
        """Set location bias for distance-based scoring."""
        self.options.center_latitude = latitude
        self.options.center_longitude = longitude
        return self

    def admin_weights(
        self,
        text: float = 0.4,
        population: float = 0.25,
        parent: float = 0.20,
        feature: float = 0.15,
    ) -> "SearchConfigBuilder":
        """Set scoring weights for administrative entities."""
        self.options.admin_text_weight = text
        self.options.admin_population_weight = population
        self.options.admin_parent_weight = parent
        self.options.admin_feature_weight = feature
        return self

    def place_weights(
        self,
        text: float = 0.4,
        importance: float = 0.20,
        feature: float = 0.15,
        parent: float = 0.15,
        distance: float = 0.05,
    ) -> "SearchConfigBuilder":
        """Set scoring weights for places."""
        self.options.place_text_weight = text
        self.options.place_importance_weight = importance
        self.options.place_feature_weight = feature
        self.options.place_parent_weight = parent
        self.options.place_distance_weight = distance
        return self

    def build(self) -> SearchOptions:
        """Build the final search configuration."""
        return self.options

    @classmethod
    def fast(cls) -> "SearchConfigBuilder":
        """Create a configuration optimized for speed."""
        return (
            cls().limit(10).place_importance(3).admin_search(False).fuzzy_search(False)
        )

    @classmethod
    def comprehensive(cls) -> "SearchConfigBuilder":
        """Create a configuration optimized for comprehensive results."""
        return cls().limit(50).place_importance(5).admin_search(True).fuzzy_search(True)

    @classmethod
    def quality_places(cls) -> "SearchConfigBuilder":
        """Create a configuration for high-quality places only."""
        return cls().limit(20).place_importance(2).admin_search(True).fuzzy_search(True)


class LocationSearcher:
    """High-level Python interface to the Rust location searcher.

    This class provides a convenient Python API over the raw Rust bindings,
    converting between Rust DataFrames and Python SearchResult objects,
    and providing convenient search methods.
    """

    def __init__(self, rebuild_indexes: bool = False):
        """Initialize the location searcher.

        Args:
            rebuild_indexes: Whether to rebuild the search indexes from scratch.
        """
        self._rust_searcher = RustLocationSearcher(rebuild_indexes)

    def find(
        self, query: Union[str, List[str]], config: Optional[SearchOptions] = None
    ) -> List[SearchResult]:
        """Find locations matching the query.

        Args:
            query: Search query string or list of query terms.
            config: Optional search configuration.

        Returns:
            List of SearchResult objects.
        """
        if isinstance(query, str):
            query = [query]

        try:
            if config:
                resolved_results = self._rust_searcher.resolve_location_with_config(
                    query, config.to_rust_config()
                )
            else:
                resolved_results = self._rust_searcher.resolve_location(query)

            return [
                SearchResult.from_resolved_result(result) for result in resolved_results
            ]
        except Exception as e:
            logger.error(f"Error in find: {e}")
            return []

    def find_quick(self, query: Union[str, List[str]]) -> List[SearchResult]:
        """Quick search with fast configuration.

        Args:
            query: Search query string or list of query terms.

        Returns:
            List of SearchResult objects.
        """
        config = SearchConfigBuilder.fast().build()
        return self.find(query, config)

    def find_comprehensive(self, query: Union[str, List[str]]) -> List[SearchResult]:
        """Comprehensive search with detailed configuration.

        Args:
            query: Search query string or list of query terms.

        Returns:
            List of SearchResult objects.
        """
        config = SearchConfigBuilder.comprehensive().build()
        return self.find(query, config)

    def find_important_places(self, query: Union[str, List[str]]) -> List[SearchResult]:
        """Search for important places only.

        Args:
            query: Search query string or list of query terms.

        Returns:
            List of SearchResult objects for important places.
        """
        config = SearchConfigBuilder.quality_places().build()
        return self.find(query, config)

    def find_batch(
        self, queries: List[List[str]], config: Optional[SearchOptions] = None
    ) -> List[List[SearchResult]]:
        """Find locations for multiple queries in batch.

        Args:
            queries: List of query lists.
            config: Optional search configuration.

        Returns:
            List of lists of SearchResult objects.
        """
        try:
            if config:
                resolved_batches = (
                    self._rust_searcher.resolve_location_batch_with_config(
                        queries, config.to_rust_config()
                    )
                )
            else:
                resolved_batches = self._rust_searcher.resolve_location_batch(queries)

            return [
                [SearchResult.from_resolved_result(result) for result in batch]
                for batch in resolved_batches
            ]
        except Exception as e:
            logger.error(f"Error in find_batch: {e}")
            return [[] for _ in queries]

    # Low-level access to Rust methods for advanced users
    def admin_search(
        self,
        term: str,
        levels: List[int],
        previous_result: Optional["pl.DataFrame"] = None,
    ) -> Optional["pl.DataFrame"]:
        """Search for administrative entities at specific levels.

        Args:
            term: Search term.
            levels: Administrative levels to search (0-4).
            previous_result: Optional previous search result to filter by.

        Returns:
            Polars DataFrame with search results, or None if no results.
        """
        return self._rust_searcher.admin_search(term, levels, previous_result)

    def place_search(
        self, term: str, previous_result: Optional["pl.DataFrame"] = None
    ) -> Optional["pl.DataFrame"]:
        """Search for places.

        Args:
            term: Search term.
            previous_result: Optional previous search result to filter by.

        Returns:
            Polars DataFrame with search results, or None if no results.
        """
        return self._rust_searcher.place_search(term, previous_result)

    def search_raw(self, input_terms: List[str]) -> List["pl.DataFrame"]:
        """Low-level search returning raw DataFrames.

        Args:
            input_terms: List of search terms.

        Returns:
            List of Polars DataFrames.
        """
        return self._rust_searcher.search(input_terms)

    def search_raw_with_config(
        self, input_terms: List[str], config: SearchOptions
    ) -> List["pl.DataFrame"]:
        """Low-level search with configuration returning raw DataFrames.

        Args:
            input_terms: List of search terms.
            config: Search configuration.

        Returns:
            List of Polars DataFrames.
        """
        return self._rust_searcher.search_with_config(
            input_terms, config.to_rust_config()
        )

    def resolve_location(
        self, input_terms: List[str]
    ) -> List[ResolvedGenericSearchResult]:
        """Resolve location search results.

        Args:
            input_terms: List of search terms.

        Returns:
            List of resolved search results.
        """
        return self._rust_searcher.resolve_location(input_terms)

    def resolve_location_batch(
        self, input_terms_batch: List[List[str]]
    ) -> List[List[ResolvedGenericSearchResult]]:
        """Resolve location search results in batch.

        Args:
            input_terms_batch: List of lists of search terms.

        Returns:
            List of lists of resolved search results.
        """
        return self._rust_searcher.resolve_location_batch(input_terms_batch)


# Convenience functions for quick access
def find_location(query: Union[str, List[str]]) -> List[SearchResult]:
    """Convenience function to find a location quickly.

    Args:
        query: Search query string or list of query terms.

    Returns:
        List of SearchResult objects.
    """
    searcher = LocationSearcher()
    return searcher.find(query)


def find_locations_batch(queries: List[List[str]]) -> List[List[SearchResult]]:
    """Convenience function to find locations in batch.

    Args:
        queries: List of query lists.

    Returns:
        List of lists of SearchResult objects.
    """
    searcher = LocationSearcher()
    return searcher.find_batch(queries)


# Re-export the Rust types for advanced users
__all__ = [
    "LocationSearcher",
    "SearchOptions",
    "SearchConfigBuilder",
    "SearchResult",
    "find_location",
    "find_locations_batch",
    # Rust types
    "RustLocationSearcher",
    "RustSearchConfig",
    "RustSearchConfigBuilder",
    "BasicEntry",
    "GenericEntry",
    "LocationContextBasic",
    "LocationContextGeneric",
    "ResolvedBasicSearchResult",
    "ResolvedGenericSearchResult",
    "__version__",
]
