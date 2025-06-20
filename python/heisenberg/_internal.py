"""
Internal interface to the Rust Heisenberg library.

This module provides the main interface to the Rust LocationSearcher and related
functionality, serving as the bridge between the Rust implementation and the
Python user-facing API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
if TYPE_CHECKING:
    import polars as pl


# Import the Rust bindings
from .heisenberg import DataSource as RustDataSource
from .heisenberg import LocationContext, LocationEntry, ResolvedSearchResult, __version__
from .heisenberg import LocationSearcher as RustLocationSearcher
from .heisenberg import LocationSearcherBuilder as RustLocationSearcherBuilder
from .heisenberg import SearchConfig as RustSearchConfig
from .heisenberg import SearchConfigBuilder as RustSearchConfigBuilder

logger = logging.getLogger(__name__)


class DataSource:
    """High-level Python wrapper for DataSource.

    This provides a clean Python interface for selecting data sources without
    directly exposing the Rust bindings to end users.

    Examples:
        Using embedded data (recommended):

        >>> ds = DataSource.embedded()
        >>> searcher = LocationSearcher(data_source=ds)

        Using different population thresholds:

        >>> ds = DataSource.cities5000()  # Cities with pop > 5,000
        >>> searcher = LocationSearcher.with_data_source(ds)

        For comprehensive global coverage:

        >>> ds = DataSource.all_countries()
        >>> searcher = LocationSearcher.with_data_source(ds)
    """

    def __init__(self, rust_data_source: RustDataSource) -> None:
        """Initialize with a Rust DataSource object.

        Args:
            rust_data_source: The underlying Rust DataSource object.
        """
        self._rust_data_source = rust_data_source

    @classmethod
    def embedded(cls) -> Self:
        """Get the embedded data source.

        This uses the data that was compiled into the library at build time.
        This is the fastest option and works without any downloads.

        Returns:
            DataSource: Wrapper for embedded data source.
        """
        return cls(RustDataSource.embedded())

    @classmethod
    def cities15000(cls) -> Self:
        """Get cities with population > 15,000.

        This provides good coverage of major cities worldwide while keeping
        the dataset manageable in size.

        Returns:
            DataSource: Wrapper for cities15000 data source.
        """
        return cls(RustDataSource.cities15000())

    @classmethod
    def cities5000(cls) -> Self:
        """Get cities with population > 5,000.

        This includes more smaller cities and towns, providing more comprehensive
        coverage at the cost of larger dataset size.

        Returns:
            DataSource: Wrapper for cities5000 data source.
        """
        return cls(RustDataSource.cities5000())

    @classmethod
    def cities1000(cls) -> Self:
        """Get cities with population > 1,000.

        This includes small towns and villages, providing very comprehensive
        coverage but with a significantly larger dataset.

        Returns:
            DataSource: Wrapper for cities1000 data source.
        """
        return cls(RustDataSource.cities1000())

    @classmethod
    def cities500(cls) -> Self:
        """Get cities with population > 500.

        This includes very small settlements, providing maximum coverage
        but with the largest dataset size.

        Returns:
            DataSource: Wrapper for cities500 data source.
        """
        return cls(RustDataSource.cities500())

    @classmethod
    def all_countries(cls) -> Self:
        """Get the complete GeoNames dataset.

        This includes all geographic features from the GeoNames database,
        not just populated places. This is the most comprehensive option
        but also the largest.

        Returns:
            DataSource: Wrapper for all_countries data source.
        """
        return cls(RustDataSource.all_countries())

    def __str__(self) -> str:
        """Return string representation."""
        return str(self._rust_data_source)

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"DataSource({self._rust_data_source!r})"


class LocationSearcherBuilderWrapper:
    """High-level Python wrapper for LocationSearcherBuilder.

    This provides a fluent interface for building LocationSearcher instances with
    customizable data sources and configuration options.

    Examples:
        Basic usage:

        >>> builder = LocationSearcherBuilderWrapper()
        >>> searcher = builder.build()

        With custom data source:

        >>> ds = DataSource.cities5000()
        >>> builder = LocationSearcherBuilderWrapper()
        >>> searcher = builder.data_source(ds).build()

        Force rebuild indexes:

        >>> builder = LocationSearcherBuilderWrapper()
        >>> searcher = builder.force_rebuild(True).build()

        Complete configuration:

        >>> ds = DataSource.cities15000()
        >>> builder = LocationSearcherBuilderWrapper()
        >>> searcher = builder.data_source(ds).force_rebuild(True).embedded_fallback(False).build()
    """

    def __init__(self) -> None:
        """Initialize a new LocationSearcherBuilderWrapper."""
        self._rust_builder = RustLocationSearcherBuilder()

    def data_source(self, data_source: DataSource) -> Self:
        """Set the data source to use.

        Args:
            data_source: The data source to use.

        Returns:
            LocationSearcherBuilderWrapper: Self for method chaining.
        """
        self._rust_builder.data_source(data_source._rust_data_source)
        return self

    def force_rebuild(self, *, rebuild: bool) -> Self:
        """Set whether to force rebuild indexes.

        Args:
            rebuild: Whether to force rebuild of indexes.

        Returns:
            LocationSearcherBuilderWrapper: Self for method chaining.
        """
        self._rust_builder.force_rebuild(rebuild)
        return self

    def embedded_fallback(self, *, fallback: bool) -> Self:
        """Set whether to use embedded data as fallback.

        Args:
            fallback: Whether to enable embedded data fallback.

        Returns:
            LocationSearcherBuilderWrapper: Self for method chaining.
        """
        self._rust_builder.embedded_fallback(fallback)
        return self

    def build(self) -> LocationSearcher:
        """Build the LocationSearcher.

        Returns:
            LocationSearcher: A new LocationSearcher instance.

        Raises:
            RuntimeError: If building fails.
        """
        rust_searcher = self._rust_builder.build()
        instance = LocationSearcher.__new__(LocationSearcher)
        instance._rust_searcher = rust_searcher
        return instance

    def __repr__(self) -> str:
        """Return string representation."""
        return f"LocationSearcherBuilderWrapper({self._rust_builder!r})"


@dataclass
class SearchOptions:
    """Configuration options for location searches.

    This class provides a pythonic way to configure search parameters with
    sensible defaults and validation. Most users will use SearchConfigBuilder
    instead of creating this directly.

    Examples:
        Basic usage with defaults:

        >>> options = SearchOptions()
        >>> options.limit
        20

        Customized search for important places only:

        >>> options = SearchOptions(limit=10, place_importance_threshold=2, fuzzy_search=False)

        Location-biased search:

        >>> # Search biased toward New York coordinates
        >>> options = SearchOptions(center_latitude=40.7128, center_longitude=-74.0060)

    Attributes:
        limit: Maximum number of results to return. Default is 20.
        include_all_columns: Whether to include all available data columns
            in results. Default is False.
        max_admin_terms: Maximum number of administrative terms to process
            when searching. Default is 5.
        place_importance_threshold: Filter places by importance level.
            1 = most important (major cities), 5 = least important (small villages).
            Default is 5 (include all).
        proactive_admin_search: Whether to automatically search for administrative
            entities when processing multi-term queries. Default is True.
        fuzzy_search: Enable fuzzy string matching for typos and variations.
            Default is True.
        search_multiplier: How many more results to fetch initially for ranking.
            Higher values may improve result quality. Default is 3.

        Scoring weights for administrative entities (should sum to ~1.0):
        admin_text_weight: Weight for text similarity. Default 0.4.
        admin_population_weight: Weight for population size. Default 0.25.
        admin_parent_weight: Weight for parent hierarchy match. Default 0.20.
        admin_feature_weight: Weight for feature type match. Default 0.15.

        Scoring weights for places (should sum to ~1.0):
        place_text_weight: Weight for text similarity. Default 0.4.
        place_importance_weight: Weight for place importance. Default 0.20.
        place_feature_weight: Weight for feature type match. Default 0.15.
        place_parent_weight: Weight for parent hierarchy match. Default 0.15.
        place_distance_weight: Weight for distance from bias point. Default 0.05.

        Location bias (for distance scoring):
        center_latitude: Latitude for location bias. Default None.
        center_longitude: Longitude for location bias. Default None.
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
    center_latitude: float | None = None
    center_longitude: float | None = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.limit <= 0:
            msg = "limit must be positive"
            raise ValueError(msg)
        if not 1 <= self.place_importance_threshold <= 5:
            msg = "place_importance_threshold must be between 1 and 5"
            raise ValueError(msg)

        # Validate weights sum approximately to 1.0 for each category
        admin_weight_sum = (
            self.admin_text_weight + self.admin_population_weight + self.admin_parent_weight + self.admin_feature_weight
        )
        if not 0.95 <= admin_weight_sum <= 1.05:
            logger.warning(f"Admin weights sum to {admin_weight_sum:.3f}, should be close to 1.0")

        place_weight_sum = (
            self.place_text_weight
            + self.place_importance_weight
            + self.place_feature_weight
            + self.place_parent_weight
            + self.place_distance_weight
        )
        if not 0.95 <= place_weight_sum <= 1.05:
            logger.warning(f"Place weights sum to {place_weight_sum:.3f}, should be close to 1.0")

    def to_rust_config(self) -> RustSearchConfig:
        """Convert to a Rust SearchConfig object.

        Returns:
            RustSearchConfig: The equivalent Rust configuration object.
        """
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
    """A location search result with geographic information and confidence score.

    This class represents a single location found during a search, including
    its basic information, administrative hierarchy, geographic coordinates,
    and a confidence score indicating how well it matches the search query.

    Examples:
        Accessing basic information:

        >>> result = search_results[0]
        >>> print(f"{result.name} (ID: {result.geoname_id})")
        London (ID: 2643743)
        >>> print(f"Score: {result.score:.3f}")
        Score: 0.952

        Getting the administrative hierarchy:

        >>> hierarchy = result.admin_hierarchy()
        >>> print(" -> ".join(hierarchy))
        England -> United Kingdom

        Building a complete address:

        >>> full_name = result.full_name()
        >>> print(full_name)
        London, England, United Kingdom

        Converting to dictionary for serialization:

        >>> data = result.to_dict()
        >>> import json
        >>> json.dumps(data, indent=2)

    Attributes:
        geoname_id: Unique identifier from the GeoNames database.
        name: Primary name of the location.
        feature_code: GeoNames feature code (e.g., 'PPLC' for capital city).
        admin0_name: Country name.
        admin1_name: State/province name.
        admin2_name: County/region name.
        admin3_name: Sub-region name.
        admin4_name: Local area name.
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        population: Population if available.
        score: Confidence score (0.0 to 1.0, higher is better).
        admin_level: Administrative level of this result (0=country, higher=more local).
    """

    # Core location information
    geoname_id: int
    name: str
    feature_code: str

    # Administrative hierarchy
    admin0_name: str | None = None
    admin1_name: str | None = None
    admin2_name: str | None = None
    admin3_name: str | None = None
    admin4_name: str | None = None

    # Geographic information
    latitude: float | None = None
    longitude: float | None = None
    population: int | None = None

    # Search metadata
    score: float = 0.0
    admin_level: int = 0

    @classmethod
    def from_polars_row(cls, row: dict[str, Any]) -> Self:
        """Create SearchResult from a Polars DataFrame row.

        Args:
            row: Dictionary representing a DataFrame row.

        Returns:
            SearchResult: New SearchResult instance.
        """
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
    def from_resolved_result(cls, resolved: ResolvedSearchResult) -> Self:
        """Create SearchResult from a ResolvedSearchResult.

        Args:
            resolved: Resolved search result from Rust layer.

        Returns:
            SearchResult: New SearchResult instance.
        """
        # Get context
        context = resolved.context  # Use property access instead of method

        # Try to get the best entry (place first, then highest admin level)
        entry = None
        admin_level = 0

        if context.place:
            entry = context.place
            admin_level = 0
        else:
            # Find the highest admin level that exists
            for level in range(4, -1, -1):
                admin_entry = getattr(context, f"admin{level}", None)
                if admin_entry:
                    entry = admin_entry
                    admin_level = level
                    break

        if not entry:
            # Fallback to empty values
            return cls(geoname_id=0, name="Unknown", feature_code="", score=resolved.score, admin_level=admin_level)

        # Build admin hierarchy
        admin_names = {}
        for i in range(5):
            admin_entry = getattr(context, f"admin{i}", None)
            if admin_entry:
                admin_names[f"admin{i}_name"] = admin_entry.name

        return cls(
            geoname_id=entry.geoname_id,
            name=entry.name,
            feature_code=getattr(entry, "feature_code", ""),
            admin0_name=admin_names.get("admin0_name"),
            admin1_name=admin_names.get("admin1_name"),
            admin2_name=admin_names.get("admin2_name"),
            admin3_name=admin_names.get("admin3_name"),
            admin4_name=admin_names.get("admin4_name"),
            latitude=getattr(entry, "latitude", None),
            longitude=getattr(entry, "longitude", None),
            population=getattr(entry, "population", None),
            score=resolved.score,
            admin_level=admin_level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary with all result fields.
        """
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

    def admin_hierarchy(self) -> list[str]:
        """Get the administrative hierarchy as a list of names.

        Returns:
            List[str]: Administrative names from most local to most general.
                      For example: ['England', 'United Kingdom']
        """
        return [
            admin_name
            for admin_name in [self.admin0_name, self.admin1_name, self.admin2_name, self.admin3_name, self.admin4_name]
            if admin_name
        ]

    def full_name(self) -> str:
        """Get the full name including administrative hierarchy.

        Returns:
            str: Complete location name with administrative context.
                 For example: "London, England, United Kingdom"
        """
        parts = [self.name, *self.admin_hierarchy()]
        return ", ".join(parts)


class SearchConfigBuilder:
    """Builder for creating search configurations with a fluent interface.

    This class provides a convenient way to build search configurations
    step by step using method chaining. It's the recommended way to create
    custom search configurations.

    Examples:
        Quick search configuration:

        >>> config = SearchConfigBuilder().limit(5).place_importance(2).fuzzy_search(False).build()

        Location-biased search:

        >>> # Search near London with custom weights
        >>> config = (
        ...     SearchConfigBuilder()
        ...     .limit(10)
        ...     .location_bias(51.5074, -0.1278)
        ...     .place_weights(text=0.5, importance=0.3, distance=0.2)
        ...     .build()
        ... )

        Using preset configurations:

        >>> fast_config = SearchConfigBuilder.fast().build()
        >>> comprehensive_config = SearchConfigBuilder.comprehensive().build()
        >>> quality_config = SearchConfigBuilder.quality_places().build()
    """

    def __init__(self) -> None:
        """Initialize a new SearchConfigBuilder with default options."""
        self.options = SearchOptions()

    def limit(self, limit: int) -> Self:
        """Set the maximum number of results to return.

        Args:
            limit: Maximum number of results (must be positive).

        Returns:
            SearchConfigBuilder: Self for method chaining.
        """
        self.options.limit = limit
        return self

    def place_importance(self, threshold: int) -> Self:
        """Set the place importance threshold.

        Args:
            threshold: Importance level (1=most important cities, 5=all places).

        Returns:
            SearchConfigBuilder: Self for method chaining.
        """
        self.options.place_importance_threshold = threshold
        return self

    def admin_search(self, *, enabled: bool) -> Self:
        """Enable or disable proactive administrative search.

        When enabled, the searcher will automatically look for administrative
        entities (countries, states, etc.) when processing multi-term queries.

        Args:
            enabled: Whether to enable proactive admin search.

        Returns:
            SearchConfigBuilder: Self for method chaining.
        """
        self.options.proactive_admin_search = enabled
        return self

    def fuzzy_search(self, *, enabled: bool) -> Self:
        """Enable or disable fuzzy string matching.

        Fuzzy matching helps find results even with typos or slight variations
        in spelling, but may be slower and less precise.

        Args:
            enabled: Whether to enable fuzzy matching.

        Returns:
            SearchConfigBuilder: Self for method chaining.
        """
        self.options.fuzzy_search = enabled
        return self

    def location_bias(self, latitude: float, longitude: float) -> Self:
        """Set location bias for distance-based scoring.

        When set, results closer to this location will be scored higher.
        Useful for disambiguating place names (e.g., Cambridge, UK vs Cambridge, MA).

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.

        Returns:
            SearchConfigBuilder: Self for method chaining.
        """
        self.options.center_latitude = latitude
        self.options.center_longitude = longitude
        return self

    def admin_weights(
        self, text: float = 0.4, population: float = 0.25, parent: float = 0.20, feature: float = 0.15
    ) -> Self:
        """Set scoring weights for administrative entities.

        These weights control how administrative entities (countries, states, etc.)
        are scored. The weights should sum to approximately 1.0.

        Args:
            text: Weight for text similarity (how well the name matches).
            population: Weight for population size (larger = higher score).
            parent: Weight for parent hierarchy match.
            feature: Weight for feature type relevance.

        Returns:
            SearchConfigBuilder: Self for method chaining.
        """
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
    ) -> Self:
        """Set scoring weights for places.

        These weights control how places (cities, landmarks, etc.) are scored.
        The weights should sum to approximately 1.0.

        Args:
            text: Weight for text similarity.
            importance: Weight for place importance (major cities score higher).
            feature: Weight for feature type relevance.
            parent: Weight for parent hierarchy match.
            distance: Weight for distance from bias point (if set).

        Returns:
            SearchConfigBuilder: Self for method chaining.
        """
        self.options.place_text_weight = text
        self.options.place_importance_weight = importance
        self.options.place_feature_weight = feature
        self.options.place_parent_weight = parent
        self.options.place_distance_weight = distance
        return self

    def build(self) -> SearchOptions:
        """Build the final search configuration.

        Returns:
            SearchOptions: The configured search options.
        """
        return self.options

    @classmethod
    def fast(cls) -> Self:
        """Create a configuration optimized for speed.

        Returns fewer results with stricter filtering to provide fast responses.
        Good for interactive applications where speed matters more than completeness.

        Returns:
            SearchConfigBuilder: Builder configured for fast searches.
        """
        return cls().limit(10).place_importance(3).admin_search(enabled=False).fuzzy_search(enabled=False)

    @classmethod
    def comprehensive(cls) -> Self:
        """Create a configuration optimized for comprehensive results.

        Returns more results with broader matching to find obscure or variant
        location names. Good for data processing where you want to catch everything.

        Returns:
            SearchConfigBuilder: Builder configured for comprehensive searches.
        """
        return cls().limit(50).place_importance(5).admin_search(enabled=True).fuzzy_search(enabled=True)

    @classmethod
    def quality_places(cls) -> Self:
        """Create a configuration for high-quality places only.

        Filters to important places like major cities and well-known landmarks.
        Good for user-facing applications where you want clean, recognizable results.

        Returns:
            SearchConfigBuilder: Builder configured for quality places.
        """
        return cls().limit(20).place_importance(2).admin_search(enabled=True).fuzzy_search(enabled=True)


class LocationSearcher:
    """High-level Python interface to the Rust location searcher.

    This is the main class that most users will interact with. It provides
    a convenient Python API over the raw Rust bindings, converting between
    Rust DataFrames and Python SearchResult objects.

    Examples:
        Basic usage:

        >>> searcher = LocationSearcher()
        >>> results = searcher.find("London")
        >>> print(f"Found {len(results)} results")
        >>> print(f"Top result: {results[0].name}")

        Search with custom configuration:

        >>> config = SearchConfigBuilder().limit(5).place_importance(2).build()
        >>> results = searcher.find("Paris", config)

        Batch processing:

        >>> queries = [["London", "UK"], ["Paris", "France"], ["Tokyo", "Japan"]]
        >>> batch_results = searcher.find_batch(queries)
        >>> for i, results in enumerate(batch_results):
        ...     print(f"Query {i}: {len(results)} results")

        Different search modes:

        >>> # Quick search for speed
        >>> quick_results = searcher.find_quick("Berlin")
        >>>
        >>> # Comprehensive search for completeness
        >>> comprehensive_results = searcher.find_comprehensive("Berlin")
        >>>
        >>> # Important places only
        >>> important_results = searcher.find_important_places("Berlin")
    """

    def __init__(self, data_source: DataSource | None = None, *, rebuild_indexes: bool = False) -> None:
        """Initialize the location searcher.

        Args:
            rebuild_indexes: Whether to rebuild the search indexes from scratch.
                            Set to True if you want to force a fresh index build,
                            which may take several minutes but ensures up-to-date data.
            data_source: Optional data source to use. If None, uses embedded data.
        """
        if rebuild_indexes:
            if data_source is None:
                data_source = DataSource.embedded()
            self._rust_searcher = RustLocationSearcher.with_fresh_indexes(data_source._rust_data_source)
        else:
            # Use default embedded data loading
            self._rust_searcher = RustLocationSearcher()

    @classmethod
    def with_data_source(cls, data_source: DataSource) -> Self:
        """Create a LocationSearcher with a specific data source.

        This method will use smart initialization with fallback, trying embedded data first,
        then downloading and processing the specified data source if needed.

        Args:
            data_source: The data source to use.

        Returns:
            LocationSearcher: New LocationSearcher instance.

        Examples:
            >>> ds = DataSource.cities5000()
            >>> searcher = LocationSearcher.with_data_source(ds)
        """
        instance = cls.__new__(cls)
        instance._rust_searcher = RustLocationSearcher.with_data_source(data_source._rust_data_source)
        return instance

    @classmethod
    def with_fresh_indexes(cls, data_source: DataSource) -> Self:
        """Create a LocationSearcher with fresh indexes for a specific data source.

        This forces rebuilding of all indexes from scratch, which may take longer
        but ensures the most up-to-date data.

        Args:
            data_source: The data source to use.

        Returns:
            LocationSearcher: New LocationSearcher instance.

        Examples:
            >>> ds = DataSource.cities15000()
            >>> searcher = LocationSearcher.with_fresh_indexes(ds)
        """
        instance = cls.__new__(cls)
        instance._rust_searcher = RustLocationSearcher.with_fresh_indexes(data_source._rust_data_source)
        return instance

    @classmethod
    def load_existing(cls, data_source: DataSource) -> Self | None:
        """Try to load an existing LocationSearcher instance.

        This method attempts to load existing cached indexes without rebuilding.
        Returns None if no existing indexes are found.

        Args:
            data_source: The data source to use.

        Returns:
            LocationSearcher instance if found, None otherwise.

        Examples:
            >>> ds = DataSource.embedded()
            >>> searcher = LocationSearcher.load_existing(ds)
            >>> if searcher is None:
            ...     searcher = LocationSearcher.with_data_source(ds)
        """
        rust_searcher = RustLocationSearcher.load_existing(data_source._rust_data_source)
        if rust_searcher is None:
            return None
        instance = cls.__new__(cls)
        instance._rust_searcher = rust_searcher
        return instance

    def find(self, query: str | list[str], config: SearchOptions | None = None) -> list[SearchResult]:
        """Find locations matching the query.

        This is the main search method that handles both simple string queries
        and multi-term searches with optional custom configuration.

        Important: When providing multiple terms, the input should be in descending
        'size' order (largest to smallest location) for optimal results:
        ['Country', 'State', 'County', 'Place']

        Args:
            query: Search query as a string ("New York") or list of terms
                  in descending size order (["California", "San Francisco"]).
            config: Optional search configuration. If None, uses default settings.

        Returns:
            List[SearchResult]: List of matching locations sorted by relevance.

        Examples:
            Simple string search:

            >>> results = searcher.find("Tokyo")

            Multi-term search (largest to smallest):

            >>> results = searcher.find(["United States", "California", "San Francisco"])
            >>> results = searcher.find(["France", "Paris"])

            With custom configuration:

            >>> config = SearchConfigBuilder().limit(3).build()
            >>> results = searcher.find("London", config)
        """
        if isinstance(query, str):
            query = [query]

        try:
            if config:
                resolved_results = self._rust_searcher.resolve_location_with_config(query, config.to_rust_config())
            else:
                resolved_results = self._rust_searcher.resolve_location(query)

            return [SearchResult.from_resolved_result(result) for result in resolved_results]
        except Exception as e:
            logger.exception("Error in find", extra={"exception": str(e)})
            return []

    def find_quick(self, query: str | list[str]) -> list[SearchResult]:
        """Quick search with fast configuration.

        Optimized for speed over completeness. Returns fewer results with
        stricter matching criteria.

        Args:
            query: Search query string or list of query terms.

        Returns:
            List[SearchResult]: List of matching locations.
        """
        config = SearchConfigBuilder.fast().build()
        return self.find(query, config)

    def find_comprehensive(self, query: str | list[str]) -> list[SearchResult]:
        """Comprehensive search with detailed configuration.

        Optimized for completeness over speed. Returns more results with
        broader matching criteria, including fuzzy matching.

        Args:
            query: Search query string or list of query terms.

        Returns:
            List[SearchResult]: List of matching locations.
        """
        config = SearchConfigBuilder.comprehensive().build()
        return self.find(query, config)

    def find_important_places(self, query: str | list[str]) -> list[SearchResult]:
        """Search for important places only.

        Filters results to include only major cities and well-known locations.
        Good for user-facing applications where you want clean, recognizable results.

        Args:
            query: Search query string or list of query terms.

        Returns:
            List[SearchResult]: List of important places matching the query.
        """
        config = SearchConfigBuilder.quality_places().build()
        return self.find(query, config)

    def find_batch(self, queries: list[list[str]], config: SearchOptions | None = None) -> list[list[SearchResult]]:
        """Find locations for multiple queries in batch.

        More efficient than calling find() multiple times, especially for
        large numbers of queries.

        Important: Each query list should contain terms in descending 'size' order
        (largest to smallest location) for optimal results:
        ['Country', 'State', 'County', 'Place']

        Args:
            queries: List of query lists. Each inner list represents one search
                    with terms in descending size order.
            config: Optional search configuration applied to all queries.

        Returns:
            List[List[SearchResult]]: List of result lists, one per input query.

        Examples:
            Process multiple queries efficiently (largest to smallest terms):

            >>> queries = [["United Kingdom", "London"], ["France", "Paris"], ["Japan", "Tokyo"]]
            >>> batch_results = searcher.find_batch(queries)
            >>> for i, results in enumerate(batch_results):
            ...     if results:
            ...         print(f"Query {i}: {results[0].name}")
        """
        try:
            if config:
                resolved_batches = self._rust_searcher.resolve_location_batch_with_config(
                    queries, config.to_rust_config()
                )
            else:
                resolved_batches = self._rust_searcher.resolve_location_batch(queries)

            return [[SearchResult.from_resolved_result(result) for result in batch] for batch in resolved_batches]
        except Exception as e:
            logger.exception("Error in find_batch", extra={"exception": str(e)})
            return [[] for _ in queries]

    # Low-level access to Rust methods for advanced users
    def admin_search(
        self, term: str, levels: list[int], previous_result: pl.DataFrame | None = None
    ) -> pl.DataFrame | None:
        """Search for administrative entities at specific levels.

        Args:
            term: Search term.
            levels: Administrative levels to search (0-4).
            previous_result: Optional previous search result to filter by.

        Returns:
            Polars DataFrame with search results, or None if no results.
        """
        return self._rust_searcher.admin_search(term, levels, previous_result)

    def place_search(self, term: str, previous_result: pl.DataFrame | None = None) -> pl.DataFrame | None:
        """Search for places.

        Args:
            term: Search term.
            previous_result: Optional previous search result to filter by.

        Returns:
            Polars DataFrame with search results, or None if no results.
        """
        return self._rust_searcher.place_search(term, previous_result)

    def search_raw(self, input_terms: list[str]) -> list[pl.DataFrame]:
        """Low-level search returning raw DataFrames.

        Important: Input terms should be in descending 'size' order (largest to
        smallest location) for optimal results: ['Country', 'State', 'County', 'Place']

        Args:
            input_terms: List of search terms in descending size order.

        Returns:
            List of Polars DataFrames.
        """
        return self._rust_searcher.search(input_terms)

    def search_raw_with_config(self, input_terms: list[str], config: SearchOptions) -> list[pl.DataFrame]:
        """Low-level search with configuration returning raw DataFrames.

        Important: Input terms should be in descending 'size' order (largest to
        smallest location) for optimal results: ['Country', 'State', 'County', 'Place']

        Args:
            input_terms: List of search terms in descending size order.
            config: Search configuration.

        Returns:
            List of Polars DataFrames.
        """
        return self._rust_searcher.search_with_config(input_terms, config.to_rust_config())

    def resolve_location(self, input_terms: list[str]) -> list[ResolvedSearchResult]:
        """Resolve location search results.

        Important: Input terms should be in descending 'size' order (largest to
        smallest location) for optimal results: ['Country', 'State', 'County', 'Place']

        Args:
            input_terms: List of search terms in descending size order.

        Returns:
            List of resolved search results.
        """
        return self._rust_searcher.resolve_location(input_terms)

    def resolve_location_batch(self, input_terms_batch: list[list[str]]) -> list[list[ResolvedSearchResult]]:
        """Resolve location search results in batch.

        Important: Each list of input terms should be in descending 'size' order
        (largest to smallest location) for optimal results:
        ['Country', 'State', 'County', 'Place']

        Args:
            input_terms_batch: List of lists of search terms, each in descending size order.

        Returns:
            List of lists of resolved search results.
        """
        return self._rust_searcher.resolve_location_batch(input_terms_batch)


# Convenience functions for quick access
def find_location(query: str | list[str]) -> list[SearchResult]:
    """Convenience function to find a location quickly.

    Creates a temporary LocationSearcher instance and performs a search.
    Good for one-off searches, but if you're doing multiple searches,
    create a LocationSearcher instance to reuse the loaded data.

    Important: When providing multiple terms, list them in descending 'size' order
    (largest to smallest location) for optimal results:
    ['Country', 'State', 'County', 'Place']

    Args:
        query: Search query string or list of query terms in descending size order.

    Returns:
        List[SearchResult]: List of matching locations.

    Examples:
        Quick one-off search:

        >>> results = find_location("Berlin")
        >>> if results:
        ...     print(f"Found: {results[0].name}")

        Multi-term search (largest to smallest):

        >>> results = find_location(["Germany", "Berlin"])
        >>> results = find_location(["United States", "Florida", "Miami"])
    """
    searcher = LocationSearcher()
    return searcher.find(query)


def find_locations_batch(queries: list[list[str]]) -> list[list[SearchResult]]:
    """Convenience function to find locations in batch.

    Creates a temporary LocationSearcher instance and performs batch search.
    Good for one-off batch operations.

    Important: Each query list should contain terms in descending 'size' order
    (largest to smallest location) for optimal results:
    ['Country', 'State', 'County', 'Place']

    Args:
        queries: List of query lists, each with terms in descending size order.

    Returns:
        List[List[SearchResult]]: List of result lists, one per input query.

    Examples:
        Quick batch search (largest to smallest terms):

        >>> queries = [["Italy", "Rome"], ["Austria", "Vienna"], ["Czech Republic", "Prague"]]
        >>> batch_results = find_locations_batch(queries)
        >>> for results in batch_results:
        ...     if results:
        ...         print(f"Found: {results[0].name}")
    """
    searcher = LocationSearcher()
    return searcher.find_batch(queries)


# Export both user-facing wrappers and raw Rust types
# User-facing wrappers are imported by __init__.py
# Raw Rust types are only accessible via direct _internal import

__all__ = [
    "DataSource",
    "LocationContext",
    "LocationEntry",
    # User-facing Python wrapper classes (imported by __init__.py)
    "LocationSearcher",
    "LocationSearcherBuilderWrapper",
    "ResolvedSearchResult",
    "RustDataSource",
    # Raw Rust bindings (only accessible via _internal import for debugging/testing)
    "RustLocationSearcher",
    "RustLocationSearcherBuilder",
    "RustSearchConfig",
    "RustSearchConfigBuilder",
    "SearchConfigBuilder",
    "SearchOptions",
    "SearchResult",
    "__version__",
    "find_location",
    "find_locations_batch",
]
