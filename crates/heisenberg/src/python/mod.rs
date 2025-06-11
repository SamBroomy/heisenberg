//! Python bindings for the Heisenberg location search library.
//!
//! This module provides the main Python interface to the Rust implementation,
//! wrapping the core functionality in Python-friendly types and methods.

use heisenberg_data_processing::DataSource;
use pyo3::{prelude::*, types::PyType};
use pyo3_polars::PyDataFrame;

use crate::{
    LocationEntryCore, LocationSearcher, LocationSearcherBuilder, ResolveSearchConfig,
    SearchConfig, SearchConfigBuilder, SearchResult,
    backfill::{BasicEntry, GenericEntry, LocationContext, ResolvedSearchResult},
    data::embedded::METADATA,
};

/// Python wrapper for the main LocationSearcher.
///
/// This class provides access to all location search functionality including
/// administrative and place searches, as well as high-level resolution methods.
#[pyclass(name = "LocationSearcher")]
#[derive(Clone)]
struct PyLocationSearcher {
    inner: LocationSearcher,
}

#[pymethods]
impl PyLocationSearcher {
    /// Create a new LocationSearcher instance using embedded data.
    ///
    /// This is the default constructor that uses the embedded dataset
    /// for instant startup with no downloads required.
    ///
    /// Returns:
    ///     A new LocationSearcher instance.
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails.
    #[new]
    fn py_new(py: Python) -> PyResult<Self> {
        py.allow_threads(|| match LocationSearcher::new_embedded() {
            Ok(inner) => Ok(PyLocationSearcher { inner }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize LocationSearcher: {e}"
            ))),
        })
    }

    /// Create a new LocationSearcher instance with a specific data source.
    ///
    /// This method will use smart initialization with fallback, trying embedded data first,
    /// then downloading and processing the specified data source if needed.
    ///
    /// Args:
    ///     data_source: The data source to use (DataSource.cities15000(), etc.)
    ///
    /// Returns:
    ///     A new LocationSearcher instance.
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails.
    #[staticmethod]
    fn with_data_source(py: Python, data_source: &PyDataSource) -> PyResult<Self> {
        py.allow_threads(|| match LocationSearcher::initialize(data_source.inner) {
            Ok(inner) => Ok(PyLocationSearcher { inner }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize LocationSearcher with data source {:?}: {e}",
                data_source.inner
            ))),
        })
    }

    /// Create a new LocationSearcher instance with fresh indexes.
    ///
    /// This method forces rebuilding of all indexes from scratch, which may take longer
    /// but ensures the most up-to-date data.
    ///
    /// Args:
    ///     data_source: The data source to use (DataSource.cities15000(), etc.)
    ///
    /// Returns:
    ///     A new LocationSearcher instance.
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails.
    #[staticmethod]
    fn with_fresh_indexes(py: Python, data_source: &PyDataSource) -> PyResult<Self> {
        py.allow_threads(
            || match LocationSearcher::new_with_fresh_indexes(data_source.inner) {
                Ok(inner) => Ok(PyLocationSearcher { inner }),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to initialize LocationSearcher with fresh indexes: {e}"
                ))),
            },
        )
    }

    /// Try to load an existing LocationSearcher instance.
    ///
    /// This method attempts to load existing cached indexes without rebuilding.
    /// Returns None if no existing indexes are found.
    ///
    /// Args:
    ///     data_source: The data source to use (DataSource.cities15000(), etc.)
    ///
    /// Returns:
    ///     A LocationSearcher instance if found, None otherwise.
    ///
    /// Raises:
    ///     RuntimeError: If loading fails.
    #[staticmethod]
    fn load_existing(py: Python, data_source: &PyDataSource) -> PyResult<Option<Self>> {
        py.allow_threads(
            || match LocationSearcher::load_existing(data_source.inner) {
                Ok(Some(inner)) => Ok(Some(PyLocationSearcher { inner })),
                Ok(None) => Ok(None),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load existing LocationSearcher: {e}"
                ))),
            },
        )
    }

    /// Search for administrative entities at specific levels.
    ///
    /// Args:
    ///     term: The search term to look for.
    ///     levels: List of administrative levels to search (0-4, where 0 is country).
    ///     previous_result: Optional DataFrame to filter results by previous search.
    ///
    /// Returns:
    ///     Optional DataFrame with search results, None if no matches found.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    #[pyo3(name = "admin_search")]
    #[pyo3(signature = (term, levels, previous_result=None))]
    fn py_admin_search(
        &self,
        py: Python,
        term: &str,
        levels: Vec<u8>,
        previous_result: Option<PyDataFrame>,
    ) -> PyResult<Option<PyDataFrame>> {
        py.allow_threads(|| {
            let prev_df = previous_result.map(|df| df.into());

            match self
                .inner
                .admin_search(term, &levels, prev_df, &Default::default())
            {
                Ok(Some(df)) => Ok(Some(PyDataFrame(df))),
                Ok(None) => Ok(None),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Admin search error: {e}"
                ))),
            }
        })
    }

    /// Search for places (cities, towns, points of interest, etc.).
    ///
    /// Args:
    ///     term: The search term to look for.
    ///     previous_result: Optional DataFrame to filter results by previous search.
    ///
    /// Returns:
    ///     Optional DataFrame with search results, None if no matches found.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    #[pyo3(name = "place_search")]
    #[pyo3(signature = (term, previous_result=None))]
    fn py_place_search(
        &self,
        py: Python,
        term: &str,
        previous_result: Option<PyDataFrame>,
    ) -> PyResult<Option<PyDataFrame>> {
        py.allow_threads(|| {
            let prev_df = previous_result.map(|df| df.into());

            match self.inner.place_search(term, prev_df, &Default::default()) {
                Ok(Some(df)) => Ok(Some(PyDataFrame(df))),
                Ok(None) => Ok(None),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Place search error: {e}"
                ))),
            }
        })
    }

    /// Perform a flexible search with multiple terms using default configuration.
    ///
    /// Args:
    ///     input_terms: List of search terms to process.
    ///
    /// Returns:
    ///     List of search results.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search(&self, py: Python, input_terms: Vec<String>) -> PyResult<Vec<PySearchResult>> {
        py.allow_threads(|| match self.inner.search(&input_terms) {
            Ok(results) => {
                let py_results = results.into_iter().map(PySearchResult::from).collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Search error: {e}"
            ))),
        })
    }

    /// Perform a flexible search with multiple terms using custom configuration.
    ///
    /// Args:
    ///     input_terms: List of search terms to process.
    ///     config: Custom search configuration.
    ///
    /// Returns:
    ///     List of search results.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search_with_config(
        &self,
        py: Python,
        input_terms: Vec<String>,
        config: &PySearchConfig,
    ) -> PyResult<Vec<PySearchResult>> {
        py.allow_threads(
            || match self.inner.search_with_config(&input_terms, &config.inner) {
                Ok(results) => {
                    let py_results = results.into_iter().map(PySearchResult::from).collect();
                    Ok(py_results)
                }
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Search with config error: {e}"
                ))),
            },
        )
    }

    /// Perform batch search for multiple sets of terms using default configuration.
    ///
    /// Args:
    ///     input_terms: List of lists, where each inner list contains terms for one search.
    ///
    /// Returns:
    ///     List of lists of search results, one per input term set.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search_bulk(
        &self,
        py: Python,
        input_terms: Vec<Vec<String>>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        py.allow_threads(|| {
            let results = self.inner.search_bulk(&input_terms).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Batch search error: {e}"
                ))
            })?;
            let py_results = results
                .into_iter()
                .map(|search_results| {
                    search_results
                        .into_iter()
                        .map(PySearchResult::from)
                        .collect()
                })
                .collect();
            Ok(py_results)
        })
    }

    /// Perform batch search for multiple sets of terms using custom configuration.
    ///
    /// Args:
    ///     input_terms: List of lists, where each inner list contains terms for one search.
    ///     config: Custom search configuration.
    ///
    /// Returns:
    ///     List of lists of search results, one per input term set.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search_bulk_with_config(
        &self,
        py: Python,
        input_terms: Vec<Vec<String>>,
        config: &PySearchConfig,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        py.allow_threads(|| {
            let results = self
                .inner
                .search_bulk_with_config(&input_terms, &config.inner)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Batch search with config error: {e}"
                    ))
                })?;
            let py_results = results
                .into_iter()
                .map(|search_results| {
                    search_results
                        .into_iter()
                        .map(PySearchResult::from)
                        .collect()
                })
                .collect();
            Ok(py_results)
        })
    }

    /// Resolve location search terms into structured location data.
    ///
    /// This is the main high-level method that combines search and resolution
    /// to provide clean, structured location results.
    ///
    /// Args:
    ///     input_terms: List of search terms to resolve.
    ///
    /// Returns:
    ///     List of resolved search results with full location context.
    ///
    /// Raises:
    ///     RuntimeError: If the resolution operation fails.
    #[pyo3(name = "resolve_location", signature = (input_terms))]
    fn resolve_location(
        &self,
        py: Python,
        input_terms: Vec<String>,
    ) -> PyResult<Vec<PyResolvedGenericSearchResult>> {
        let resolved_results_rust =
            py.allow_threads(|| self.inner.resolve_location::<_, GenericEntry>(&input_terms));

        match resolved_results_rust {
            Ok(resolved_vec) => {
                let py_results = resolved_vec
                    .into_iter()
                    .map(PyResolvedGenericSearchResult::from)
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Resolve location error: {e}"
            ))),
        }
    }

    /// Resolve location with custom configuration.
    ///
    /// Args:
    ///     input_terms: List of search terms to resolve.
    ///     config: Custom search configuration.
    ///
    /// Returns:
    ///     List of resolved search results with full location context.
    ///
    /// Raises:
    ///     RuntimeError: If the resolution operation fails.
    #[pyo3(name = "resolve_location_with_config", signature = (input_terms, config))]
    fn resolve_location_with_config(
        &self,
        py: Python,
        input_terms: Vec<String>,
        config: &PySearchConfig,
    ) -> PyResult<Vec<PyResolvedGenericSearchResult>> {
        let resolve_config = ResolveSearchConfig {
            search_config: config.inner.clone(),
            resolve_config: Default::default(),
        };

        let resolved_results_rust = py.allow_threads(|| {
            self.inner
                .resolve_location_with_config::<_, GenericEntry>(&input_terms, &resolve_config)
        });

        match resolved_results_rust {
            Ok(resolved_vec) => {
                let py_results = resolved_vec
                    .into_iter()
                    .map(PyResolvedGenericSearchResult::from)
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Resolve location with config error: {e}"
            ))),
        }
    }

    /// Resolve locations in batch.
    ///
    /// Args:
    ///     input_terms_batch: List of lists, where each inner list contains terms for one search.
    ///
    /// Returns:
    ///     List of lists of resolved search results with full location context.
    ///
    /// Raises:
    ///     RuntimeError: If the resolution operation fails.
    #[pyo3(name = "resolve_location_batch", signature = (input_terms_batch))]
    fn resolve_location_batch(
        &self,
        py: Python,
        input_terms_batch: Vec<Vec<String>>,
    ) -> PyResult<Vec<Vec<PyResolvedGenericSearchResult>>> {
        let resolved_results_rust = py.allow_threads(|| {
            self.inner
                .resolve_location_batch::<GenericEntry, _, _>(&input_terms_batch)
        });

        match resolved_results_rust {
            Ok(resolved_batches) => {
                let py_results = resolved_batches
                    .into_iter()
                    .map(|batch| {
                        batch
                            .into_iter()
                            .map(PyResolvedGenericSearchResult::from)
                            .collect()
                    })
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Resolve location batch error: {e}"
            ))),
        }
    }

    /// Resolve locations in batch with custom configuration.
    ///
    /// Args:
    ///     input_terms_batch: List of lists, where each inner list contains terms for one search.
    ///     config: Custom search configuration.
    ///
    /// Returns:
    ///     List of lists of resolved search results with full location context.
    ///
    /// Raises:
    ///     RuntimeError: If the resolution operation fails.
    #[pyo3(name = "resolve_location_batch_with_config", signature = (input_terms_batch, config))]
    fn resolve_location_batch_with_config(
        &self,
        py: Python,
        input_terms_batch: Vec<Vec<String>>,
        config: &PySearchConfig,
    ) -> PyResult<Vec<Vec<PyResolvedGenericSearchResult>>> {
        let resolve_config = ResolveSearchConfig {
            search_config: config.inner.clone(),
            resolve_config: Default::default(),
        };

        let resolved_results_rust = py.allow_threads(|| {
            self.inner
                .resolve_location_batch_with_config::<GenericEntry, _, _>(
                    &input_terms_batch,
                    &resolve_config,
                )
        });

        match resolved_results_rust {
            Ok(resolved_batches) => {
                let py_results = resolved_batches
                    .into_iter()
                    .map(|batch| {
                        batch
                            .into_iter()
                            .map(PyResolvedGenericSearchResult::from)
                            .collect()
                    })
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Resolve location batch with config error: {e}"
            ))),
        }
    }
}

/// Python wrapper for SearchConfig.
///
/// Represents a search configuration with various parameters for controlling
/// location search behavior. Most users should use SearchConfigBuilder instead
/// of creating this directly.
#[pyclass(name = "SearchConfig")]
#[derive(Clone)]
struct PySearchConfig {
    inner: SearchConfig,
}

#[pymethods]
impl PySearchConfig {
    /// Create a new SearchConfig with default settings.
    #[new]
    fn py_new() -> Self {
        PySearchConfig {
            inner: SearchConfig::default(),
        }
    }

    /// Return a string representation of the configuration.
    fn __repr__(&self) -> String {
        format!("SearchConfig(inner={:?})", self.inner)
    }
}

/// Python wrapper for SearchConfigBuilder.
///
/// Provides a fluent interface for building search configurations with
/// customizable parameters for different search scenarios.
#[pyclass(name = "SearchConfigBuilder")]
#[derive(Clone)]
struct PySearchConfigBuilder {
    inner: SearchConfigBuilder,
}

#[pymethods]
impl PySearchConfigBuilder {
    /// Create a new SearchConfigBuilder with default settings.
    #[new]
    fn py_new() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::new(),
        }
    }

    /// Create a configuration optimized for fast searches.
    ///
    /// Returns:
    ///     A SearchConfigBuilder configured for speed over comprehensiveness.
    #[staticmethod]
    fn fast() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::fast(),
        }
    }

    /// Create a configuration optimized for comprehensive results.
    ///
    /// Returns:
    ///     A SearchConfigBuilder configured for thorough searches.
    #[staticmethod]
    fn comprehensive() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::comprehensive(),
        }
    }

    /// Create a configuration optimized for high-quality places only.
    ///
    /// Returns:
    ///     A SearchConfigBuilder configured for important places.
    #[staticmethod]
    fn quality_places() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::quality_places(),
        }
    }

    /// Set the maximum number of results to return.
    ///
    /// Args:
    ///     limit: Maximum number of results (must be positive).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn limit(mut slf: PyRefMut<Self>, limit: usize) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().limit(limit);
        slf
    }

    /// Set the place importance threshold.
    ///
    /// Args:
    ///     threshold: Importance level (1=most important, 5=least important).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn place_importance_threshold(mut slf: PyRefMut<Self>, threshold: u8) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().place_importance_threshold(threshold);
        slf
    }

    /// Enable or disable proactive administrative entity searching.
    ///
    /// Args:
    ///     enabled: Whether to enable proactive admin search.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn proactive_admin_search(mut slf: PyRefMut<Self>, enabled: bool) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().proactive_admin_search(enabled);
        slf
    }

    /// Set the maximum number of administrative terms to process.
    ///
    /// Args:
    ///     max_terms: Maximum number of admin terms.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn max_admin_terms(mut slf: PyRefMut<Self>, max_terms: usize) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().max_admin_terms(max_terms);
        slf
    }

    /// Include all available columns in the search results.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn include_all_columns(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().include_all_columns();
        slf
    }

    /// Configure text search parameters.
    ///
    /// Args:
    ///     fuzzy: Whether to enable fuzzy text matching.
    ///     limit_multiplier: Multiplier for initial result fetching.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn text_search(
        mut slf: PyRefMut<Self>,
        fuzzy: bool,
        limit_multiplier: usize,
    ) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().text_search(fuzzy, limit_multiplier);
        slf
    }

    /// Build the final search configuration.
    ///
    /// Returns:
    ///     A SearchConfig object ready for use.
    fn build(&self) -> PySearchConfig {
        PySearchConfig {
            inner: self.inner.clone().build(),
        }
    }

    /// Return a string representation of the builder.
    fn __repr__(&self) -> String {
        format!("SearchConfigBuilder(inner={:?})", self.inner)
    }
}

/// Python wrapper for SearchResult.
#[pyclass(name = "SearchResult")]
#[derive(Clone)]
struct PySearchResult {
    inner: SearchResult,
}

impl From<SearchResult> for PySearchResult {
    fn from(inner: SearchResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySearchResult {
    /// Get the name of the location.
    fn name(&self) -> Option<String> {
        self.inner.name().map(|s| s.to_string())
    }

    /// Get the geoname ID.
    fn geoname_id(&self) -> Option<u64> {
        self.inner.geoname_id().map(|id| id as u64)
    }

    /// Get the search score.
    fn score(&self) -> Option<f32> {
        self.inner.score().map(|s| s as f32)
    }

    /// Get the feature code.
    fn feature_code(&self) -> Option<String> {
        self.inner.feature_code().map(|s| s.to_string())
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(name={:?}, score={:?})",
            self.name(),
            self.score()
        )
    }
}

/// Python wrapper for LocationContext with BasicEntry.
#[pyclass(name = "LocationContext")]
#[derive(Clone)]
struct PyLocationContextBasic {
    inner: LocationContext<BasicEntry>,
}

impl From<LocationContext<BasicEntry>> for PyLocationContextBasic {
    fn from(inner: LocationContext<BasicEntry>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLocationContextBasic {
    /// Get the admin0 (country) entry.
    #[getter]
    fn admin0(&self) -> Option<BasicEntry> {
        self.inner.admin0.clone()
    }

    /// Get the admin1 (state/province) entry.
    #[getter]
    fn admin1(&self) -> Option<BasicEntry> {
        self.inner.admin1.clone()
    }

    /// Get the admin2 (county) entry.
    #[getter]
    fn admin2(&self) -> Option<BasicEntry> {
        self.inner.admin2.clone()
    }

    /// Get the admin3 entry.
    #[getter]
    fn admin3(&self) -> Option<BasicEntry> {
        self.inner.admin3.clone()
    }

    /// Get the admin4 entry.
    #[getter]
    fn admin4(&self) -> Option<BasicEntry> {
        self.inner.admin4.clone()
    }

    /// Get the place entry.
    #[getter]
    fn place(&self) -> Option<BasicEntry> {
        self.inner.place.clone()
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!(
            "LocationContext(admin0={:?}, place={:?})",
            self.inner.admin0.as_ref().map(|e| e.name()),
            self.inner.place.as_ref().map(|e| e.name())
        )
    }
}

/// Python wrapper for LocationContext with GenericEntry.
#[pyclass(name = "LocationContextGeneric")]
#[derive(Clone)]
struct PyLocationContextGeneric {
    inner: LocationContext<GenericEntry>,
}

impl From<LocationContext<GenericEntry>> for PyLocationContextGeneric {
    fn from(inner: LocationContext<GenericEntry>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLocationContextGeneric {
    /// Get the admin0 (country) entry.
    #[getter]
    fn admin0(&self) -> Option<GenericEntry> {
        self.inner.admin0.clone()
    }

    /// Get the admin1 (state/province) entry.
    #[getter]
    fn admin1(&self) -> Option<GenericEntry> {
        self.inner.admin1.clone()
    }

    /// Get the admin2 (county) entry.
    #[getter]
    fn admin2(&self) -> Option<GenericEntry> {
        self.inner.admin2.clone()
    }

    /// Get the admin3 entry.
    #[getter]
    fn admin3(&self) -> Option<GenericEntry> {
        self.inner.admin3.clone()
    }

    /// Get the admin4 entry.
    #[getter]
    fn admin4(&self) -> Option<GenericEntry> {
        self.inner.admin4.clone()
    }

    /// Get the place entry.
    #[getter]
    fn place(&self) -> Option<GenericEntry> {
        self.inner.place.clone()
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!(
            "LocationContextGeneric(admin0={:?}, place={:?})",
            self.inner.admin0.as_ref().map(|e| e.name()),
            self.inner.place.as_ref().map(|e| e.name())
        )
    }
}

/// Python wrapper for ResolvedSearchResult with BasicEntry.
#[pyclass(name = "ResolvedSearchResult")]
#[derive(Clone)]
struct PyResolvedBasicSearchResult {
    inner: ResolvedSearchResult<BasicEntry>,
}

impl From<ResolvedSearchResult<BasicEntry>> for PyResolvedBasicSearchResult {
    fn from(inner: ResolvedSearchResult<BasicEntry>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyResolvedBasicSearchResult {
    /// Get the location context.
    #[getter]
    fn context(&self) -> PyLocationContextBasic {
        PyLocationContextBasic::from(self.inner.context.clone())
    }

    /// Get the resolution score.
    #[getter]
    fn score(&self) -> f32 {
        self.inner.score as f32
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!("ResolvedSearchResult(score={:.3})", self.inner.score)
    }
}

/// Python wrapper for ResolvedSearchResult with GenericEntry.
#[pyclass(name = "ResolvedSearchResultGeneric")]
#[derive(Clone)]
struct PyResolvedGenericSearchResult {
    inner: ResolvedSearchResult<GenericEntry>,
}

impl From<ResolvedSearchResult<GenericEntry>> for PyResolvedGenericSearchResult {
    fn from(inner: ResolvedSearchResult<GenericEntry>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyResolvedGenericSearchResult {
    /// Get the location context.
    #[getter]
    fn context(&self) -> PyLocationContextGeneric {
        PyLocationContextGeneric::from(self.inner.context.clone())
    }

    /// Get the resolution score.
    #[getter]
    fn score(&self) -> f32 {
        self.inner.score as f32
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!("ResolvedSearchResultGeneric(score={:.3})", self.inner.score)
    }
}

/// Python wrapper for DataSource enum.
///
/// Represents the different data sources available for location search.
/// This provides a strongly-typed way to specify data sources in Python.
#[pyclass(name = "DataSource")]
#[derive(Clone)]
struct PyDataSource {
    inner: DataSource,
}

#[pymethods]
impl PyDataSource {
    /// Create Cities15000 data source (cities with population > 15,000).
    ///
    /// This is the default embedded data source with comprehensive coverage
    /// of major cities worldwide.
    #[classmethod]
    fn cities15000(_cls: &Bound<PyType>) -> Self {
        PyDataSource {
            inner: DataSource::Cities15000,
        }
    }

    /// Create Cities5000 data source (cities with population > 5,000).
    #[classmethod]
    fn cities5000(_cls: &Bound<PyType>) -> Self {
        PyDataSource {
            inner: DataSource::Cities5000,
        }
    }

    /// Create Cities1000 data source (cities with population > 1,000).
    #[classmethod]
    fn cities1000(_cls: &Bound<PyType>) -> Self {
        PyDataSource {
            inner: DataSource::Cities1000,
        }
    }

    /// Create Cities500 data source (cities with population > 500).
    #[classmethod]
    fn cities500(_cls: &Bound<PyType>) -> Self {
        PyDataSource {
            inner: DataSource::Cities500,
        }
    }

    /// Create AllCountries data source (complete GeoNames dataset).
    #[classmethod]
    fn all_countries(_cls: &Bound<PyType>) -> Self {
        PyDataSource {
            inner: DataSource::AllCountries,
        }
    }

    #[classmethod]
    fn embedded(_cls: &Bound<PyType>) -> Self {
        PyDataSource {
            inner: METADATA.source,
        }
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!("DataSource.{:?}", self.inner)
    }

    /// Return the string name of the data source.
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Python wrapper for LocationSearcherBuilder.
///
/// Provides a fluent interface for building LocationSearcher instances with
/// customizable data sources and configuration options.
#[pyclass(name = "LocationSearcherBuilder")]
#[derive(Clone)]
struct PyLocationSearcherBuilder {
    inner: LocationSearcherBuilder,
}

#[pymethods]
impl PyLocationSearcherBuilder {
    /// Create a new LocationSearcherBuilder.
    #[new]
    fn py_new() -> Self {
        PyLocationSearcherBuilder {
            inner: LocationSearcherBuilder::new(),
        }
    }

    /// Set the data source to use.
    ///
    /// Args:
    ///     data_source: The data source to use.
    fn data_source(&mut self, data_source: &PyDataSource) {
        self.inner = self.inner.clone().data_source(data_source.inner);
    }

    /// Set whether to force rebuild indexes.
    ///
    /// Args:
    ///     rebuild: Whether to force rebuild of indexes.
    fn force_rebuild(&mut self, rebuild: bool) {
        self.inner = self.inner.clone().force_rebuild(rebuild);
    }

    /// Set whether to use embedded data as fallback.
    ///
    /// Args:
    ///     fallback: Whether to enable embedded data fallback.
    fn embedded_fallback(&mut self, fallback: bool) {
        self.inner = self.inner.clone().embedded_fallback(fallback);
    }

    /// Build the LocationSearcher.
    ///
    /// Returns:
    ///     A new LocationSearcher instance.
    ///
    /// Raises:
    ///     RuntimeError: If building fails.
    fn build(&self, py: Python) -> PyResult<PyLocationSearcher> {
        py.allow_threads(|| match self.inner.clone().build() {
            Ok(inner) => Ok(PyLocationSearcher { inner }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to build LocationSearcher: {e}"
            ))),
        })
    }

    /// Return a string representation.
    fn __repr__(&self) -> String {
        format!("LocationSearcherBuilder(inner={:?})", self.inner)
    }
}

//Define the Python module

/// The main Heisenberg Python module.
///
/// This module provides location search functionality using a Rust backend,
/// with comprehensive search capabilities for geographic locations worldwide.
#[pymodule]
fn heisenberg(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Initialize Python logging for Rust components
    pyo3_log::init();

    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add main classes
    m.add_class::<PyLocationSearcher>()?;
    m.add_class::<PySearchConfig>()?;
    m.add_class::<PySearchConfigBuilder>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyDataSource>()?;
    m.add_class::<PyLocationSearcherBuilder>()?;

    // Add entry types (not exposed in high-level API but available for advanced users)
    m.add_class::<BasicEntry>()?;
    m.add_class::<GenericEntry>()?;

    // Add context and result types
    m.add_class::<PyLocationContextBasic>()?;
    m.add_class::<PyLocationContextGeneric>()?;
    m.add_class::<PyResolvedBasicSearchResult>()?;
    m.add_class::<PyResolvedGenericSearchResult>()?;

    Ok(())
}
