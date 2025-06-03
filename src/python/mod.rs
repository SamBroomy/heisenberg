//! Python bindings for the Heisenberg location search library.
//!
//! This module provides the main Python interface to the Rust implementation,
//! wrapping the core functionality in Python-friendly types and methods.

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::backfill::python_wrappers::*;
use crate::config::SearchConfigBuilder;
use crate::core::LocationSearcher;
use crate::search::SearchConfig;

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
    /// Create a new LocationSearcher instance.
    ///
    /// Args:
    ///     overwrite_indexes: Whether to rebuild search indexes from scratch.
    ///
    /// Returns:
    ///     A new LocationSearcher instance.
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails.
    #[new]
    fn py_new(py: Python, overwrite_indexes: bool) -> PyResult<Self> {
        py.allow_threads(|| match LocationSearcher::new(overwrite_indexes) {
            Ok(inner) => Ok(PyLocationSearcher { inner }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize LocationSearcher: {}",
                e
            ))),
        })
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
                    "Admin search error: {}",
                    e
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
                    "Place search error: {}",
                    e
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
    ///     List of DataFrames containing search results for each term.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search(&self, py: Python, input_terms: Vec<String>) -> PyResult<Vec<PyDataFrame>> {
        py.allow_threads(|| match self.inner.search(&input_terms) {
            Ok(results) => {
                let py_results = results
                    .into_iter()
                    .map(|df| PyDataFrame(df.into_df()))
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Search error: {}",
                e
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
    ///     List of DataFrames containing search results for each term.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search_with_config(
        &self,
        py: Python,
        input_terms: Vec<String>,
        config: &PySearchConfig,
    ) -> PyResult<Vec<PyDataFrame>> {
        py.allow_threads(
            || match self.inner.search_with_config(&input_terms, &config.inner) {
                Ok(results) => {
                    let py_results = results
                        .into_iter()
                        .map(|df| PyDataFrame(df.into_df()))
                        .collect();
                    Ok(py_results)
                }
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Search with config error: {}",
                    e
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
    ///     List of lists of DataFrames, one per input term set.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search_bulk(
        &self,
        py: Python,
        input_terms: Vec<Vec<String>>,
    ) -> PyResult<Vec<Vec<PyDataFrame>>> {
        py.allow_threads(|| {
            let results = self.inner.search_bulk(&input_terms).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Batch search error: {}",
                    e
                ))
            })?;
            let py_results = results
                .into_iter()
                .map(|df_vec| {
                    df_vec
                        .into_iter()
                        .map(|df| PyDataFrame(df.into_df()))
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
    ///     List of lists of DataFrames, one per input term set.
    ///
    /// Raises:
    ///     RuntimeError: If the search operation fails.
    fn search_bulk_with_config(
        &self,
        py: Python,
        input_terms: Vec<Vec<String>>,
        config: &PySearchConfig,
    ) -> PyResult<Vec<Vec<PyDataFrame>>> {
        py.allow_threads(|| {
            let results = self
                .inner
                .search_bulk_with_config(&input_terms, &config.inner)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Batch search with config error: {}",
                        e
                    ))
                })?;
            let py_results = results
                .into_iter()
                .map(|df_vec| {
                    df_vec
                        .into_iter()
                        .map(|df| PyDataFrame(df.into_df()))
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
        let resolved_results_rust = py.allow_threads(|| {
            self.inner
                .resolve_location::<_, crate::backfill::GenericEntry>(&input_terms)
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
                "Resolve location error: {}",
                e
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
        use crate::core::ResolveSearchConfig;

        let resolve_config = ResolveSearchConfig {
            search_config: config.inner.clone(),
            resolve_config: Default::default(),
        };

        let resolved_results_rust = py.allow_threads(|| {
            self.inner
                .resolve_location_with_config::<_, crate::backfill::GenericEntry>(
                    &input_terms,
                    &resolve_config,
                )
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
                "Resolve location with config error: {}",
                e
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
                .resolve_location_batch::<crate::backfill::GenericEntry, _, _>(&input_terms_batch)
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
                "Resolve location batch error: {}",
                e
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
        use crate::core::ResolveSearchConfig;

        let resolve_config = ResolveSearchConfig {
            search_config: config.inner.clone(),
            resolve_config: Default::default(),
        };

        let resolved_results_rust = py.allow_threads(|| {
            self.inner
                .resolve_location_batch_with_config::<crate::backfill::GenericEntry, _, _>(
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
                "Resolve location batch with config error: {}",
                e
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
