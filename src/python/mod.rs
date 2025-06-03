use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::backfill::python_wrappers::*;
use crate::config::SearchConfigBuilder;
use crate::core::LocationSearcher;
use crate::search::SearchConfig;

#[pyclass(name = "LocationSearcher")]
#[derive(Clone)]
struct PyLocationSearcher {
    inner: LocationSearcher,
}

#[pymethods]
impl PyLocationSearcher {
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

    /// Search for administrative entities
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
            // Convert Python optional DataFrame to Rust optional DataFrame

            let prev_df = previous_result.map(|df| df.into());

            // Call the Rust function
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

    /// Search for places
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

    /// Flexible search with multiple terms
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

    /// Flexible search with custom configuration
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

    /// Batch search with custom configuration
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

    #[pyo3(name = "resolve_location", signature = (input_terms))]
    fn resolve_location(
        &self,
        py: Python,
        input_terms: Vec<String>,
    ) -> PyResult<Vec<PyResolvedGenericSearchResult>> {
        // Return the concrete Python type
        let resolved_results_rust = py.allow_threads(|| {
            self.inner
                .resolve_location::<_, crate::backfill::GenericEntry>(&input_terms)
        });

        match resolved_results_rust {
            Ok(resolved_vec) => {
                // Convert Vec<ResolvedSearchResult<GenericEntry>> to Vec<PyResolvedGenericSearchResult>
                let py_results = resolved_vec
                    .into_iter()
                    .map(PyResolvedGenericSearchResult::from) // Use the From trait
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Resolve location error: {}",
                e
            ))),
        }
    }

    /// Resolve location with custom configuration
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

    /// Resolve locations in batch
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

    /// Resolve locations in batch with custom configuration
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

/// Python wrapper for SearchConfig
#[pyclass(name = "SearchConfig")]
#[derive(Clone)]
struct PySearchConfig {
    inner: SearchConfig,
}

#[pymethods]
impl PySearchConfig {
    #[new]
    fn py_new() -> Self {
        PySearchConfig {
            inner: SearchConfig::default(),
        }
    }
}

/// Python wrapper for SearchConfigBuilder
#[pyclass(name = "SearchConfigBuilder")]
#[derive(Clone)]
struct PySearchConfigBuilder {
    inner: SearchConfigBuilder,
}

#[pymethods]
impl PySearchConfigBuilder {
    #[new]
    fn py_new() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::new(),
        }
    }

    /// Create a fast preset configuration
    #[staticmethod]
    fn fast() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::fast(),
        }
    }

    /// Create a comprehensive preset configuration
    #[staticmethod]
    fn comprehensive() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::comprehensive(),
        }
    }

    /// Create a quality places preset configuration
    #[staticmethod]
    fn quality_places() -> Self {
        PySearchConfigBuilder {
            inner: SearchConfigBuilder::quality_places(),
        }
    }

    /// Set result limit
    fn limit(mut slf: PyRefMut<Self>, limit: usize) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().limit(limit);
        slf
    }

    /// Set place importance threshold
    fn place_importance_threshold(mut slf: PyRefMut<Self>, threshold: u8) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().place_importance_threshold(threshold);
        slf
    }

    /// Enable or disable proactive admin search
    fn proactive_admin_search(mut slf: PyRefMut<Self>, enabled: bool) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().proactive_admin_search(enabled);
        slf
    }

    /// Set max admin terms
    fn max_admin_terms(mut slf: PyRefMut<Self>, max_terms: usize) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().max_admin_terms(max_terms);
        slf
    }

    /// Include all columns in results
    fn include_all_columns(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().include_all_columns();
        slf
    }

    /// Configure text search parameters
    fn text_search(
        mut slf: PyRefMut<Self>,
        fuzzy: bool,
        limit_multiplier: usize,
    ) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().text_search(fuzzy, limit_multiplier);
        slf
    }

    /// Build the final configuration
    fn build(&self) -> PySearchConfig {
        PySearchConfig {
            inner: self.inner.clone().build(),
        }
    }
}

//Define the Python module

#[pymodule]
fn heisenberg(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyLocationSearcher>()?;
    m.add_class::<PySearchConfig>()?;
    m.add_class::<PySearchConfigBuilder>()?;
    m.add_class::<BasicEntry>()?;
    m.add_class::<GenericEntry>()?;
    m.add_class::<PyLocationContextBasic>()?;
    m.add_class::<PyLocationContextGeneric>()?;
    m.add_class::<PyResolvedBasicSearchResult>()?;
    m.add_class::<PyResolvedGenericSearchResult>()?;
    Ok(())
}
