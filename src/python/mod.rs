use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::backfill::python_wrappers::*;
use crate::service::Heisenberg as RustHeisenberg;

#[pyclass(name = "Heisenberg")]
#[derive(Clone)]
struct PyHeisenberg {
    inner: RustHeisenberg,
}

#[pymethods]
impl PyHeisenberg {
    #[new]
    fn py_new(py: Python, overwrite_indexes: bool) -> PyResult<Self> {
        py.allow_threads(|| match RustHeisenberg::new(overwrite_indexes) {
            Ok(inner) => Ok(PyHeisenberg { inner }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize Heisenberg: {}",
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

    /// Flexible search with multiple terms
    fn search(&self, py: Python, input_terms: Vec<String>) -> PyResult<Vec<PyDataFrame>> {
        py.allow_threads(|| match self.inner.search(&input_terms) {
            Ok(results) => {
                let py_results = results.into_iter().map(PyDataFrame).collect();
                Ok(py_results)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Search error: {}",
                e
            ))),
        })
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
                .map(|df_vec| df_vec.into_iter().map(PyDataFrame).collect())
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

    // #[pyo3(name = "resolve_location")]
    // fn py_resolve_location<'py>(
    //     &self,
    //     py: Python<'py>,
    //     input_terms: Vec<String>,
    // ) -> PyResult<Vec<Bound<'py, PyAny>>> {
    //     let resolved = py.allow_threads(|| {
    //         self.inner
    //             .resolve_location::<_, crate::GenericEntry>(&input_terms)
    //     });
    //     match resolved {
    //         Ok(resolved) => {
    //             let results = resolved
    //                 .into_iter()
    //                 .map(|result| {
    //                     let p: Bound<'py, PyAny> = pythonize(py, &result).unwrap();
    //                     //let dict = p.downcast_into::<PyDict>().unwrap();

    //                     Ok(p)
    //                 })
    //                 .collect::<PyResult<Vec<_>>>()?;

    //             Ok(results)
    //         }
    //         Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
    //             "Resolve location error: {}",
    //             e
    //         ))),
    //     }
    // }

    // Add more methods as needed
}
//Define the Python module

#[pymodule]
fn heisenberg(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyHeisenberg>()?;
    m.add_class::<BasicEntry>()?;
    m.add_class::<GenericEntry>()?;
    m.add_class::<PyLocationContextBasic>()?;
    m.add_class::<PyLocationContextGeneric>()?;
    m.add_class::<PyResolvedBasicSearchResult>()?;
    m.add_class::<PyResolvedGenericSearchResult>()?;
    Ok(())
}
