use std::fmt;

use error::Result;
use itertools::Itertools;
mod entry;
mod resolve;
pub use entry::{BasicEntry, GenericEntry, LocationEntry};
pub(crate) use error::BackfillError;
pub use resolve::{
    LocationResults, ResolveConfig, resolve_search_candidate, resolve_search_candidate_batches,
};

/// Holds the resolved context for a search result, including admin hierarchy and a potential place.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Default)]
pub struct LocationContext<E: LocationEntry> {
    pub admin0: Option<E>,
    pub admin1: Option<E>,
    pub admin2: Option<E>,
    pub admin3: Option<E>,
    pub admin4: Option<E>,
    pub place: Option<E>, // The specific place, if the matched entity was a place
}

impl<E: LocationEntry + fmt::Display> fmt::Display for LocationContext<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(admin0) = &self.admin0 {
            parts.push(format!("Admin0: {}", admin0));
        }
        if let Some(admin1) = &self.admin1 {
            parts.push(format!("Admin1: {}", admin1));
        }
        if let Some(admin2) = &self.admin2 {
            parts.push(format!("Admin2: {}", admin2));
        }
        if let Some(admin3) = &self.admin3 {
            parts.push(format!("Admin3: {}", admin3));
        }
        if let Some(admin4) = &self.admin4 {
            parts.push(format!("Admin4: {}", admin4));
        }
        if let Some(place) = &self.place {
            parts.push(format!("Place: {}", place));
        }

        if parts.is_empty() {
            write!(f, "LocationContext {{ Empty }}")
        } else {
            write!(f, "LocationContext {{\n  {}\n}}", parts.join(",\n  "))
        }
    }
}

impl<E: LocationEntry> LocationContext<E> {
    fn candidate_already_in_context(&self, candidate: &E) -> bool {
        self.admin0
            .as_ref()
            .is_some_and(|e| e.geoname_id() == candidate.geoname_id())
            || self
                .admin1
                .as_ref()
                .is_some_and(|e| e.geoname_id() == candidate.geoname_id())
            || self
                .admin2
                .as_ref()
                .is_some_and(|e| e.geoname_id() == candidate.geoname_id())
            || self
                .admin3
                .as_ref()
                .is_some_and(|e| e.geoname_id() == candidate.geoname_id())
            || self
                .admin4
                .as_ref()
                .is_some_and(|e| e.geoname_id() == candidate.geoname_id())
    }
}

/// Represents a search result that has been fully resolved and enriched.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct ResolvedSearchResult<E: LocationEntry> {
    pub context: LocationContext<E>,
    pub score: f64,
}

impl<E: LocationEntry + fmt::Display> fmt::Display for ResolvedSearchResult<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResolvedSearchResult {{ score: {:.4}, context: {} }}",
            self.score, self.context
        )
    }
}

impl<E> ResolvedSearchResult<E>
where
    E: LocationEntry,
{
    pub fn simple(&self) -> Vec<String> {
        self.full().into_iter().flatten().unique().collect()
    }

    pub fn full(&self) -> [Option<String>; 6] {
        [
            self.context.admin0.as_ref().map(|e| e.name().to_string()),
            self.context.admin1.as_ref().map(|e| e.name().to_string()),
            self.context.admin2.as_ref().map(|e| e.name().to_string()),
            self.context.admin3.as_ref().map(|e| e.name().to_string()),
            self.context.admin4.as_ref().map(|e| e.name().to_string()),
            self.context.place.as_ref().map(|e| e.name().to_string()),
        ]
    }
}

mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum BackfillError {
        #[error("DataFrame error: {0}")]
        DataFrame(#[from] polars::prelude::PolarsError),
    }
    pub type Result<T> = std::result::Result<T, BackfillError>;
}
#[cfg(feature = "python")]
pub mod python_wrappers {
    pub use super::{
        LocationContext, ResolvedSearchResult,
        entry::{BasicEntry, GenericEntry},
    };
    use pyo3::prelude::*;
    // Macro for LocationContext<E>

    // Macro for LocationContext<E>
    macro_rules! define_py_location_context {
        ($py_context_name:ident, $rust_entry_type:ty, $py_entry_type_name_str:expr) => {
            #[pyclass(name = $py_entry_type_name_str)]
            #[cfg_attr(feature = "serde", derive(serde::Serialize))]
            #[derive(Debug, Clone)]
            pub struct $py_context_name {
                inner: LocationContext<$rust_entry_type>,
            }

            #[pymethods]
            impl $py_context_name {
                #[getter]
                fn get_admin0(&self, py: Python) -> PyResult<Option<Py<$rust_entry_type>>> {
                    // Return Py<ConcreteType>
                    self.inner
                        .admin0
                        .as_ref()
                        .map(|e| Py::new(py, e.clone())) // Pass the concrete #[pyclass] type
                        .transpose()
                }
                #[getter]
                fn get_admin1(&self, py: Python) -> PyResult<Option<Py<$rust_entry_type>>> {
                    self.inner
                        .admin1
                        .as_ref()
                        .map(|e| Py::new(py, e.clone()))
                        .transpose()
                }
                #[getter]
                fn get_admin2(&self, py: Python) -> PyResult<Option<Py<$rust_entry_type>>> {
                    self.inner
                        .admin2
                        .as_ref()
                        .map(|e| Py::new(py, e.clone()))
                        .transpose()
                }
                #[getter]
                fn get_admin3(&self, py: Python) -> PyResult<Option<Py<$rust_entry_type>>> {
                    self.inner
                        .admin3
                        .as_ref()
                        .map(|e| Py::new(py, e.clone()))
                        .transpose()
                }
                #[getter]
                fn get_admin4(&self, py: Python) -> PyResult<Option<Py<$rust_entry_type>>> {
                    self.inner
                        .admin4
                        .as_ref()
                        .map(|e| Py::new(py, e.clone()))
                        .transpose()
                }
                #[getter]
                fn get_place(&self, py: Python) -> PyResult<Option<Py<$rust_entry_type>>> {
                    self.inner
                        .place
                        .as_ref()
                        .map(|e| Py::new(py, e.clone()))
                        .transpose()
                }

                fn __repr__(&self, py: Python) -> String {
                    format!("{:#?}", self.inner)
                }
                fn __str__(&self) -> String {
                    self.inner.to_string()
                }
                fn to_dict<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
                    Ok(pythonize::pythonize(py, &self.inner)?)
                }
            }

            impl From<LocationContext<$rust_entry_type>> for $py_context_name {
                fn from(context: LocationContext<$rust_entry_type>) -> Self {
                    Self { inner: context }
                }
            }
        };
    }

    // Macro for ResolvedSearchResult<E>
    macro_rules! define_py_resolved_search_result {
        ($py_wrapper_name:ident, $rust_entry_type:ty, $py_context_type:ty, $py_class_name_str:expr) => {
            #[pyclass(name = $py_class_name_str)]
            #[cfg_attr(feature = "serde", derive(serde::Serialize))]
            #[derive(Debug, Clone)]
            pub struct $py_wrapper_name {
                inner: ResolvedSearchResult<$rust_entry_type>,
            }

            #[pymethods]
            impl $py_wrapper_name {
                #[getter]
                fn get_context(&self) -> $py_context_type {
                    self.inner.context.clone().into()
                }

                #[getter]
                fn get_score(&self) -> f64 {
                    self.inner.score
                }

                pub fn simple(&self) -> Vec<String> {
                    self.inner.simple()
                }

                pub fn full(&self) -> [Option<String>; 6] {
                    self.inner.full()
                }

                fn __repr__(&self, py: Python) -> String {
                    // To get context repr, you'd need to call __repr__ on the PyLocationContext* object
                    // This requires getting it as a PyObject and then calling its repr.
                    // For simplicity here, we'll just indicate its presence or use its score.
                    format!("{:#?}", self.inner)
                }
                fn __str__(&self) -> String {
                    self.inner.to_string() // Uses the Display trait of ResolvedSearchResult<E>
                }

                fn to_dict<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
                    Ok(pythonize::pythonize(py, &self.inner)?)
                }
            }

            impl From<ResolvedSearchResult<$rust_entry_type>> for $py_wrapper_name {
                fn from(result: ResolvedSearchResult<$rust_entry_type>) -> Self {
                    Self { inner: result }
                }
            }
        };
    }

    // Instantiate the macros
    define_py_location_context!(PyLocationContextBasic, BasicEntry, "LocationContextBasic");
    define_py_location_context!(
        PyLocationContextGeneric,
        GenericEntry,
        "LocationContextGeneric"
    );

    define_py_resolved_search_result!(
        PyResolvedBasicSearchResult,
        BasicEntry,
        PyLocationContextBasic,
        "ResolvedBasicSearchResult"
    );
    define_py_resolved_search_result!(
        PyResolvedGenericSearchResult,
        GenericEntry,
        PyLocationContextGeneric,
        "ResolvedGenericSearchResult"
    );
}
