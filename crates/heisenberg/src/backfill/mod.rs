use std::fmt;

use error::Result;
use itertools::Itertools;
mod entry;
mod resolve;
pub use entry::{BasicEntry, GenericEntry, LocationEntry, LocationEntryCore};
pub(crate) use error::BackfillError;
pub use resolve::{
    LocationResults, ResolveConfig, resolve_search_candidate, resolve_search_candidate_batches,
};

/// Holds the resolved context for a search result, including administrative hierarchy and place.
///
/// This structure represents the complete geographic context for a location,
/// from country level (admin0) down to local administrative divisions (admin4)
/// and the specific place itself. Not all levels will be populated for every
/// location, depending on the administrative structure of the region.
///
/// # Administrative Levels
///
/// - `admin0`: Country level (e.g., "United States", "France")
/// - `admin1`: State/Province level (e.g., "California", "Île-de-France")
/// - `admin2`: County/Department level (e.g., "Los Angeles County", "Seine")
/// - `admin3`: Local administrative division
/// - `admin4`: Sub-local administrative division
/// - `place`: The specific place (e.g., "Los Angeles", "Paris")
///
/// # Examples
///
/// ```rust
/// use heisenberg::{GenericEntry, LocationEntryCore, LocationSearcher};
///
/// let searcher = LocationSearcher::new(false)?;
/// let results = searcher.resolve_location::<_, GenericEntry>(&["Paris", "France"])?;
///
/// if let Some(result) = results.first() {
///     println!(
///         "Country: {:?}",
///         result.context.admin0.as_ref().map(|e| e.name())
///     );
///     println!(
///         "Place: {:?}",
///         result.context.place.as_ref().map(|e| e.name())
///     );
/// }
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, Default)]
pub struct LocationContext<E: LocationEntry> {
    /// Country level (admin level 0)
    pub admin0: Option<E>,
    /// State/Province level (admin level 1)
    pub admin1: Option<E>,
    /// County/Department level (admin level 2)
    pub admin2: Option<E>,
    /// Local administrative division (admin level 3)
    pub admin3: Option<E>,
    /// Sub-local administrative division (admin level 4)
    pub admin4: Option<E>,
    /// The specific place (city, landmark, etc.)
    pub place: Option<E>,
}

impl<E: LocationEntry + fmt::Display> fmt::Display for LocationContext<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(admin0) = &self.admin0 {
            parts.push(format!("Admin0: {admin0}"));
        }
        if let Some(admin1) = &self.admin1 {
            parts.push(format!("Admin1: {admin1}"));
        }
        if let Some(admin2) = &self.admin2 {
            parts.push(format!("Admin2: {admin2}"));
        }
        if let Some(admin3) = &self.admin3 {
            parts.push(format!("Admin3: {admin3}"));
        }
        if let Some(admin4) = &self.admin4 {
            parts.push(format!("Admin4: {admin4}"));
        }
        if let Some(place) = &self.place {
            parts.push(format!("Place: {place}"));
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
///
/// This is the final output of the location resolution process, containing both
/// the complete administrative hierarchy context and a confidence score.
/// The score represents how well the resolved location matches the input query.
///
/// # Examples
///
/// ```rust
/// use heisenberg::{GenericEntry, LocationEntryCore, LocationSearcher};
///
/// let searcher = LocationSearcher::new(false)?;
/// let results = searcher.resolve_location::<_, GenericEntry>(&["Berlin", "Germany"])?;
///
/// for result in results {
///     println!("Score: {:.2}", result.score);
///     if let Some(place) = &result.context.place {
///         println!("Place: {}", place.name());
///     }
///     if let Some(country) = &result.context.admin0 {
///         println!("Country: {}", country.name());
///     }
/// }
/// # Ok::<(), heisenberg::error::HeisenbergError>(())
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone)]
pub struct ResolvedSearchResult<E: LocationEntry> {
    /// The complete administrative hierarchy and place context
    pub context: LocationContext<E>,
    /// Confidence score for this result (higher is better)
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
    use pyo3::prelude::*;

    pub use super::{
        LocationContext, ResolvedSearchResult,
        entry::{BasicEntry, GenericEntry},
    };
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

                fn __repr__(&self) -> String {
                    format!("{:#?}", self.inner)
                }

                fn __str__(&self) -> String {
                    self.inner.to_string()
                }

                fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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

                fn __repr__(&self) -> String {
                    // To get context repr, you'd need to call __repr__ on the PyLocationContext* object
                    // This requires getting it as a PyObject and then calling its repr.
                    // For simplicity here, we'll just indicate its presence or use its score.
                    format!("{:#?}", self.inner)
                }

                fn __str__(&self) -> String {
                    self.inner.to_string() // Uses the Display trait of ResolvedSearchResult<E>
                }

                fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    macro_rules! impl_python_methods {
        ($struct_name:ident) => {
            #[pyo3::pymethods]
            impl $struct_name {
                fn __repr__(&self) -> String {
                    format!("{:#?}", self)
                }

                fn __str__(&self) -> String {
                    self.to_string()
                }

                fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                    Ok(pythonize::pythonize(py, self)?)
                }
            }
        };
    }

    impl_python_methods!(BasicEntry);
    impl_python_methods!(GenericEntry);
}
