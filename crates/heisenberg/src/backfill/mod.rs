use std::fmt;

use error::Result;
use itertools::Itertools;
mod entry;
mod resolve;
pub use entry::LocationEntry;
pub use error::BackfillError;
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
/// use heisenberg::{LocationEntry, LocationSearcher};
///
/// let searcher = LocationSearcher::new_embedded()?;
/// let results = searcher.resolve_location(&["Paris", "France"])?;
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
pub struct LocationContext {
    /// Country level (admin level 0)
    pub admin0: Option<LocationEntry>,
    /// State/Province level (admin level 1)
    pub admin1: Option<LocationEntry>,
    /// County/Department level (admin level 2)
    pub admin2: Option<LocationEntry>,
    /// Local administrative division (admin level 3)
    pub admin3: Option<LocationEntry>,
    /// Sub-local administrative division (admin level 4)
    pub admin4: Option<LocationEntry>,
    /// The specific place (city, landmark, etc.)
    pub place: Option<LocationEntry>,
}

impl fmt::Display for LocationContext {
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

impl LocationContext {
    fn candidate_already_in_context(&self, candidate: &LocationEntry) -> bool {
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
/// use heisenberg::{LocationEntry, LocationSearcher};
///
/// let searcher = LocationSearcher::new_embedded()?;
/// let results = searcher.resolve_location(&["Berlin", "Germany"])?;
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
pub struct ResolvedSearchResult {
    /// The complete administrative hierarchy and place context
    pub context: LocationContext,
    /// Confidence score for this result (higher is better)
    pub score: f64,
}

impl fmt::Display for ResolvedSearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResolvedSearchResult {{ score: {:.4}, context: {} }}",
            self.score, self.context
        )
    }
}

impl ResolvedSearchResult {
    #[must_use]
    pub fn simple(&self) -> Vec<String> {
        self.full().into_iter().flatten().unique().collect()
    }

    #[must_use]
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
