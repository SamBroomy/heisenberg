use itertools::Itertools;

mod enrichment;
mod entry;
pub use enrichment::{resolve_search_candidate, resolve_search_candidate_batches};
pub use entry::{GenericEntry, GeonameEntry, LocationEntry};

/// Holds the resolved context for a search result, including admin hierarchy and a potential place.
#[derive(Debug, Clone, Default)]
pub struct LocationContext<E: LocationEntry> {
    pub admin0: Option<E>,
    pub admin1: Option<E>,
    pub admin2: Option<E>,
    pub admin3: Option<E>,
    pub admin4: Option<E>,
    pub place: Option<E>, // The specific place, if the matched entity was a place
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
#[derive(Debug, Clone)]
pub struct ResolvedSearchResult<E: LocationEntry> {
    pub context: LocationContext<E>,
    pub score: f64,
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
