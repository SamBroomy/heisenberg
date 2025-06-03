use crate::data::LocationSearchData;
use crate::{ResolveConfig, error::HeisenbergError};
use polars::prelude::*;
use tracing::{info, info_span};

use crate::search::{
    AdminSearchParams, PlaceSearchParams, SearchConfig, SearchResult, admin_search_inner,
    bulk_location_search_inner, location_search_inner, place_search_inner,
};
use crate::{
    backfill::{
        LocationEntry, LocationResults, resolve_search_candidate, resolve_search_candidate_batches,
    },
    index::{AdminIndexDef, FTSIndex, PlacesIndexDef},
};

pub type SearchResults = Vec<SearchResult>;
pub type SearchResultsBatch = Vec<Vec<SearchResult>>;

#[derive(Debug, Clone, Default)]
pub struct ResolveSearchConfig {
    pub search_config: SearchConfig,
    pub resolve_config: ResolveConfig,
}

#[derive(Clone)]
pub struct LocationSearcher {
    admin_fts_index: FTSIndex<AdminIndexDef>,
    places_fts_index: FTSIndex<PlacesIndexDef>,
    data: LocationSearchData,
}

impl LocationSearcher {
    pub fn new(overwrite_fts_indexes: bool) -> Result<Self, HeisenbergError> {
        info!("Initializing LocationSearchService...");
        let t_init = std::time::Instant::now();

        let data = crate::data::LocationSearchData::new()?;

        let admin_fts_index = {
            let _ = info_span!("load_service_admin_index").entered();
            FTSIndex::new(
                AdminIndexDef,
                data.admin_search_df()?.clone(),
                overwrite_fts_indexes,
            )?
        };
        let places_fts_index = {
            let _ = info_span!("load_service_places_index").entered();
            FTSIndex::new(
                PlacesIndexDef,
                data.place_search_df()?.clone(),
                overwrite_fts_indexes,
            )?
        };

        info!(
            elapsed_seconds = ?t_init.elapsed(),
            "LocationSearchService initialized."
        );
        Ok(Self {
            admin_fts_index,
            places_fts_index,
            data,
        })
    }
    // === Low-level searches (unchanged but with custom return types) ===
    pub fn admin_search(
        &self,
        term: impl AsRef<str>,
        levels: &[u8],
        previous_result: Option<DataFrame>,
        params: &AdminSearchParams,
    ) -> Result<Option<DataFrame>, HeisenbergError> {
        admin_search_inner(
            term.as_ref(),
            levels,
            &self.admin_fts_index,
            self.data.admin_search_df()?.clone(),
            previous_result.map(From::from),
            params,
        )
        .map(|result| result.map(From::from))
        .map_err(From::from)
    }

    pub fn place_search(
        &self,
        term: impl AsRef<str>,
        previous_result: Option<DataFrame>,
        params: &PlaceSearchParams,
    ) -> Result<Option<DataFrame>, HeisenbergError> {
        place_search_inner(
            term.as_ref(),
            &self.places_fts_index,
            self.data.place_search_df()?.clone(),
            previous_result.map(From::from),
            params,
        )
        .map(|result| result.map(From::from))
        .map_err(From::from)
    }

    // === Simplified mid-level search methods ===

    pub fn search<Term>(&self, input_terms: &[Term]) -> Result<SearchResults, HeisenbergError>
    where
        Term: AsRef<str>,
    {
        self.search_with_config(input_terms, &SearchConfig::default())
    }

    pub fn search_with_config<Term>(
        &self,
        input_terms: &[Term],
        config: &SearchConfig,
    ) -> Result<SearchResults, HeisenbergError>
    where
        Term: AsRef<str>,
    {
        let input_terms = input_terms.iter().map(|s| s.as_ref()).collect::<Vec<_>>();

        location_search_inner(
            &input_terms,
            &self.admin_fts_index,
            self.data.admin_search_df()?.clone(),
            &self.places_fts_index,
            self.data.place_search_df()?.clone(),
            config,
        )
        .map_err(From::from)
    }

    pub fn search_bulk<Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
    ) -> Result<SearchResultsBatch, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
    {
        self.search_bulk_with_config(all_raw_input_batches, &SearchConfig::default())
    }

    pub fn search_bulk_with_config<Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
        config: &SearchConfig,
    ) -> Result<SearchResultsBatch, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
    {
        let all_raw_input_batches = all_raw_input_batches
            .iter()
            .map(|batch| {
                batch
                    .as_ref()
                    .iter()
                    .map(|term| term.as_ref())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let all_raw_input_batches = all_raw_input_batches
            .iter()
            .map(|inner_vec| inner_vec.as_slice())
            .collect::<Vec<_>>();

        bulk_location_search_inner(
            &all_raw_input_batches,
            &self.admin_fts_index,
            self.data.admin_search_df()?.clone(),
            &self.places_fts_index,
            self.data.place_search_df()?.clone(),
            config,
        )
        .map_err(From::from)
    }

    pub fn resolve<Entry: LocationEntry>(
        &self,
        search_results: SearchResults,
    ) -> Result<LocationResults<Entry>, HeisenbergError> {
        self.resolve_with_config(search_results, &ResolveConfig::default())
    }
    pub fn resolve_with_config<Entry: LocationEntry>(
        &self,
        search_results: SearchResults,
        config: &ResolveConfig,
    ) -> Result<LocationResults<Entry>, HeisenbergError> {
        resolve_search_candidate(
            search_results,
            &self.data.admin_search_df()?.clone(),
            config,
        )
        .map_err(From::from)
    }
    pub fn resolve_batch<Entry: LocationEntry>(
        &self,
        search_results_batches: SearchResultsBatch,
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError> {
        self.resolve_batch_with_config(search_results_batches, &ResolveConfig::default())
    }
    pub fn resolve_batch_with_config<Entry: LocationEntry>(
        &self,
        search_results_batches: SearchResultsBatch,
        config: &ResolveConfig,
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError> {
        resolve_search_candidate_batches(
            search_results_batches,
            &self.data.admin_search_df()?.clone(),
            config,
        )
        .map_err(From::from)
    }

    // === High-level search and resolve methods ===

    pub fn resolve_location<Term, Entry>(
        &self,
        input_terms: &[Term],
    ) -> Result<LocationResults<Entry>, HeisenbergError>
    where
        Term: AsRef<str>,
        Entry: LocationEntry,
    {
        self.resolve_location_with_config(input_terms, &ResolveSearchConfig::default())
    }

    pub fn resolve_location_with_config<Term, Entry>(
        &self,
        input_terms: &[Term],
        config: &ResolveSearchConfig,
    ) -> Result<LocationResults<Entry>, HeisenbergError>
    where
        Term: AsRef<str>,
        Entry: LocationEntry,
    {
        let search_results = self.search_with_config(input_terms, &config.search_config)?;

        resolve_search_candidate(
            search_results,
            &self.data.admin_search_df()?.clone(),
            &config.resolve_config,
        )
        .map_err(From::from)
    }

    pub fn resolve_location_batch<Entry, Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
        Entry: LocationEntry,
    {
        self.resolve_location_batch_with_config(
            all_raw_input_batches,
            &ResolveSearchConfig::default(),
        )
    }

    pub fn resolve_location_batch_with_config<Entry, Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
        config: &ResolveSearchConfig,
    ) -> Result<Vec<LocationResults<Entry>>, HeisenbergError>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
        Entry: LocationEntry,
    {
        let search_results_batches =
            self.search_bulk_with_config(all_raw_input_batches, &config.search_config)?;

        resolve_search_candidate_batches(
            search_results_batches,
            &self.data.admin_search_df()?.clone(),
            &config.resolve_config,
        )
        .map_err(From::from)
    }
}
