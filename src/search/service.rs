use anyhow::Result;
use polars::prelude::*;
use tracing::{info, info_span};

use super::location_search::{
    admin_search_inner, bulk_location_search_inner, location_search_inner, place_search_inner,
    AdminSearchParams, PlaceSearchParams, SmartFlexibleSearchConfig,
};
use crate::{
    backfill::{
        resolve_search_candidate, resolve_search_candidate_batches, LocationEntry,
        ResolvedSearchResult,
    },
    index::{AdminIndexDef, FTSIndex, PlacesIndexDef},
};

pub struct LocationSearchService {
    admin_fts_index: FTSIndex<AdminIndexDef>,
    admin_data_lf: LazyFrame,
    places_fts_index: FTSIndex<PlacesIndexDef>,
    place_data_lf: LazyFrame,
}

impl LocationSearchService {
    pub fn new(overwrite_fts_indexes: bool) -> Result<Self> {
        info!("Initializing LocationSearchService...");
        let t_init = std::time::Instant::now();

        let (admin_data_lf, place_data_lf) = crate::data::get_data()?;

        let admin_fts_index = {
            let _ = info_span!("load_service_admin_index").entered();
            FTSIndex::new(AdminIndexDef, admin_data_lf.clone(), overwrite_fts_indexes)?
        };
        let places_fts_index = {
            let _ = info_span!("load_service_places_index").entered();
            FTSIndex::new(PlacesIndexDef, place_data_lf.clone(), overwrite_fts_indexes)?
        };

        info!(
            elapsed_seconds = ?t_init.elapsed(),
            "LocationSearchService initialized."
        );
        Ok(Self {
            admin_fts_index,
            admin_data_lf,
            places_fts_index,
            place_data_lf,
        })
    }

    pub fn admin_search(
        &self,
        term: impl AsRef<str>,
        levels: &[u8],
        previous_result: Option<DataFrame>,
        params: &AdminSearchParams,
    ) -> Result<Option<DataFrame>> {
        admin_search_inner(
            term.as_ref(),
            levels,
            &self.admin_fts_index,
            self.admin_data_lf.clone(),
            previous_result,
            params,
        )
    }

    pub fn place_search(
        &self,
        term: impl AsRef<str>,
        previous_result: Option<DataFrame>,
        params: &PlaceSearchParams,
    ) -> Result<Option<DataFrame>> {
        place_search_inner(
            term.as_ref(),
            &self.places_fts_index,
            self.place_data_lf.clone(),
            previous_result,
            params,
        )
    }

    pub fn smart_flexible_search<Term>(
        &self,
        input_terms: &[Term],
        config: &SmartFlexibleSearchConfig,
    ) -> Result<Vec<DataFrame>>
    where
        Term: AsRef<str>,
    {
        let input_terms = input_terms.iter().map(|s| s.as_ref()).collect::<Vec<_>>();

        location_search_inner(
            &input_terms,
            &self.admin_fts_index,
            self.admin_data_lf.clone(),
            &self.places_fts_index,
            self.place_data_lf.clone(),
            config,
        )
    }

    pub fn bulk_smart_flexible_search<Term, Batch>(
        &self,
        all_raw_input_batches: &[Batch],
        config: &SmartFlexibleSearchConfig,
    ) -> Result<Vec<Vec<DataFrame>>>
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
            .map(|inner_vec| inner_vec.as_slice()) // Convert each Vec<&str> to &[&str]
            .collect::<Vec<_>>();

        bulk_location_search_inner(
            &all_raw_input_batches,
            &self.admin_fts_index,
            self.admin_data_lf.clone(),
            &self.places_fts_index,
            self.place_data_lf.clone(),
            config,
        )
    }

    pub fn resolve_locations<Term, Entry>(
        &self,
        input_terms: &[Term],
        config: &SmartFlexibleSearchConfig,
        limit_per_query: usize,
    ) -> Result<Vec<ResolvedSearchResult<Entry>>>
    where
        Term: AsRef<str>,
        Entry: LocationEntry,
    {
        let search_results = self.smart_flexible_search(input_terms.as_ref(), config)?;

        resolve_search_candidate(search_results, &self.admin_data_lf, limit_per_query)
    }

    pub fn resolve_locations_batch<Term, Batch, Entry>(
        &self,
        all_raw_input_batches: &[Batch],
        config: &SmartFlexibleSearchConfig,
        limit_per_query: usize,
    ) -> Result<Vec<Vec<ResolvedSearchResult<Entry>>>>
    where
        Term: AsRef<str>,
        Batch: AsRef<[Term]>,
        Entry: LocationEntry,
    {
        let search_results_batches =
            self.bulk_smart_flexible_search(all_raw_input_batches.as_ref(), config)?;

        resolve_search_candidate_batches(
            search_results_batches,
            &self.admin_data_lf,
            limit_per_query,
        )
    }
}
