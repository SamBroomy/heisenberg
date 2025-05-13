use anyhow::{Context, Result};
use polars::prelude::*;
use tracing::{debug, info, info_span, warn};

use crate::search::location_search::{
    backfill_administrative_context, AdministrativeContext, ResolvedSearchResult,
};
use crate::search::TargetLocationAdminCodes;

use super::fts_search::{AdminIndexDef, FTSIndex, PlacesIndexDef};
use super::location_search::{
    admin_search_inner, bulk_location_search_inner, get_admin_df, get_places_df,
    location_search_inner, place_search_inner, AdminSearchParams, Entry, PlaceSearchParams,
    SmartFlexibleSearchConfig,
};

pub struct LocationSearchService {
    admin_fts_index: FTSIndex<AdminIndexDef>,
    admin_data_lf: LazyFrame,
    places_fts_index: FTSIndex<PlacesIndexDef>,
    places_data_lf: LazyFrame,
}

impl LocationSearchService {
    pub fn new(overwrite_fts_indexes: bool) -> Result<Self> {
        info!("Initializing LocationSearchService...");
        let t_init = std::time::Instant::now();

        let admin_fts_index = {
            let _ = info_span!("load_service_admin_index").entered();
            FTSIndex::new(AdminIndexDef, overwrite_fts_indexes)?
        };
        let places_fts_index = {
            let _ = info_span!("load_service_places_index").entered();
            FTSIndex::new(PlacesIndexDef, overwrite_fts_indexes)?
        };

        let admin_data_lf = {
            let _ = info_span!("load_service_admin_df").entered();
            get_admin_df()?.clone()
        };
        let places_data_lf = {
            let _ = info_span!("load_service_places_df").entered();
            get_places_df()?.clone()
        };

        info!(
            elapsed_seconds = ?t_init.elapsed(),
            "LocationSearchService initialized."
        );
        Ok(Self {
            admin_fts_index,
            admin_data_lf,
            places_fts_index,
            places_data_lf,
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
            self.places_data_lf.clone(),
            previous_result,
            params,
        )
    }

    pub fn smart_flexible_search(
        &self,
        input_terms: &[impl AsRef<str>],
        config: &SmartFlexibleSearchConfig,
    ) -> Result<Vec<DataFrame>> {
        let input_terms = input_terms.iter().map(|s| s.as_ref()).collect::<Vec<_>>();

        location_search_inner(
            &input_terms,
            &self.admin_fts_index,
            self.admin_data_lf.clone(),
            &self.places_fts_index,
            self.places_data_lf.clone(),
            config,
        )
    }

    pub fn bulk_smart_flexible_search<Batch, Term>(
        &self,
        all_raw_input_batches: &[Batch],
        config: &SmartFlexibleSearchConfig,
    ) -> Result<Vec<Vec<DataFrame>>>
    where
        Batch: AsRef<[Term]> + Sync,
        Term: AsRef<str> + Sync,
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
            self.places_data_lf.clone(),
            config,
        )
    }
}
