use anyhow::{Context, Result};
use itertools::Itertools;
use polars::{frame::row::Row, prelude::*};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

/// Trait for entities that can be extracted from a search result row.
pub trait Entry: Sized + Default + Clone + Send + Sync + 'static {
    /// Tries to create an instance from a Polars DataFrame row.
    fn try_from(row: &Row<'_>) -> Result<Self>;

    fn try_from_struct<'a>(value: &'a [AnyValue<'a>]) -> Result<Self>;
    fn from_dataframe(df: &DataFrame) -> Result<Vec<Self>> {
        df.clone()
            .into_struct("struct".into())
            .into_series()
            .iter()
            .map(|row: AnyValue<'_>| {
                let row_values = row._iter_struct_av().collect::<Vec<_>>();
                Self::try_from_struct(&row_values)
            })
            .collect()
    }
    /// Returns the geonameId of the entity.
    fn geoname_id(&self) -> u32;
    /// Returns the primary name of the entity.
    fn name(&self) -> &str;

    fn field_names() -> Vec<&'static str>;
}

/// Holds the resolved context for a search result, including admin hierarchy and a potential place.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocationContext<E: Entry> {
    pub admin0: Option<E>,
    pub admin1: Option<E>,
    pub admin2: Option<E>,
    pub admin3: Option<E>,
    pub admin4: Option<E>,
    pub place: Option<E>, // The specific place, if the matched entity was a place
}
/// Represents a search result that has been fully resolved and enriched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedSearchResult<E: Entry> {
    pub context: LocationContext<E>,
    pub score: f64,
}

impl<E> ResolvedSearchResult<E>
where
    E: Entry,
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

/// Stores the admin codes for a target location, used to backfill its context.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetLocationAdminCodes {
    pub geoname_id: u64,
    pub admin0_code: Option<String>,
    pub admin1_code: Option<String>,
    pub admin2_code: Option<String>,
    pub admin3_code: Option<String>,
    pub admin4_code: Option<String>,
}

impl TargetLocationAdminCodes {
    pub fn from_row(row: &Row<'_>) -> Result<Self> {
        Ok(Self {
            geoname_id: row.0[0]
                .try_extract::<u64>()
                .with_context(|| "Failed to extract geoname_id")?,
            admin0_code: row.0[1].get_str().map(|s| s.to_owned()),
            admin1_code: row.0[2].get_str().map(|s| s.to_owned()),
            admin2_code: row.0[3].get_str().map(|s| s.to_owned()),
            admin3_code: row.0[4].get_str().map(|s| s.to_owned()),
            admin4_code: row.0[5].get_str().map(|s| s.to_owned()),
        })
    }
}
/// A generic entry struct that can hold common fields from search results.
/// This can be used as the type `E` in `ResolvedSearchResult` and `AdministrativeContext`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenericEntry {
    pub geoname_id: u32,
    pub name: String,
    pub admin_level: u8,
    pub admin0_code: Option<String>,
    pub admin1_code: Option<String>,
    pub admin2_code: Option<String>,
    pub admin3_code: Option<String>,
    pub admin4_code: Option<String>,
    pub feature_code: String,
    pub latitude: Option<f32>,
    pub longitude: Option<f32>,
    pub population: Option<i64>,
}
impl Entry for GenericEntry {
    fn try_from(row: &Row<'_>) -> Result<Self> {
        Ok(Self {
            geoname_id: row.0[0]
                .try_extract::<u32>()
                .with_context(|| "Failed to extract geoname_id")?,
            name: row.0[1]
                .get_str()
                .map(|s| s.to_owned())
                .with_context(|| "Failed to extract name as string")?,
            admin_level: row.0[2]
                .try_extract::<u8>()
                .with_context(|| "Failed to extract admin_level")?,
            admin0_code: row.0[3].get_str().map(|s| s.to_owned()),
            admin1_code: row.0[4].get_str().map(|s| s.to_owned()),
            admin2_code: row.0[5].get_str().map(|s| s.to_owned()),
            admin3_code: row.0[6].get_str().map(|s| s.to_owned()),
            admin4_code: row.0[7].get_str().map(|s| s.to_owned()),
            feature_code: row.0[8]
                .get_str()
                .map(|s| s.to_owned())
                .with_context(|| "Failed to extract feature_code as string")?,
            latitude: row.0[9].try_extract::<f32>().ok(),
            longitude: row.0[10].try_extract::<f32>().ok(),
            population: row.0[11].try_extract::<i64>().ok(),
        })
    }

    fn try_from_struct<'a>(value: &'a [AnyValue<'a>]) -> Result<Self> {
        match value {
            [AnyValue::UInt32(geoname_id), AnyValue::String(name), AnyValue::UInt8(admin_level), AnyValue::String(admin0_code), AnyValue::String(admin1_code), AnyValue::String(admin2_code), AnyValue::String(admin3_code), AnyValue::String(admin4_code), AnyValue::String(feature_code), AnyValue::Float32(latitude), AnyValue::Float32(longitude), AnyValue::Int64(population)] => {
                Ok(Self {
                    geoname_id: *geoname_id,
                    name: name.to_string(),
                    admin_level: *admin_level,
                    admin0_code: Some(admin0_code.to_string()),
                    admin1_code: Some(admin1_code.to_string()),
                    admin2_code: Some(admin2_code.to_string()),
                    admin3_code: Some(admin3_code.to_string()),
                    admin4_code: Some(admin4_code.to_string()),
                    feature_code: feature_code.to_string(),
                    latitude: Some(*latitude),
                    longitude: Some(*longitude),
                    population: Some(*population),
                })
            }
            _ => Err(anyhow::anyhow!("Row does not match expected structure")),
        }
    }

    fn geoname_id(&self) -> u32 {
        self.geoname_id
    }

    fn name(&self) -> &str {
        self.name.as_ref()
    }
    fn field_names() -> Vec<&'static str> {
        vec![
            "geonameId",
            "name",
            "admin_level",
            "admin0_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "feature_code",
            "latitude",
            "longitude",
            "population",
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeonameEntry {
    pub geoname_id: u32,
    pub name: String,
}
impl Entry for GeonameEntry {
    fn try_from(row: &Row<'_>) -> Result<Self> {
        Ok(Self {
            geoname_id: row.0[0]
                .try_extract::<u32>()
                .with_context(|| "Failed to extract geoname_id")?,
            name: row.0[1]
                .get_str()
                .with_context(|| "Failed to extract name as string")?
                .to_owned(),
        })
    }
    fn try_from_struct<'a>(value: &'a [AnyValue<'a>]) -> Result<Self> {
        match value {
            [AnyValue::UInt32(geoname_id), AnyValue::String(name)] => Ok(Self {
                geoname_id: *geoname_id,
                name: name.to_string(),
            }),
            _ => Err(anyhow::anyhow!("Row does not match expected structure")),
        }
    }
    fn geoname_id(&self) -> u32 {
        self.geoname_id
    }
    fn name(&self) -> &str {
        self.name.as_ref()
    }
    fn field_names() -> Vec<&'static str> {
        vec!["geonameId", "name"]
    }
}

impl From<GeonameFullEntry> for GeonameEntry {
    fn from(target_codes: GeonameFullEntry) -> Self {
        Self {
            geoname_id: target_codes.geoname_id,
            name: target_codes.name,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeonameFullEntry {
    pub geoname_id: u32,
    pub name: String,
    pub asciiname: String,
    pub admin_level: u8,
    pub admin0_code: Option<String>,
    pub admin1_code: Option<String>,
    pub admin2_code: Option<String>,
    pub admin3_code: Option<String>,
    pub admin4_code: Option<String>,
    pub feature_class: String,
    pub feature_code: String,
    #[serde(rename = "ISO")]
    pub iso: Option<String>,
    #[serde(rename = "ISO3")]
    pub iso3: Option<String>,
    #[serde(rename = "ISO_Numeric")]
    pub iso_numeric: Option<u16>,
    pub official_name: Option<String>,
    pub fips: Option<String>,
    pub latitude: f32,
    pub longitude: f32,
    pub population: Option<i64>,
    pub area: Option<f32>,
    pub alternatenames: Option<String>,
    pub country_name: Option<String>,
}

impl Entry for GeonameFullEntry {
    fn try_from(row: &Row<'_>) -> Result<Self> {
        Ok(Self {
            geoname_id: row.0[0]
                .try_extract::<u32>()
                .map_err(|_| anyhow::anyhow!("Failed to extract geoname_id"))?,
            name: row.0[1]
                .get_str()
                .ok_or_else(|| anyhow::anyhow!("Failed to extract name as string"))?
                .to_owned(),
            asciiname: row.0[2]
                .get_str()
                .ok_or_else(|| anyhow::anyhow!("Failed to extract asciiname as string"))?
                .to_owned(),
            admin_level: row.0[3].try_extract::<u8>().expect("admin_level"),
            admin0_code: row.0[4].get_str().map(|s| s.to_owned()),
            admin1_code: row.0[5].get_str().map(|s| s.to_owned()),
            admin2_code: row.0[6].get_str().map(|s| s.to_owned()),
            admin3_code: row.0[7].get_str().map(|s| s.to_owned()),
            admin4_code: row.0[8].get_str().map(|s| s.to_owned()),
            feature_class: row.0[9]
                .get_str()
                .ok_or_else(|| anyhow::anyhow!("Failed to extract feature_class as string"))?
                .to_owned(),
            feature_code: row.0[10]
                .get_str()
                .ok_or_else(|| anyhow::anyhow!("Failed to extract feature_code as string"))?
                .to_owned(),
            iso: row.0[11].get_str().map(|s| s.to_owned()),
            iso3: row.0[12].get_str().map(|s| s.to_owned()),
            iso_numeric: row.0[13].try_extract::<u16>().ok(),
            official_name: row.0[14].get_str().map(|s| s.to_owned()),
            fips: row.0[15].get_str().map(|s| s.to_owned()),
            latitude: row.0[16]
                .try_extract::<f32>()
                .with_context(|| "Failed to extract latitude")?,
            longitude: row.0[17]
                .try_extract::<f32>()
                .with_context(|| "Failed to extract longitude")?,
            population: row.0[18].try_extract::<i64>().ok(),
            area: row.0[19].try_extract::<f32>().ok(),
            alternatenames: row.0[20].get_str().map(|s| s.to_owned()),
            country_name: row.0[21].get_str().map(|s| s.to_owned()),
        })
    }
    fn try_from_struct<'a>(value: &'a [AnyValue<'a>]) -> Result<Self> {
        match value {
            [AnyValue::UInt32(geoname_id), AnyValue::String(name), AnyValue::String(asciiname), AnyValue::UInt8(admin_level), AnyValue::String(admin0_code), AnyValue::String(admin1_code), AnyValue::String(admin2_code), AnyValue::String(admin3_code), AnyValue::String(admin4_code), AnyValue::String(feature_class), AnyValue::String(feature_code), AnyValue::String(iso), AnyValue::String(iso3), AnyValue::UInt16(iso_numeric), AnyValue::String(official_name), AnyValue::String(fips), AnyValue::Float32(latitude), AnyValue::Float32(longitude), AnyValue::Int64(population), AnyValue::Float32(area), AnyValue::String(alternatenames), AnyValue::String(country_name)] => {
                Ok(Self {
                    geoname_id: *geoname_id,
                    name: name.to_string(),
                    asciiname: asciiname.to_string(),
                    admin_level: *admin_level,
                    admin0_code: Some(admin0_code.to_string()),
                    admin1_code: Some(admin1_code.to_string()),
                    admin2_code: Some(admin2_code.to_string()),
                    admin3_code: Some(admin3_code.to_string()),
                    admin4_code: Some(admin4_code.to_string()),
                    feature_class: feature_class.to_string(),
                    feature_code: feature_code.to_string(),
                    iso: Some(iso.to_string()),
                    iso3: Some(iso3.to_string()),
                    iso_numeric: Some(*iso_numeric),
                    official_name: Some(official_name.to_string()),
                    fips: Some(fips.to_string()),
                    latitude: *latitude,
                    longitude: *longitude,
                    population: Some(*population),
                    area: Some(*area),
                    alternatenames: Some(alternatenames.to_string()),
                    country_name: Some(country_name.to_string()),
                })
            }
            _ => Err(anyhow::anyhow!("Row does not match expected structure")),
        }
    }
    fn geoname_id(&self) -> u32 {
        self.geoname_id
    }
    fn name(&self) -> &str {
        self.name.as_ref()
    }
    fn field_names() -> Vec<&'static str> {
        vec![
            "geonameId",
            "name",
            "asciiname",
            "admin_level",
            "admin0_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "feature_class",
            "feature_code",
            "ISO",
            "ISO3",
            "ISO_Numeric",
            "official_name",
            "fips",
            "latitude",
            "longitude",
            "population",
            "area",
            "alternatenames",
            "country_name",
        ]
    }
}

// Helper function to query a single admin entity by its codes and level
#[instrument(level = "trace", skip(admin_data_lf, filter_expr), fields(filter = format!("{:?}", filter_expr)))]
fn query_admin_level_entity<E: Entry>(
    admin_data_lf: &LazyFrame,
    filter_expr: Expr,
) -> Result<Option<E>> {
    let result_df = admin_data_lf
        .clone()
        .filter(filter_expr)
        .select(E::field_names().iter().map(|s| col(*s)).collect::<Vec<_>>())
        .limit(1) // Expecting only one match for a full code path at a specific level
        .collect()
        .context("Failed to collect admin entity during backfill")?;

    if result_df.is_empty() {
        Ok(None)
    } else {
        let row = result_df
            .get_row(0)
            .context("Failed to get row from non-empty DataFrame")?;
        E::try_from(&row)
            .map(Some)
            .context("Failed to convert row to Entry type E")
    }
}

#[instrument(level = "trace", skip(admin_data_lf), fields(target_codes = ?target_codes))]
pub fn backfill_administrative_context<E: Entry>(
    target_codes: &TargetLocationAdminCodes,
    admin_data_lf: LazyFrame,
) -> Result<LocationContext<E>> {
    let mut context = LocationContext::<E>::default();
    let codes_hierarchy = [
        (&target_codes.admin0_code, "admin0_code"),
        (&target_codes.admin1_code, "admin1_code"),
        (&target_codes.admin2_code, "admin2_code"),
        (&target_codes.admin3_code, "admin3_code"),
        (&target_codes.admin4_code, "admin4_code"),
    ];
    let mut cumulative_filter_parts: Vec<Expr> = Vec::new();

    for (admin_level_idx, (level_code_opt, level_code_col_name)) in
        codes_hierarchy.iter().enumerate()
    {
        if let Some(current_level_code) = level_code_opt {
            cumulative_filter_parts
                .push(col(*level_code_col_name).eq(lit(current_level_code.clone())));

            let mut level_specific_filter_parts =
                vec![col("admin_level").eq(lit(admin_level_idx as u8))]; // Use admin_level_idx
            level_specific_filter_parts.extend_from_slice(&cumulative_filter_parts);

            let final_filter_for_level = level_specific_filter_parts
                .into_iter()
                .reduce(|acc, expr| acc.and(expr))
                .expect("Filter parts should not be empty if code is present");

            match query_admin_level_entity::<E>(&admin_data_lf, final_filter_for_level) {
                Ok(Some(entity)) => {
                    debug!(
                        level = admin_level_idx,
                        geoname_id = entity.geoname_id(),
                        name = entity.name(),
                        "Found admin entity"
                    );
                    match admin_level_idx {
                        0 => context.admin0 = Some(entity),
                        1 => context.admin1 = Some(entity),
                        2 => context.admin2 = Some(entity),
                        3 => context.admin3 = Some(entity),
                        4 => context.admin4 = Some(entity),
                        _ => unreachable!(),
                    }
                }
                Ok(None) => {
                    warn!(level = admin_level_idx, code = current_level_code, cumulative_filters = ?cumulative_filter_parts, "Admin entity not found for level and codes.");
                }
                Err(e) => {
                    warn!(level = admin_level_idx, code = current_level_code, error = ?e, "Error querying admin entity.");
                }
            }
        } else {
            debug!(
                level = admin_level_idx,
                "No code provided in TargetLocationAdminCodes for this admin level."
            );
        }
    }
    Ok(context)
}

#[instrument(level = "info", skip(search_results_batches, admin_data_lf), fields(num_batches = search_results_batches.len(), limit_per_query))]
pub fn resolve_search_candidates<E: Entry>(
    search_results_batches: Vec<Vec<DataFrame>>,
    admin_data_lf: &LazyFrame,
    limit_per_query: usize,
) -> Result<Vec<Vec<ResolvedSearchResult<E>>>> {
    let mut all_resolved_batches = Vec::with_capacity(search_results_batches.len());

    for (batch_idx, mut result_dfs_for_query) in search_results_batches.into_iter().enumerate() {
        let _batch_span = tracing::info_span!("resolve_batch", batch_idx).entered();
        if result_dfs_for_query.is_empty() {
            debug!("Batch {} is empty, skipping.", batch_idx);
            all_resolved_batches.push(Vec::new());
            continue;
        }

        // The last DataFrame in the list contains the primary candidates
        let primary_candidates_df = match result_dfs_for_query.pop() {
            Some(df) if !df.is_empty() => df,
            _ => {
                debug!(
                    "Batch {} has no suitable primary candidate DataFrame or it's empty.",
                    batch_idx
                );
                all_resolved_batches.push(Vec::new());
                continue;
            }
        };

        debug!(
            "Processing {} candidates from primary DataFrame for batch {}.",
            primary_candidates_df.height(),
            batch_idx
        );

        let mut resolved_for_this_query = Vec::new();
        let num_candidates_to_process =
            std::cmp::min(primary_candidates_df.height(), limit_per_query);

        // TODO: Fix this as its brittle. Needs data to be in right order. Maybe do a select on the target df.
        // And then loop over the zipped columns rather than the rows.
        for i in 0..num_candidates_to_process {
            // 1. Extract TargetLocationAdminCodes
            let target_codes = match TargetLocationAdminCodes::from_row(
                &primary_candidates_df
                    .select([
                        "geonameId",
                        "admin0_code",
                        "admin1_code",
                        "admin2_code",
                        "admin3_code",
                        "admin4_code",
                    ]).with_context(|| format!(
                        "Failed to select columns for TargetLocationAdminCodes from row {} in batch {}",
                        i, batch_idx
                    ))?
                    .get_row(i)
                    .with_context(|| format!(
                        "Failed to get row {} for TargetLocationAdminCodes from primary_candidates_df in batch {}",
                        i, batch_idx
                    ))?,
            ) {
                Ok(tc) => tc,
                Err(e) => {
                    warn!("Failed to create TargetLocationAdminCodes for row {} in batch {}: {:?}. Skipping row.", i, batch_idx, e);
                    continue;
                }
            };
            let candidate_row_result = primary_candidates_df.get_row(i);
            let candidate_row = match candidate_row_result {
                Ok(r) => r,
                Err(e) => {
                    warn!("Failed to get row {} from primary_candidates_df for batch {}: {:?}. Skipping row.", i, batch_idx, e);
                    continue;
                }
            };

            // 2. Create Matched Entity
            let primary_candidate_entity = match E::try_from(&candidate_row) {
                Ok(me) => me,
                Err(e) => {
                    warn!("Failed to create Primary Candidate Entity (type E) for row {} in batch {}: {:?}. Skipping row.", i, batch_idx, e);
                    continue;
                }
            };

            // 3. Extract Score
            let score_value = candidate_row
                .0
                .last() // Assuming the last column is the score
                .and_then(|av| av.try_extract::<f64>().ok())
                .with_context(|| "Failed to extract score from row")?;

            // 4. Backfill Administrative Context
            // Note: backfill_administrative_context takes owned LazyFrame, so clone admin_data_lf
            let mut final_context = match backfill_administrative_context::<E>(
                &target_codes,
                admin_data_lf.clone(),
            ) {
                Ok(ac) => ac,
                Err(e) => {
                    warn!(
                        "Failed to backfill context for geonameId {} in batch {}: {:?}. Proceeding with empty context.",
                        target_codes.geoname_id, batch_idx, e
                    );
                    LocationContext::<E>::default() // Use default empty context on error
                }
            };

            // Determine if primary_candidate_entity is a "place"
            // This requires GeonameFullEntry to have a feature_class field.
            // Let's assume E is GeonameFullEntry for this logic.
            // A more generic way would be a trait method on E like `is_place_like()`.
            let feature_class_str = candidate_row
                .0
                .get(8) // Assuming feature_class is at index 9
                .and_then(|av| av.get_str())
                .map(|s| s.to_owned());

            if let Some(fc) = feature_class_str {
                // Define what constitutes a "place" based on feature class
                // P: populated place. Others like S (spot), H (hydrographic), L (area/park) could also be considered.
                if fc == "P"
                    || fc == "S"
                    || fc == "H"
                    || fc == "L"
                    || fc == "T"
                    || fc == "V"
                    || fc == "R"
                    || fc == "U"
                {
                    // If the primary entity is a place, ensure it's not also one of the admin entities
                    // by geoname_id to avoid duplication in the `full()` display if it happened to be an admin unit
                    // that is also a PPL (e.g. a city that is its own admin level).
                    // This check might be overly cautious if backfill_administrative_context is robust.
                    let is_already_in_admin_context =
                        final_context.admin0.as_ref().is_some_and(|a| {
                            a.geoname_id() == primary_candidate_entity.geoname_id()
                        }) || final_context.admin1.as_ref().is_some_and(|a| {
                            a.geoname_id() == primary_candidate_entity.geoname_id()
                        }) || final_context.admin2.as_ref().is_some_and(|a| {
                            a.geoname_id() == primary_candidate_entity.geoname_id()
                        }) || final_context.admin3.as_ref().is_some_and(|a| {
                            a.geoname_id() == primary_candidate_entity.geoname_id()
                        }) || final_context.admin4.as_ref().is_some_and(|a| {
                            a.geoname_id() == primary_candidate_entity.geoname_id()
                        });

                    if !is_already_in_admin_context {
                        final_context.place = Some(primary_candidate_entity);
                    } else {
                        // The primary entity was a place-like feature class but it's already represented
                        // as one of the admin levels (e.g. a city that is its own ADM3).
                        // In this case, we don't need to set it in `place` again.
                        // The `backfill_administrative_context` should have placed it correctly.
                        debug!("Primary entity (geonameId: {}) with place-like feature class '{}' is already in admin context. Not setting context.place.", primary_candidate_entity.geoname_id(), fc);
                    }
                }
                // If feature_class is 'A', primary_candidate_entity is an admin unit.
                // It should have been populated into one of the admin0-4 slots by backfill_administrative_context.
                // So, final_context.place remains None.
            } else {
                // If no feature_class, assume it's not a place for this purpose, or handle as error/default
                warn!("Missing feature_class for geonameId {} in batch {}. Cannot determine if it's a place.", target_codes.geoname_id, batch_idx);
            }

            resolved_for_this_query.push(ResolvedSearchResult {
                context: final_context,
                score: score_value,
            });
        }
        all_resolved_batches.push(resolved_for_this_query);
    }

    all_resolved_batches.iter_mut().for_each(|resolved_batch| {
        resolved_batch.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    });

    Ok(all_resolved_batches)
}
