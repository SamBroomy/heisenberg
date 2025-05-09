use anyhow::{Context, Result};
use polars::{frame::row::Row, prelude::*};
use serde::{Deserialize, Serialize};
use tracing::debug; // For potential future use

/// Represents the known administrative codes of a target location to be enriched.
/// These codes are typically extracted from a row of a search result.
#[derive(Debug, Clone, Default)]
pub struct TargetLocationAdminCodes {
    pub admin0_code: Option<String>,
    pub admin1_code: Option<String>,
    pub admin2_code: Option<String>,
    pub admin3_code: Option<String>,
    pub admin4_code: Option<String>,
    // Optionally, you could include the geonameId of the target if known,
    // but it's not strictly needed for the backfill logic itself if codes are present.
    // pub target_geoname_id: Option<u32>,
}

impl TargetLocationAdminCodes {
    /// Creates TargetLocationAdminCodes from a Polars DataFrame row.
    /// Assumes the DataFrame contains columns like "admin0_code", "admin1_code", etc.
    pub fn from_dataframe_row(df: &DataFrame, row_idx: usize) -> Result<Self> {
        let get_opt_string = |col_name: &str| -> Result<Option<String>> {
            df.column(col_name)
                .ok()
                .and_then(|s| s.str().ok())
                .and_then(|ca| ca.get(row_idx).map(|s| s.to_string()))
                .map_or(Ok(None), |v| Ok(Some(v)))
        };

        Ok(Self {
            admin0_code: get_opt_string("admin0_code")?,
            admin1_code: get_opt_string("admin1_code")?,
            admin2_code: get_opt_string("admin2_code")?,
            admin3_code: get_opt_string("admin3_code")?,
            admin4_code: get_opt_string("admin4_code")?,
        })
    }
}

/// Detailed information for a single administrative entity in the hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminHierarchyLevelDetail {
    pub geoname_id: u32,
    pub name: String,
    pub asciiname: Option<String>,
    pub admin_level: u8,
    pub admin0_code: Option<String>,
    pub admin1_code: Option<String>,
    pub admin2_code: Option<String>,
    pub admin3_code: Option<String>,
    pub admin4_code: Option<String>,
    pub feature_class: Option<String>,
    pub feature_code: Option<String>,
    #[serde(rename = "ISO")] // Match parquet/SQL column name if different
    pub iso: Option<String>,
    #[serde(rename = "ISO3")]
    pub iso3: Option<String>,
    #[serde(rename = "ISO_Numeric")]
    pub iso_numeric: Option<u16>, // Assuming USMALLINT maps to u16
    pub official_name: Option<String>,
    pub fips: Option<String>,
    pub latitude: Option<f32>,
    pub longitude: Option<f32>,
    pub population: Option<i64>,
    pub area: Option<f32>,
    pub alternatenames: Option<String>,
    pub country_name: Option<String>,
}

// search_terms_raw
// polars_core::frame::row
// pub struct Row<'a>(pub Vec<AnyValue<'a>>)
impl TryFrom<&Row<'_>> for AdminHierarchyLevelDetail {
    type Error = anyhow::Error;
    fn try_from(row: &Row<'_>) -> Result<Self> {
        Ok(Self {
            geoname_id: row.0[0]
                .try_extract::<u32>()
                .map_err(|_| anyhow::anyhow!("Failed to extract geoname_id"))?,
            name: row.0[1]
                .get_str()
                .ok_or_else(|| anyhow::anyhow!("Failed to extract name as string"))?
                .to_owned(),
            asciiname: row.0[2].get_str().map(|s| s.to_owned()),
            admin_level: row.0[3].try_extract::<u8>().expect("admin_level"),
            admin0_code: row.0[4].get_str().map(|s| s.to_owned()),
            admin1_code: row.0[5].get_str().map(|s| s.to_owned()),
            admin2_code: row.0[6].get_str().map(|s| s.to_owned()),
            admin3_code: row.0[7].get_str().map(|s| s.to_owned()),
            admin4_code: row.0[8].get_str().map(|s| s.to_owned()),
            feature_class: row.0[9].get_str().map(|s| s.to_owned()),
            feature_code: row.0[10].get_str().map(|s| s.to_owned()),
            iso: row.0[11].get_str().map(|s| s.to_owned()),
            iso3: row.0[12].get_str().map(|s| s.to_owned()),
            iso_numeric: row.0[13].try_extract::<u16>().ok(),
            official_name: row.0[14].get_str().map(|s| s.to_owned()),
            fips: row.0[15].get_str().map(|s| s.to_owned()),
            latitude: row.0[16].try_extract::<f32>().ok(),
            longitude: row.0[17].try_extract::<f32>().ok(),
            population: row.0[18].try_extract::<i64>().ok(),
            area: row.0[19].try_extract::<f32>().ok(),
            alternatenames: row.0[20].get_str().map(|s| s.to_owned()),
            country_name: row.0[21].get_str().map(|s| s.to_owned()),
        })
    }
}

/// Represents the fully enriched administrative hierarchy for a location.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FullAdminHierarchy {
    pub level0: Option<AdminHierarchyLevelDetail>,
    pub level1: Option<AdminHierarchyLevelDetail>,
    pub level2: Option<AdminHierarchyLevelDetail>,
    pub level3: Option<AdminHierarchyLevelDetail>,
    pub level4: Option<AdminHierarchyLevelDetail>,
}

/// Fetches the enriched administrative hierarchy for a location identified by its admin codes.
///
/// This function queries the `admin_data_lf` (typically the global "admin_search" table)
/// to find the canonical administrative entities for each level (0 through 4)
/// based on the provided `target_codes`.
pub fn backfill_hierarchy_from_codes(
    target_codes: &TargetLocationAdminCodes,
    admin_data_lf: LazyFrame,
) -> Result<FullAdminHierarchy> {
    let mut hierarchy = FullAdminHierarchy::default();
    let all_needed_columns = &[
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
    ];

    let codes_vec = [
        &target_codes.admin0_code,
        &target_codes.admin1_code,
        &target_codes.admin2_code,
        &target_codes.admin3_code,
        &target_codes.admin4_code,
    ];

    for level_to_fetch in 0..=4 {
        let mut filter_expr = col("admin_level").eq(lit(level_to_fetch as u8));

        // Add filters for all admin codes up to and including the current level_to_fetch
        for (i, code_val) in codes_vec.into_iter().enumerate().take(level_to_fetch + 1) {
            //for i in 0..=level_to_fetch {
            let admin_code_col_name = format!("admin{}_code", i);
            if let Some(code_val) = code_val {
                filter_expr = filter_expr.and(col(&admin_code_col_name).eq(lit(code_val.clone())));
            } else {
                // If a required code for this path is missing in target_codes,
                // we cannot find this level or any deeper levels using this specific path.
                // So, we break and won't find entities for `level_to_fetch` or higher.
                debug!(
                    "Missing target admin code for level {} (column {}), cannot backfill level {} or higher.",
                    i, admin_code_col_name, level_to_fetch
                );
                filter_expr = lit(false); // Ensure no match
                break;
            }
        }
        if matches!(filter_expr, Expr::Literal(LiteralValue::Boolean(false))) {
            break; // Stop if we determined no match is possible
        }

        let result_df = admin_data_lf
            .clone()
            .filter(filter_expr)
            .select(
                all_needed_columns
                    .iter()
                    .map(|s| col(*s))
                    .collect::<Vec<_>>(),
            )
            .limit(1) // Should be unique, but limit 1 for safety
            .collect()
            .with_context(|| format!("Failed to query admin data for level {}", level_to_fetch))?;

        if !result_df.is_empty() {
            // Convert the first row to AdminHierarchyLevelDetail
            // This requires a bit more manual mapping or a derive macro if you use one
            // For simplicity, we'll use the DataFrameRow derive if available and working,
            // otherwise manual construction.
            // Assuming DataFrameRow derive works as expected:
            let level_detail = AdminHierarchyLevelDetail::try_from(&result_df.get_row(0)?)?;

            match level_to_fetch {
                0 => hierarchy.level0 = Some(level_detail),
                1 => hierarchy.level1 = Some(level_detail),
                2 => hierarchy.level2 = Some(level_detail),
                3 => hierarchy.level3 = Some(level_detail),
                4 => hierarchy.level4 = Some(level_detail),
                _ => {} // Should not happen
            }
        } else {
            tracing::debug!("No entity found for admin level {}", level_to_fetch);
        }
    }

    Ok(hierarchy)
}
