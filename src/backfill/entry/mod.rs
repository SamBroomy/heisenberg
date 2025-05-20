use super::Result;

use polars::prelude::*;
mod basic;
mod generic;

pub use basic::BasicEntry;
pub use generic::GenericEntry;

/// Trait for entities that can be extracted from a search result row.
pub trait LocationEntry: Sized + Default + Clone + Send + Sync + 'static {
    fn from_df(df: &DataFrame) -> Result<Vec<Self>>;
    /// Returns the geonameId of the entity.
    fn geoname_id(&self) -> u32;
    /// Returns the primary name of the entity.
    fn name(&self) -> &str;
    fn field_names() -> Vec<&'static str>;
}

// impl From<GeonameFullEntry> for GeonameEntry {
//     fn from(target_codes: GeonameFullEntry) -> Self {
//         Self {
//             geoname_id: target_codes.geoname_id,
//             name: target_codes.name,
//         }
//     }
// }

// #[derive(Debug, Clone, Serialize, Deserialize, Default)]
// pub struct GeonameFullEntry {
//     pub geoname_id: u32,
//     pub name: String,
//     pub asciiname: String,
//     pub admin_level: u8,
//     pub admin0_code: Option<String>,
//     pub admin1_code: Option<String>,
//     pub admin2_code: Option<String>,
//     pub admin3_code: Option<String>,
//     pub admin4_code: Option<String>,
//     pub feature_class: String,
//     pub feature_code: String,
//     #[serde(rename = "ISO")]
//     pub iso: Option<String>,
//     #[serde(rename = "ISO3")]
//     pub iso3: Option<String>,
//     #[serde(rename = "ISO_Numeric")]
//     pub iso_numeric: Option<u16>,
//     pub official_name: Option<String>,
//     pub fips: Option<String>,
//     pub latitude: f32,
//     pub longitude: f32,
//     pub population: Option<i64>,
//     pub area: Option<f32>,
//     pub alternatenames: Option<String>,
//     pub country_name: Option<String>,
// }

// impl LocationEntry for GeonameFullEntry {

//     fn geoname_id(&self) -> u32 {
//         self.geoname_id
//     }
//     fn name(&self) -> &str {
//         self.name.as_ref()
//     }
//     fn field_names() -> Vec<&'static str> {
//         vec![
//             "geonameId",
//             "name",
//             "asciiname",
//             "admin_level",
//             "admin0_code",
//             "admin1_code",
//             "admin2_code",
//             "admin3_code",
//             "admin4_code",
//             "feature_class",
//             "feature_code",
//             "ISO",
//             "ISO3",
//             "ISO_Numeric",
//             "official_name",
//             "fips",
//             "latitude",
//             "longitude",
//             "population",
//             "area",
//             "alternatenames",
//             "country_name",
//         ]
//     }
// }
