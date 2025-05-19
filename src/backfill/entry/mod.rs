use super::Result;
use itertools::izip;
use polars::prelude::*;

/// Trait for entities that can be extracted from a search result row.
pub trait LocationEntry: Sized + Default + Clone + Send + Sync + 'static {
    fn from_df(df: &DataFrame) -> Result<Vec<Self>>;
    /// Returns the geonameId of the entity.
    fn geoname_id(&self) -> u32;
    /// Returns the primary name of the entity.
    fn name(&self) -> &str;
    fn field_names() -> Vec<&'static str>;
}

/// A generic entry struct that can hold common fields from search results.
/// This can be used as the type `E` in `ResolvedSearchResult` and `AdministrativeContext`.
#[derive(Debug, Clone, Default)]
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
impl LocationEntry for GenericEntry {
    fn from_df(df: &DataFrame) -> Result<Vec<Self>> {
        let cols = df.select(Self::field_names())?.take_columns();

        // For some reason izip! can only handle 9 args so here we need to do a wierd zip thing and unpack slightly strangely to get it working.
        Ok(izip!(
            cols[0].u32()?,
            cols[1].str()?,
            cols[2].u8()?,
            cols[3].str()?,
            cols[4].str()?,
            cols[5].str()?,
            cols[6].str()?,
            cols[7].str()?,
            cols[8].str()?,
        )
        .zip(
            cols[9]
                .f32()?
                .iter()
                .zip(cols[10].f32()?.iter().zip(cols[11].i64()?.iter())),
        )
        .map(
            |(
                (
                    geoname_id,
                    name,
                    admin_level,
                    admin0_code,
                    admin1_code,
                    admin2_code,
                    admin3_code,
                    admin4_code,
                    feature_code,
                ),
                (latitude, (longitude, population)),
            )| Self {
                geoname_id: geoname_id.expect("geonameId should never be None"),
                name: name.expect("name should never be None").to_string(),
                admin_level: admin_level.expect("admin_level should never be None"),
                admin0_code: admin0_code.map(|s| s.to_string()),
                admin1_code: admin1_code.map(|s| s.to_string()),
                admin2_code: admin2_code.map(|s| s.to_string()),
                admin3_code: admin3_code.map(|s| s.to_string()),
                admin4_code: admin4_code.map(|s| s.to_string()),
                feature_code: feature_code
                    .expect("feature_code should never be None")
                    .to_string(),
                latitude,
                longitude,
                population,
            },
        )
        .collect())
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

#[derive(Debug, Clone, Default)]
pub struct GeonameEntry {
    pub geoname_id: u32,
    pub name: String,
}
impl LocationEntry for GeonameEntry {
    fn from_df(df: &DataFrame) -> Result<Vec<Self>> {
        let cols = df.select(Self::field_names())?.take_columns();

        Ok(izip!(cols[0].u32()?, cols[1].str()?,)
            .map(|(geoname_id, name)| Self {
                geoname_id: geoname_id.expect("geonameId should not be None"),
                name: name.expect("name should not be None").to_string(),
            })
            .collect())
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
