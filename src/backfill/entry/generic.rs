use std::fmt;

use super::{LocationEntryCore, Result};
use itertools::izip;
use polars::prelude::*;

/// A generic entry struct that can hold common fields from search results.
/// This can be used as the type `E` in `ResolvedSearchResult` and `AdministrativeContext`.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, name = "GenericEntry"))]
#[derive(Debug, Clone, Default)]
pub struct GenericEntry {
    pub geoname_id: u32,
    pub name: String,
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
impl LocationEntryCore for GenericEntry {
    fn from_df(df: &DataFrame) -> Result<Vec<Self>> {
        let cols = df.select(Self::field_names())?.take_columns();

        // For some reason izip! can only handle 9 args so here we need to do a wierd zip thing and unpack slightly strangely to get it working.
        Ok(izip!(
            cols[0].u32()?,
            cols[1].str()?,
            cols[2].str()?,
            cols[3].str()?,
            cols[4].str()?,
            cols[5].str()?,
            cols[6].str()?,
            cols[7].str()?,
        )
        .zip(
            cols[8]
                .f32()?
                .iter()
                .zip(cols[9].f32()?.iter().zip(cols[10].i64()?.iter())),
        )
        .map(
            |(
                (
                    geoname_id,
                    name,
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

impl fmt::Display for GenericEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GenericEntry {{ geoname_id: {}, name: \"{}\" }}",
            self.geoname_id, self.name
        )
    }
}
