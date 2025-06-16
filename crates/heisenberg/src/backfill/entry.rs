use std::fmt;

use itertools::izip;
use polars::prelude::*;

use super::Result;

/// Unified location entry containing all available location data.
///
/// This struct combines all useful fields from the `GeoNames` dataset, providing
/// a rich representation of location information without the complexity of
/// multiple entry types. Users can access only the fields they need.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, name = "LocationEntry"))]
#[derive(Debug, Clone, Default)]
pub struct LocationEntry {
    /// `GeoNames` unique identifier
    pub geoname_id: u32,
    /// Primary name of the location
    pub name: String,
    /// ISO country code (admin level 0)
    pub admin0_code: Option<String>,
    /// Administrative subdivision code (admin level 1) - state/province
    pub admin1_code: Option<String>,
    /// Administrative subdivision code (admin level 2) - county/region
    pub admin2_code: Option<String>,
    /// Administrative subdivision code (admin level 3) - local admin division
    pub admin3_code: Option<String>,
    /// Administrative subdivision code (admin level 4) - sub-local admin division
    pub admin4_code: Option<String>,
    /// `GeoNames` feature code (e.g., "PPL" for populated place, "ADM1" for admin division)
    pub feature_code: String,
    /// Latitude in decimal degrees
    pub latitude: Option<f32>,
    /// Longitude in decimal degrees
    pub longitude: Option<f32>,
    /// Population count (if available)
    pub population: Option<i64>,
}

impl LocationEntry {
    /// Create `LocationEntry` instances from a Polars `DataFrame`.
    ///
    /// The `DataFrame` is expected to contain columns matching the field names
    /// returned by `field_names()`.
    pub fn from_df(df: &DataFrame) -> Result<Vec<Self>> {
        let cols = df.select(Self::field_names())?.take_columns();

        // Handle the complex zip due to izip! limitation with many arguments
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
                admin0_code: admin0_code.map(ToString::to_string),
                admin1_code: admin1_code.map(ToString::to_string),
                admin2_code: admin2_code.map(ToString::to_string),
                admin3_code: admin3_code.map(ToString::to_string),
                admin4_code: admin4_code.map(ToString::to_string),
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

    /// Returns the geoname ID of the location.
    #[must_use]
    pub fn geoname_id(&self) -> u32 {
        self.geoname_id
    }

    /// Returns the primary name of the location.
    #[must_use]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Returns the column names expected in `DataFrames` for this entry type.
    #[must_use]
    pub fn field_names() -> Vec<&'static str> {
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

    /// Returns basic identification info as a tuple (id, name).
    /// Convenience method for users who only need minimal data.
    #[must_use]
    pub fn basic_info(&self) -> (u32, &str) {
        (self.geoname_id, &self.name)
    }

    /// Returns coordinates as a tuple (latitude, longitude) if both are available.
    #[must_use]
    pub fn coordinates(&self) -> Option<(f32, f32)> {
        self.latitude.zip(self.longitude)
    }

    /// Returns true if this entry represents an administrative division.
    #[must_use]
    pub fn is_admin(&self) -> bool {
        self.feature_code.starts_with("ADM")
    }

    /// Returns true if this entry represents a populated place.
    #[must_use]
    pub fn is_place(&self) -> bool {
        self.feature_code.starts_with("PPL") || self.feature_code == "PPLC"
    }

    /// Returns the administrative level if this is an administrative division.
    /// Returns None for non-administrative entries.
    #[must_use]
    pub fn admin_level(&self) -> Option<u8> {
        match self.feature_code.as_str() {
            "ADM0" => Some(0),
            "ADM1" => Some(1),
            "ADM2" => Some(2),
            "ADM3" => Some(3),
            "ADM4" => Some(4),
            _ => None,
        }
    }
}

impl fmt::Display for LocationEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LocationEntry {{ geoname_id: {}, name: \"{}\" }}",
            self.geoname_id, self.name
        )
    }
}
