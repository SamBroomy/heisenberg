use std::fmt;

use itertools::izip;
use polars::prelude::*;

use super::{LocationEntryCore, Result};

#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, name = "BasicEntry"))]
#[derive(Debug, Clone, Default)]
pub struct BasicEntry {
    pub geoname_id: u32,
    pub name: String,
}
impl LocationEntryCore for BasicEntry {
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

impl fmt::Display for BasicEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BasicEntry {{ geoname_id: {}, name: \"{}\" }}",
            self.geoname_id, self.name
        )
    }
}
