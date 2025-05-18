pub mod backfill;
pub mod data;
pub mod index;
pub mod search;
use anyhow::Result;
pub use backfill::{GenericEntry, GeonameEntry, LocationEntry};
use once_cell::sync::OnceCell;
pub use search::{
    AdminSearchParams, LocationSearchService, PlaceSearchParams, SmartFlexibleSearchConfig,
};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

static PLACES_DF_CACHE: OnceCell<()> = OnceCell::new();

pub fn init_logging(level: impl Into<LevelFilter>) -> Result<&'static ()> {
    PLACES_DF_CACHE.get_or_try_init(|| {
        let filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(level.into().to_string()))?
            .add_directive("tantivy=info".parse()?);

        tracing_subscriber::fmt::fmt()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::CLOSE)
            .init();
        Ok(())
    })
}
