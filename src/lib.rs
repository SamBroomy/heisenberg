pub mod search;
use anyhow::Result;
use once_cell::sync::OnceCell;
pub use search::{
    AdminSearchParams, // If users need to customize admin search directly
    // Re-export the entry types if users need to specify them for enrichment
    // (e.g., if they want to use GeonameEntry or GeonameFullEntry specifically)
    // Or, if you want to simplify, you might only expose FullAdminHierarchy<GeonameFullEntry>
    // via a type alias or make GeonameFullEntry the default.
    // For now, let's re-export them so the user has the choice.
    GeonameEntry,     // from search::location_search::enrichment
    GeonameFullEntry, // from search::location_search::enrichment
    LocationSearchService,
    PlaceSearchParams, // If users need to customize place search directly
    SmartFlexibleSearchConfig,
    TargetLocationAdminCodes,
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
