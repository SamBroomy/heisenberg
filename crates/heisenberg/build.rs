use std::{env, str::FromStr};

use heisenberg_data_processing::{
    DataError, DataSource,
    embedded::{
        ADMIN_DATA_PATH, EMBEDDED_DIR, EmbeddedMetadata, METADATA_PATH, PLACE_DATA_PATH,
        generate_embedded_dataset,
    },
    error::Result,
};

fn main() -> Result<()> {
    // Only generate embedded data when explicitly requested or if files don't exist
    if let Some(data_source) = should_generate_embedded_data() {
        generate_embedded_dataset_files(data_source)?;
    } else {
        println!(
            "cargo:warning=Skipping embedded dataset generation (set GENERATE_EMBEDDED_DATA=1 to force)"
        );
    }

    // Tell cargo to rerun if embedded data files change or relevant env vars change
    println!("cargo:rerun-if-changed={}", EMBEDDED_DIR.display());
    println!("cargo:rerun-if-env-changed=GENERATE_EMBEDDED_DATA");
    println!("cargo:rerun-if-env-changed=EMBEDDED_DATA_SOURCE");

    Ok(())
}

fn get_configured_data_source() -> DataSource {
    match env::var("EMBEDDED_DATA_SOURCE") {
        Ok(source) => DataSource::from_str(&source).unwrap_or_default(),
        Err(_) => DataSource::default(),
    }
}

fn should_generate_embedded_data() -> Option<DataSource> {
    // Generate if explicitly requested
    if env::var("GENERATE_EMBEDDED_DATA").unwrap_or_default() == "1" {
        return Some(get_configured_data_source());
    }

    if !EMBEDDED_DIR.exists()
        || !EMBEDDED_DIR.join(ADMIN_DATA_PATH).exists()
        || !EMBEDDED_DIR.join(PLACE_DATA_PATH).exists()
        || !EMBEDDED_DIR.join(METADATA_PATH).exists()
    {
        return Some(get_configured_data_source());
    }

    // Generate if the data source has changed (check metadata)
    if let Ok(existing_metadata) = check_existing_metadata() {
        let configured_source = get_configured_data_source();
        if existing_metadata.source != configured_source {
            println!(
                "cargo:warning=Data source changed from {:?} to {:?}, regenerating embedded data",
                existing_metadata.source, configured_source
            );
            return Some(configured_source);
        }
    }

    None
}

fn check_existing_metadata() -> Result<EmbeddedMetadata> {
    let metadata_path = EMBEDDED_DIR.join(METADATA_PATH);
    if metadata_path.exists() {
        EmbeddedMetadata::load_from_file(&metadata_path)
    } else {
        Err(DataError::MetadataFileNotFound)
    }
}

fn generate_embedded_dataset_files(data_source: DataSource) -> Result<()> {
    println!("cargo:info=Generating embedded dataset using data processing subcrate...");

    // Generate based on configured data source
    generate_embedded_dataset(data_source)?;

    println!(
        "cargo:info=Generated embedded Rust source files in {:?}",
        EMBEDDED_DIR.display()
    );
    println!("cargo:info=Embedded data will be compiled into the binary using include_bytes!");

    Ok(())
}
