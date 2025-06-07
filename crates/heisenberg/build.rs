use heisenberg_data_processing::{EmbeddedDataSource, TestDataConfig, generate_embedded_dataset};
use polars::prelude::*;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only generate embedded data when explicitly requested or if files don't exist
    if should_generate_embedded_data() {
        generate_embedded_dataset_files()?;
    } else {
        println!(
            "cargo:warning=Skipping embedded dataset generation (set GENERATE_EMBEDDED_DATA=1 to force)"
        );
    }

    // Tell cargo to rerun if embedded data files change
    println!("cargo:rerun-if-changed=src/data/embedded/");
    println!("cargo:rerun-if-env-changed=GENERATE_EMBEDDED_DATA");
    println!("cargo:rerun-if-env-changed=USE_CITIES15000");

    Ok(())
}

fn should_generate_embedded_data() -> bool {
    // Generate if explicitly requested
    if std::env::var("GENERATE_EMBEDDED_DATA").unwrap_or_default() == "1" {
        return true;
    }

    // Generate if embedded files don't exist
    let embedded_dir = Path::new("src/data/embedded");
    if !embedded_dir.exists()
        || !embedded_dir.join("admin_search.parquet").exists()
        || !embedded_dir.join("place_search.parquet").exists()
    {
        return true;
    }

    false
}

fn generate_embedded_dataset_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Generating embedded dataset using data processing subcrate...");

    // Create embedded directory
    let embedded_dir = Path::new("src/data/embedded");
    fs::create_dir_all(embedded_dir)?;

    // Generate dataset using the data processing subcrate
    // Use cities15000.zip for production embedded data
    let source = if std::env::var("USE_CITIES15000").unwrap_or_default() == "1" {
        println!("cargo:warning=Using cities15000.zip data source");
        EmbeddedDataSource::Cities15000
    } else {
        println!("cargo:warning=Using enhanced test data (set USE_CITIES15000=1 for real data)");
        EmbeddedDataSource::TestData(TestDataConfig::sample())
    };
    let dataset = generate_embedded_dataset(source)?;

    println!(
        "cargo:warning=Generated dataset with {} admin rows, {} place rows",
        dataset.metadata.admin_rows, dataset.metadata.place_rows
    );

    // Save as Parquet files
    let admin_path = embedded_dir.join("admin_search.parquet");
    let place_path = embedded_dir.join("place_search.parquet");
    let metadata_path = embedded_dir.join("metadata.json");

    // Write Parquet files
    use std::fs::File;
    let admin_file = File::create(&admin_path)?;
    let place_file = File::create(&place_path)?;

    ParquetWriter::new(admin_file).finish(&mut dataset.admin_data.clone())?;
    ParquetWriter::new(place_file).finish(&mut dataset.place_data.clone())?;

    // Write metadata
    let metadata_json = serde_json::to_string_pretty(&serde_json::json!({
        "version": dataset.metadata.version,
        "source": dataset.metadata.source,
        "generated_at": dataset.metadata.generated_at,
        "description": dataset.metadata.description,
        "admin_rows": dataset.metadata.admin_rows,
        "place_rows": dataset.metadata.place_rows,
        "files": {
            "admin_search": "admin_search.parquet",
            "place_search": "place_search.parquet"
        }
    }))?;

    fs::write(&metadata_path, metadata_json)?;

    // Calculate actual file sizes
    let admin_size = fs::metadata(&admin_path)?.len();
    let place_size = fs::metadata(&place_path)?.len();
    let total_size = admin_size + place_size;

    println!("cargo:warning=Embedded dataset saved:");
    println!(
        "cargo:warning=  Admin: {} bytes ({:.1} KB)",
        admin_size,
        admin_size as f64 / 1024.0
    );
    println!(
        "cargo:warning=  Place: {} bytes ({:.1} KB)",
        place_size,
        place_size as f64 / 1024.0
    );
    println!(
        "cargo:warning=  Total: {} bytes ({:.1} KB)",
        total_size,
        total_size as f64 / 1024.0
    );

    Ok(())
}
