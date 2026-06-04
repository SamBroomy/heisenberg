use std::{
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use heisenberg_data_processing::{
    DataError, DataSource,
    embedded::{
        ADMIN_DATA_PATH, EmbeddedMetadata, METADATA_PATH, PLACE_DATA_PATH,
        generate_embedded_dataset_to_dir,
    },
    error::Result,
};

fn main() -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let out_embedded_dir = out_dir.join("embedded_data");

    let data_source = get_configured_data_source();

    // Ensure embedded data exists in OUT_DIR (via cache or generation)
    ensure_embedded_data_in_out_dir(&out_embedded_dir, data_source)?;

    // Load metadata and generate hash constant
    generate_metadata_hash_file(&out_dir, &out_embedded_dir)?;

    // Expose embedded data directory for include_bytes! macros
    println!(
        "cargo:rustc-env=EMBEDDED_DATA_DIR={}",
        out_embedded_dir.display()
    );
    println!("cargo:rustc-env=ADMIN_DATA_PATH={ADMIN_DATA_PATH}");
    println!("cargo:rustc-env=PLACE_DATA_PATH={PLACE_DATA_PATH}");
    println!("cargo:rustc-env=METADATA_PATH={METADATA_PATH}");

    // Tell cargo to rerun if relevant env vars change
    println!("cargo:rerun-if-env-changed=GENERATE_EMBEDDED_DATA");
    println!("cargo:rerun-if-env-changed=EMBEDDED_DATA_SOURCE");

    Ok(())
}

fn get_configured_data_source() -> DataSource {
    env::var("EMBEDDED_DATA_SOURCE").map_or_else(
        |_| DataSource::default(),
        |source| DataSource::from_str(&source).unwrap_or_default(),
    )
}

/// Get user cache directory for embedded data
/// Returns ~/.cache/heisenberg-geo/embedded_data/ on Linux, ~/Library/Caches/heisenberg-geo/embedded_data/ on macOS
fn get_cache_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "heisenberg-geo")
        .map(|dirs| dirs.cache_dir().join("embedded_data"))
}

/// Check if all required embedded data files exist in a directory
fn embedded_data_exists(dir: &Path) -> bool {
    dir.join(ADMIN_DATA_PATH).exists()
        && dir.join(PLACE_DATA_PATH).exists()
        && dir.join(METADATA_PATH).exists()
}

/// Copy embedded data files from source to destination
fn copy_embedded_data(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    fs::create_dir_all(dst_dir)?;

    for filename in [ADMIN_DATA_PATH, PLACE_DATA_PATH, METADATA_PATH] {
        let src = src_dir.join(filename);
        let dst = dst_dir.join(filename);
        if src.exists() {
            fs::copy(&src, &dst)?;
        }
    }

    Ok(())
}

/// Ensure embedded data exists in `OUT_DIR`
/// Priority:
/// 1. Already in `OUT_DIR` → use as-is
/// 2. In user cache with matching version → copy to `OUT_DIR`
/// 3. Generate fresh → save to cache AND `OUT_DIR`
fn ensure_embedded_data_in_out_dir(out_embedded_dir: &Path, data_source: DataSource) -> Result<()> {
    // 1. Already exists in OUT_DIR? Done.
    if embedded_data_exists(out_embedded_dir) {
        // Still check if data source matches
        if let Ok(metadata) = check_existing_metadata(out_embedded_dir)
            && metadata.source == data_source
            && env::var("GENERATE_EMBEDDED_DATA").unwrap_or_default() != "1"
        {
            return Ok(());
        }
    }

    // 2. Check user cache
    if let Some(cache_dir) = get_cache_dir() {
        // Include library version in cache path for auto-invalidation on version bumps
        let cache_version_dir = cache_dir
            .join(env!("CARGO_PKG_VERSION"))
            .join(format!("{data_source:?}"));

        if embedded_data_exists(&cache_version_dir)
            && env::var("GENERATE_EMBEDDED_DATA").unwrap_or_default() != "1"
        {
            // Validate cache metadata matches requested source
            if let Ok(metadata) = check_existing_metadata(&cache_version_dir)
                && metadata.source == data_source
            {
                println!(
                    "cargo:warning=Using cached embedded data from {}",
                    cache_version_dir.display()
                );
                copy_embedded_data(&cache_version_dir, out_embedded_dir)?;
                return Ok(());
            }
        }
    }

    // 3. Generate fresh data
    println!("cargo:warning=Generating embedded data for {data_source:?}...");

    // Generate to cache first (if available), then copy to OUT_DIR
    if let Some(cache_dir) = get_cache_dir() {
        // Include library version in cache path
        let cache_version_dir = cache_dir
            .join(env!("CARGO_PKG_VERSION"))
            .join(format!("{data_source:?}"));

        // Generate to cache
        generate_embedded_dataset_to_dir(data_source, &cache_version_dir)?;

        println!(
            "cargo:warning=Cached embedded data at {}",
            cache_version_dir.display()
        );

        // Copy to OUT_DIR
        copy_embedded_data(&cache_version_dir, out_embedded_dir)?;
    } else {
        // No cache directory available, generate directly to OUT_DIR
        println!("cargo:warning=Cache directory not available, generating to OUT_DIR");
        generate_embedded_dataset_to_dir(data_source, out_embedded_dir)?;
    }

    println!(
        "cargo:warning=Embedded data ready in {}",
        out_embedded_dir.display()
    );

    Ok(())
}

fn check_existing_metadata(embedded_dir: &Path) -> Result<EmbeddedMetadata> {
    let metadata_path = embedded_dir.join(METADATA_PATH);
    if metadata_path.exists() {
        EmbeddedMetadata::load_from_file(&metadata_path)
    } else {
        Err(DataError::MetadataFileNotFound)
    }
}

/// Generate a Rust source file with the metadata hash constant
fn generate_metadata_hash_file(out_dir: &Path, embedded_dir: &Path) -> Result<()> {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    // Load the metadata
    let metadata = check_existing_metadata(embedded_dir)?;

    // Compute the hash (same logic as EmbeddedMetadata::content_hash)
    let mut hasher = DefaultHasher::new();
    metadata.hash(&mut hasher);
    let hash = format!("{:x}", hasher.finish());

    // Generate Rust source code
    let hash_rs = format!(
        "// This file is auto-generated by build.rs

/// Metadata content hash for cache invalidation.
///
/// This hash uniquely identifies the embedded data based on:
/// - Library version
/// - Data source (Cities15000, etc.)
/// - Data content (row counts)
///
/// When the hash changes, indexes are automatically rebuilt.
pub const METADATA_HASH: &str = \"{hash}\";\n"
    );

    fs::write(out_dir.join("metadata_hash.rs"), hash_rs).expect("Failed to write metadata_hash.rs");

    Ok(())
}
