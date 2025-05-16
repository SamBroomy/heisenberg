use hbergv4::data::{all_countries, country_info, get_admin_search_data};
use hbergv4::search::location_search::{get_admin_df, resolve_search_candidates};
use hbergv4::search::{GeonameEntry, LocationSearchService, SmartFlexibleSearchConfig};
use std::fs;
use std::io::{copy, Read}; // Added copy
use std::path::Path;

use anyhow::Result;

use ::zip::ZipArchive;
use polars::{enable_string_cache, prelude::*};
use reqwest::blocking::Client;
use tempfile::{Builder, NamedTempFile};
use tracing::{debug, info, info_span, warn};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

fn download_to_temp_file(client: &Client, url: &str) -> Result<NamedTempFile> {
    info!(url, "Starting download");
    let mut response = client.get(url).send()?.error_for_status()?;
    let temp_file = NamedTempFile::new()?;
    let mut dest_file = fs::File::create(temp_file.path())?;
    copy(&mut response, &mut dest_file)?;
    info!(path = ?temp_file.path(), "Download complete");
    Ok(temp_file)
}
fn download_zip_and_extract_first_entry_to_temp_file(
    client: &Client,
    zip_url: &str,
) -> Result<(NamedTempFile, String)> {
    info!(zip_url, "Starting ZIP download");
    let zip_temp_file = download_to_temp_file(client, zip_url)?; // This is the downloaded .zip file
    info!(path = ?zip_temp_file.path(), "ZIP download complete");

    let zip_fs_file = fs::File::open(zip_temp_file.path())?;
    let mut archive = ZipArchive::new(zip_fs_file)?;

    if archive.is_empty() {
        return Err(anyhow::anyhow!(
            "Downloaded ZIP archive is empty: {}",
            zip_url
        ));
    }

    let mut file_in_zip = archive.by_index(0)?; // Assuming the first file is the one we want
    let extracted_filename = file_in_zip.name().to_string();
    info!(extracted_filename, "Extracting file from ZIP");

    // Create a new temp file for the extracted content.
    // Suffix can be useful for debugging or if other tools expect a certain extension.
    let extracted_content_temp_file = NamedTempFile::with_suffix(".txt")?;

    let mut extracted_fs_file = fs::File::create(extracted_content_temp_file.path())?;
    copy(&mut file_in_zip, &mut extracted_fs_file)?;
    info!(path = ?extracted_content_temp_file.path(), "File extracted successfully");

    // zip_temp_file (the .zip) will be cleaned up when it goes out of scope here.
    // We return the temp file for the *extracted content*.
    Ok((extracted_content_temp_file, extracted_filename))
}

fn main() -> Result<()> {
    let t_start = std::time::Instant::now();

    // Initialize HTTP client
    let client = Client::builder().build()?;

    // --- Download and process countryInfo.txt ---
    let t_country_info_download = std::time::Instant::now();
    let country_info_url = "https://download.geonames.org/export/dump/countryInfo.txt";
    let country_info_temp_file = download_to_temp_file(&client, country_info_url)?;
    let country_info_path = country_info_temp_file.path();
    warn!(
        elapsed_seconds = t_country_info_download.elapsed().as_secs_f32(),
        "Downloading countryInfo.txt took"
    );
    let t_country_info_read = std::time::Instant::now();
    let country_info_df = country_info(country_info_path)?.collect()?;
    dbg!(&country_info_df.head(Some(3)));
    warn!(
        elapsed_seconds = t_country_info_read.elapsed().as_secs_f32(),
        "Reading countryInfo.txt into DataFrame took"
    );

    // --- Download, extract, and process allCountries.txt ---
    let t_all_countries_download_extract = std::time::Instant::now();
    let all_countries_zip_url = "https://download.geonames.org/export/dump/allCountries.zip";
    let (all_countries_extracted_temp_file, extracted_filename) =
        download_zip_and_extract_first_entry_to_temp_file(&client, all_countries_zip_url)?;
    info!(
        extracted_filename,
        "Using extracted file for allCountries DataFrame"
    );
    let all_countries_path = all_countries_extracted_temp_file.path();
    warn!(
        elapsed_seconds = t_all_countries_download_extract.elapsed().as_secs_f32(),
        "Downloading and unzipping allCountries (from {}) took", extracted_filename
    );
    let t_all_countries_read = std::time::Instant::now();
    let all_countries_df = all_countries(all_countries_path)?.collect()?;
    dbg!(&all_countries_df.head(Some(3)));
    warn!(
        elapsed_seconds = t_all_countries_read.elapsed().as_secs_f32(),
        "Reading allCountries.txt (from {}) into DataFrame took", extracted_filename
    );

    let admin_search =
        get_admin_search_data(&all_countries_df.lazy(), &country_info_df.lazy())?.collect()?;

    dbg!(admin_search.schema());
    dbg!(admin_search.sample_n_literal(10, false, false, Some(69))?);
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("debug"))?
        .add_directive("tantivy=info".parse()?);
    dbg!(admin_search);
    tracing_subscriber::fmt::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    let t_total_setup = std::time::Instant::now();
    let search_service = LocationSearchService::new(false)?;

    debug!(
        elapsed_seconds = t_total_setup.elapsed().as_secs_f32(),
        "LocationSearchService setup complete"
    );
    enable_string_cache();
    let _example_search_span = info_span!("manual_search_example").entered();

    // Example using the service
    let admins = search_service
        .admin_search(
            "The united states of america",
            &[0, 1],
            None::<DataFrame>,
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admins = ?admins, "Admin search results");

    let admins1 = search_service
        .admin_search(
            "California",
            &[1, 2],
            Some(admins), // Pass the DataFrame directly
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admins1 = ?admins1, "Admin1 search results");
    let admins2 = search_service
        .admin_search(
            "Los Angeles County",
            &[2, 3],
            Some(admins1),
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admins2 = ?admins2, "Admin2 search results");

    let admin3 = search_service
        .admin_search(
            "Beverly Hills",
            &[3, 4],
            Some(admins2.clone()), // Clone if admins2 is used again
            &Default::default(),
        )?
        .unwrap_or_default();
    debug!(admin3 = ?admin3, "Admin3 search results");

    if !admins2.is_empty() && !admin3.is_empty() {
        let places_input_df = concat(
            &[admins2.lazy(), admin3.lazy()],
            UnionArgs {
                diagonal: true,
                ..Default::default()
            },
        )?
        .collect()?; // Collect to DataFrame for place_search if it expects DataFrame

        let place = search_service
            .place_search("Beverly Hills", Some(places_input_df), &Default::default())?
            .unwrap_or_default();
        debug!(place = ?place, "Place search results");
    }

    drop(_example_search_span);

    let examples = vec![
        vec!["US", "CA", "SF", "Golden Gate Bridge"],
        vec!["FL", "Lakeland"],
        vec![
            "The united states of america",
            "California",
            "Los Angeles County",
            "Beverly Hills",
        ],
        vec!["UK", "London", "Camden", "British Museum"],
        vec!["United Kingdom", "London", "Westminster", "Parlement"],
        vec!["FR", "Provence-Alpes-Côte d'Azur", "Le Lavandou"],
        vec!["England", "Dover", "Dover Ferry Terminal"],
        vec![
            "FR",
            "Provence-Alpes-Côte d'Azur",
            "Var",
            "Arrondissement de Toulon",
            "Le Lavandou",
        ],
    ];
    let mut times = vec![];
    let smart_search_config = SmartFlexibleSearchConfig::default();

    for input in &examples {
        let t0 = std::time::Instant::now();
        let output = search_service.smart_flexible_search(input, &smart_search_config)?;
        let elapsed = t0.elapsed().as_secs_f32();

        warn!(
            ave_time = elapsed / input.len() as f32,
            "Smart flexible search seconds per example"
        );
        times.push(elapsed);

        for (i, df) in output.iter().enumerate() {
            info!(i=i,df=?df, "Smart flexible search results");
        }
    }
    let avg_time = times.iter().sum::<f32>() / times.len() as f32;
    warn!(avg_time = avg_time, "Average smart flexible search time");
    warn!(
        total_time = times.iter().sum::<f32>(),
        "Total smart flexible search time"
    );

    let t_bulk = std::time::Instant::now();
    let examples_refs: Vec<&[&str]> = examples.iter().map(|v| v.as_slice()).collect();
    let out_bulk =
        search_service.bulk_smart_flexible_search(&examples_refs, &smart_search_config)?;

    warn!(t_bulk = ?t_bulk.elapsed(), "Bulk smart flexible search took");
    warn!(t_avg_per_example = ?t_bulk.elapsed().as_secs_f32() / examples.len() as f32, "Average time per example");

    for (i, df) in out_bulk.iter().enumerate() {
        if df.is_empty() {
            warn!(i, "Bulk smart flexible search results {}: empty", i);
            continue;
        }
        info!(i, df = ?df, "Bulk smart flexible search results");
    }

    let out =
        resolve_search_candidates::<GeonameEntry>(out_bulk, &get_admin_df()?.clone().lazy(), 10)?;

    for (i, out) in out.iter().enumerate() {
        if out.is_empty() {
            warn!(i, "Bulk smart flexible search results {}: empty", i);
            continue;
        }
        info!(i, out = ?out, "Bulk smart flexible search results");
        for res in out.iter() {
            info!(simple = ?res.simple());
            info!(full = ?res.full());
        }
    }

    Ok(())
}
