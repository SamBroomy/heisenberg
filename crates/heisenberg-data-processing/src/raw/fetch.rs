use crate::{DataError, raw::DataSource};

use super::Result;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::fs;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;
use tracing::{info, instrument};
use zip::ZipArchive;

#[instrument(name = "Download data", skip_all, level = "info")]
pub fn download_data(
    data_source: &DataSource,
) -> Result<(NamedTempFile, NamedTempFile, NamedTempFile)> {
    let rt = tokio::runtime::Runtime::new()?;

    let data_source_url = data_source
        .geonames_url()
        .ok_or_else(|| DataError::NoDataDirProvided)?;

    rt.block_on(async {
        let client = reqwest::Client::new();

        let (cities15000_file, country_info_df, feature_codes_df) = tokio::try_join!(
            download_raw_data(&client, &data_source_url),
            download_country_info(&client),
            download_feature_codes(&client),
        )?;

        Ok((cities15000_file, country_info_df, feature_codes_df))
    })
}

/*

cities15000.zip: Generated dataset with 13176 admin rows, 17960 place rows (~12s)
warning: heisenberg@0.1.0:   Admin: 1657242 bytes (1618.4 KB)
warning: heisenberg@0.1.0:   Place: 1880950 bytes (1836.9 KB)
warning: heisenberg@0.1.0:   Total: 3538192 bytes (3455.3 KB)

cities5000.zip: Generated dataset with 24021 admin rows, 38525 place rows (~14s)
warning: heisenberg@0.1.0:   Admin: 2381696 bytes (2325.9 KB)
warning: heisenberg@0.1.0:   Place: 3581318 bytes (3497.4 KB)
warning: heisenberg@0.1.0:   Total: 5963014 bytes (5823.3 KB)

cities1000.zip: Generated dataset with 62733 admin rows, 94526 place rows (~16s)
warning: heisenberg@0.1.0:   Admin: 4464743 bytes (4360.1 KB)
warning: heisenberg@0.1.0:   Place: 7256275 bytes (7086.2 KB)
warning: heisenberg@0.1.0:   Total: 11721018 bytes (11446.3 KB)

cities500.zip: Generated dataset with 81545 admin rows, 136847 place rows (~1m 10s)
warning: heisenberg@0.1.0:   Admin: 5293457 bytes (5169.4 KB)
warning: heisenberg@0.1.0:   Place: 9504661 bytes (9281.9 KB)
warning: heisenberg@0.1.0:   Total: 14798118 bytes (14451.3 KB)
*/

async fn download_raw_data(client: &Client, url: &str) -> Result<NamedTempFile> {
    download_zip_and_extract_first_entry_to_temp_file(client, url).await
}

const COUNTRY_INFO_URL: &str = "https://download.geonames.org/export/dump/countryInfo.txt";
/// Downloads the country info file from GeoNames and extracts it to a temporary file.
async fn download_country_info(client: &Client) -> Result<NamedTempFile> {
    download_to_temp_file(client, COUNTRY_INFO_URL).await
}

const FEATURE_CODES_URL: &str = "https://download.geonames.org/export/dump/featureCodes_en.txt";
/// Downloads the feature codes file from GeoNames and extracts it to a temporary file.
async fn download_feature_codes(client: &Client) -> Result<NamedTempFile> {
    download_to_temp_file(client, FEATURE_CODES_URL).await
}

async fn download_to_temp_file(client: &Client, url: &str) -> Result<NamedTempFile> {
    info!(url, "Starting download");
    let response = client.get(url).send().await?.error_for_status()?;

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").expect("Progress bar template")
        .progress_chars("█░"));
    pb.set_message(format!(
        "Downloading {}",
        url.split('/').next_back().unwrap_or(url)
    ));

    let temp_file = NamedTempFile::new()?;
    let mut dest_file = tokio::fs::File::create(temp_file.path()).await?;
    //let mut dest_file = fs::File::create(temp_file.path())?;

    let mut stream = response.bytes_stream();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        dest_file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }
    dest_file.flush().await?; // Ensure all bytes are written
    pb.finish_and_clear();
    println!();
    Ok(temp_file)
}

async fn download_zip_and_extract_first_entry_to_temp_file(
    client: &Client,
    zip_url: &str,
) -> Result<NamedTempFile> {
    info!(zip_url, "Starting ZIP download");
    // Download the zip file itself (will show its own progress bar)
    let zip_temp_file = download_to_temp_file(client, zip_url).await?;
    info!(path = ?zip_temp_file.path(), "ZIP download complete");

    let zip_file_path = zip_temp_file.path().to_path_buf();

    let extracted_content_temp_file =
        tokio::task::spawn_blocking(move || extract_first_entry_from_zip(zip_file_path)).await??;

    Ok(extracted_content_temp_file)
}

fn extract_first_entry_from_zip(zip_file_path: PathBuf) -> Result<NamedTempFile> {
    let zip_fs_file = fs::File::open(&zip_file_path)?;
    let mut archive = ZipArchive::new(zip_fs_file)?;

    if archive.is_empty() {
        return Err(zip::result::ZipError::FileNotFound.into());
    }

    let mut file_in_zip = archive.by_index(0)?;

    let extracted_content_temp_file = NamedTempFile::with_suffix(".txt")?;
    let mut extracted_fs_file = fs::File::create(extracted_content_temp_file.path())?;

    std::io::copy(&mut file_in_zip, &mut extracted_fs_file)?;
    info!(path = ?extracted_content_temp_file.path(), "File extracted successfully");

    Ok(extracted_content_temp_file)
}
