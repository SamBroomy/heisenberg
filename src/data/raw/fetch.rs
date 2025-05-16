use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::fs;
use std::io::copy;
use tempfile::NamedTempFile;
use tracing::info;
use zip::ZipArchive;

pub fn download_to_temp_file(client: &Client, url: &str) -> Result<NamedTempFile> {
    info!(url, "Starting download");
    let response = client.get(url).send()?.error_for_status()?;

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));
    pb.set_message(format!(
        "Downloading {}",
        url.split('/').last().unwrap_or(url)
    ));

    let temp_file = NamedTempFile::new()?;
    let mut dest_file = fs::File::create(temp_file.path())?;

    // Wrap the response in a progress reader
    let mut source = pb.wrap_read(response);

    copy(&mut source, &mut dest_file)?;
    pb.finish_with_message(format!(
        "Downloaded {} to {:?}",
        url.split('/').last().unwrap_or(url),
        temp_file.path()
    ));
    info!(path = ?temp_file.path(), "Download complete");
    Ok(temp_file)
}

pub fn download_zip_and_extract_first_entry_to_temp_file(
    client: &Client,
    zip_url: &str,
) -> Result<NamedTempFile> {
    info!(zip_url, "Starting ZIP download");
    // Download the zip file itself (will show its own progress bar)
    let zip_temp_file = download_to_temp_file(client, zip_url)?;
    info!(path = ?zip_temp_file.path(), "ZIP download complete");

    let zip_fs_file = fs::File::open(zip_temp_file.path())?;
    let mut archive = ZipArchive::new(zip_fs_file)?;

    if archive.is_empty() {
        return Err(anyhow::anyhow!(
            "Downloaded ZIP archive is empty: {}",
            zip_url
        ));
    }

    let mut file_in_zip = archive.by_index(0)?;

    let extracted_content_temp_file = NamedTempFile::with_suffix(".txt")?;
    let mut extracted_fs_file = fs::File::create(extracted_content_temp_file.path())?;

    copy(&mut file_in_zip, &mut extracted_fs_file)?;
    info!(path = ?extracted_content_temp_file.path(), "File extracted successfully");

    Ok(extracted_content_temp_file)
}
