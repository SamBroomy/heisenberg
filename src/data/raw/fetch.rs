use anyhow::Result;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::fs;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;
use tracing::info;
use zip::ZipArchive;

pub async fn download_to_temp_file(client: &Client, url: &str) -> Result<NamedTempFile> {
    info!(url, "Starting download");
    let response = client.get(url).send().await?.error_for_status()?;

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));
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

pub async fn download_zip_and_extract_first_entry_to_temp_file(
    client: &Client,
    zip_url: &str,
) -> Result<NamedTempFile> {
    info!(zip_url, "Starting ZIP download");
    // Download the zip file itself (will show its own progress bar)
    let zip_temp_file = download_to_temp_file(client, zip_url).await?;
    info!(path = ?zip_temp_file.path(), "ZIP download complete");

    let zip_file_path = zip_temp_file.path().to_path_buf();
    let zip_url_owned = zip_url.to_string();

    let extracted_content_temp_file = tokio::task::spawn_blocking(move || {
        extract_first_entry_from_zip(zip_file_path, &zip_url_owned)
    })
    .await??;

    Ok(extracted_content_temp_file)
}

fn extract_first_entry_from_zip(zip_file_path: PathBuf, zip_url: &str) -> Result<NamedTempFile> {
    let zip_fs_file = fs::File::open(&zip_file_path)?;
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

    std::io::copy(&mut file_in_zip, &mut extracted_fs_file)?;
    info!(path = ?extracted_content_temp_file.path(), "File extracted successfully");

    Ok(extracted_content_temp_file)
}
