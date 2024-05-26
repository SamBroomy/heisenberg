use ::zip::ZipArchive;
use polars::prelude::*;
use reqwest::Client;
use std::io::{copy, Cursor, Seek};
use std::path::Path;
use tempfile::tempfile;

async fn download_file_to_path(
    target: &str,
    extracted_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = target.split('/').last().unwrap();
    let client = Client::new();

    let response = client.get(target).send().await?.bytes().await?;

    let mut file = std::fs::File::create(extracted_dir.join(Path::new(file_name)))?;

    copy(&mut Cursor::new(response), &mut file)?;

    println!("Downloaded to {:?}", extracted_dir);

    Ok(())
}

async fn download_zip_to_path_and_unzip(
    target: &str,
    extracted_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = target.split('/').last().unwrap();

    let client = Client::new();

    let response = client.get(target).send().await?.bytes().await?;

    let mut tempfile = tempfile()?;

    copy(&mut Cursor::new(response), &mut tempfile)?;

    tempfile.rewind()?;

    std::fs::create_dir_all(extracted_dir)?;

    let mut zip = ZipArchive::new(tempfile)?;

    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => extracted_dir.join(path),
            None => continue,
        };

        if file.name().ends_with("/") {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = std::fs::File::create(&outpath)?;
            copy(&mut file, &mut outfile)?;
        }
    }
    println!("Downloaded and extracted to {:?}", extracted_dir);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let extracted_dir = Path::new("./data/geonames");

    download_file_to_path(
        "https://download.geonames.org/export/dump/admin1CodesASCII.txt",
        extracted_dir,
    )
    .await?;

    download_file_to_path(
        "https://download.geonames.org/export/dump/admin2Codes.txt",
        extracted_dir,
    )
    .await?;

    download_zip_to_path_and_unzip(
        "https://download.geonames.org/export/dump/adminCode5.zip",
        extracted_dir,
    )
    .await?;
    download_zip_to_path_and_unzip(
        "https://download.geonames.org/export/dump/allCountries.zip",
        extracted_dir,
    )
    .await?;
    download_zip_to_path_and_unzip(
        "https://download.geonames.org/export/dump/alternateNamesV2.zip",
        extracted_dir,
    )
    .await?;

    download_file_to_path(
        "https://download.geonames.org/export/dump/countryInfo.txt",
        extracted_dir,
    )
    .await?;

    download_file_to_path(
        "https://download.geonames.org/export/dump/featureCodes_en.txt",
        extracted_dir,
    )
    .await?;

    download_zip_to_path_and_unzip(
        "https://download.geonames.org/export/dump/hierarchy.zip",
        extracted_dir,
    )
    .await?;

    download_zip_to_path_and_unzip(
        "https://download.geonames.org/export/dump/no-country.zip",
        extracted_dir,
    )
    .await?;

    download_zip_to_path_and_unzip(
        "https://download.geonames.org/export/dump/shapes_simplified_low.json.zip",
        extracted_dir,
    )
    .await?;

    download_file_to_path(
        "https://download.geonames.org/export/dump/timeZones.txt",
        extracted_dir,
    )
    .await?;

    download_zip_to_path_and_unzip(
        "https://download.geonames.org/export/dump/userTags.zip",
        extracted_dir,
    )
    .await?;

    // let schema = Schema::from_iter(vec![
    //     Field::new("geonameid", DataType::Int32),
    //     Field::new("name", DataType::String),
    //     Field::new("asciiname", DataType::String),
    //     Field::new("alternatenames", DataType::String),
    //     Field::new("latitude", DataType::Float64),
    //     Field::new("longitude", DataType::Float64),
    //     Field::new("feature_class", DataType::String),
    //     Field::new("feature_code", DataType::String),
    //     Field::new("country_code", DataType::String),
    //     Field::new("cc2", DataType::String),
    //     Field::new("admin1_code", DataType::String),
    //     Field::new("admin2_code", DataType::String),
    //     Field::new("admin3_code", DataType::String),
    //     Field::new("admin4_code", DataType::String),
    //     Field::new("population", DataType::Int64),
    //     Field::new("elevation", DataType::Int32),
    //     Field::new("dem", DataType::Int32),
    //     Field::new("timezone", DataType::String),
    //     Field::new(
    //         "modification_date",
    //         DataType::Datetime(TimeUnit::Microseconds, None),
    //     ),
    // ]);

    // let q = LazyCsvReader::new("data/geonames/allCountries.txt")
    //     .has_header(false)
    //     .with_separator(b'\t')
    //     .truncate_ragged_lines(true)
    //     .with_schema(Some(Arc::new(schema)))
    //     .finish()?;

    // let df = q.collect()?;

    // println!("{:#?}", df);

    Ok(())
}
