use std::io::Write;

use tempfile::NamedTempFile;
use tracing::info;

use super::error::Result;

/// Configuration for test data generation
#[derive(Debug, Clone)]
pub struct TestDataConfig {
    /// Number of rows to include for all_countries data
    pub all_countries_rows: usize,
    /// Number of rows to include for country_info data
    pub country_info_rows: usize,
    /// Number of rows to include for feature_codes data
    pub feature_codes_rows: usize,
    /// Whether to use realistic data or minimal test data
    pub realistic_data: bool,
}

impl Default for TestDataConfig {
    fn default() -> Self {
        Self {
            all_countries_rows: 100,
            country_info_rows: 10,
            feature_codes_rows: 20,
            realistic_data: true,
        }
    }
}

impl TestDataConfig {
    /// Minimal data for unit tests
    pub fn minimal() -> Self {
        Self {
            all_countries_rows: 3,
            country_info_rows: 1,
            feature_codes_rows: 3,
            realistic_data: false,
        }
    }

    /// Sample data for integration tests
    pub fn sample() -> Self {
        Self {
            all_countries_rows: 100,
            country_info_rows: 10,
            feature_codes_rows: 50,
            realistic_data: true,
        }
    }
}

/// Create test data files in temporary files
/// This function generates temporary files containing test data based on the provided configuration.
/// It creates three files: all_countries, country_info, and feature_codes.
pub fn create_test_data(
    config: &TestDataConfig,
) -> Result<(NamedTempFile, NamedTempFile, NamedTempFile)> {
    info!("Creating test data with config: {:?}", config);

    let all_countries = create_all_countries_test_data(config)?;
    let country_info = create_country_info_test_data(config)?;
    let feature_codes = create_feature_codes_test_data(config)?;

    Ok((all_countries, country_info, feature_codes))
}

fn create_all_countries_test_data(config: &TestDataConfig) -> Result<NamedTempFile> {
    let mut file = NamedTempFile::new()?;

    if config.realistic_data && config.all_countries_rows > 10 {
        // Generate more realistic sample data
        write_realistic_all_countries_data(&mut file, config.all_countries_rows)?;
    } else {
        // Use minimal test data
        write_minimal_all_countries_data(&mut file, config.all_countries_rows)?;
    }

    file.flush()?;
    Ok(file)
}

fn create_country_info_test_data(config: &TestDataConfig) -> Result<NamedTempFile> {
    let mut file = NamedTempFile::new()?;

    // Add the expected 51 header lines
    for i in 1..=51 {
        writeln!(file, "# Header line {i}")?;
    }

    if config.realistic_data && config.country_info_rows > 3 {
        write_realistic_country_info_data(&mut file, config.country_info_rows)?;
    } else {
        write_minimal_country_info_data(&mut file, config.country_info_rows)?;
    }

    file.flush()?;
    Ok(file)
}

fn create_feature_codes_test_data(config: &TestDataConfig) -> Result<NamedTempFile> {
    let mut file = NamedTempFile::new()?;

    if config.realistic_data && config.feature_codes_rows > 5 {
        write_realistic_feature_codes_data(&mut file, config.feature_codes_rows)?;
    } else {
        write_minimal_feature_codes_data(&mut file, config.feature_codes_rows)?;
    }

    file.flush()?;
    Ok(file)
}

// Realistic data generators
fn write_realistic_all_countries_data(file: &mut NamedTempFile, rows: usize) -> Result<()> {
    let base_data = [
        (
            6252001,
            "United States",
            "United States",
            "US,USA,America",
            39.5,
            -98.35,
            "A",
            "PCLI",
            "US",
            "",
            "",
            "",
            "",
            "",
            331000000,
            0,
            0,
            "America/New_York",
        ),
        (
            5332921,
            "California",
            "California",
            "CA,Calif",
            36.17,
            -119.75,
            "A",
            "ADM1",
            "US",
            "",
            "CA",
            "",
            "",
            "",
            39538223,
            0,
            0,
            "America/Los_Angeles",
        ),
        (
            5391959,
            "San Francisco",
            "San Francisco",
            "SF,San Fran",
            37.7749,
            -122.4194,
            "P",
            "PPLA2",
            "US",
            "",
            "CA",
            "075",
            "",
            "",
            873965,
            16,
            16,
            "America/Los_Angeles",
        ),
        (
            5128581,
            "New York",
            "New York",
            "NYC,Big Apple",
            40.7128,
            -74.0060,
            "P",
            "PPLA",
            "US",
            "",
            "NY",
            "061",
            "",
            "",
            8336817,
            10,
            10,
            "America/New_York",
        ),
        (
            4164138,
            "Miami",
            "Miami",
            "MIA",
            25.7617,
            -80.1918,
            "P",
            "PPLA2",
            "US",
            "",
            "FL",
            "086",
            "",
            "",
            441003,
            2,
            2,
            "America/New_York",
        ),
        // International examples
        (
            6295630,
            "France",
            "France",
            "FR,French Republic",
            46.0,
            2.0,
            "A",
            "PCLI",
            "FR",
            "",
            "",
            "",
            "",
            "",
            67390000,
            0,
            0,
            "Europe/Paris",
        ),
        (
            2988507,
            "Paris",
            "Paris",
            "Paname,Ville Lumière",
            48.8566,
            2.3522,
            "P",
            "PPLC",
            "FR",
            "",
            "11", // Île-de-France
            "75", // Paris
            "",
            "",
            2161000,
            0,
            0,
            "Europe/Paris",
        ),
        (
            1861060,
            "Japan",
            "Japan",
            "JP,Nippon,Nihon",
            36.0,
            138.0,
            "A",
            "PCLI",
            "JP",
            "",
            "",
            "",
            "",
            "",
            125800000,
            0,
            0,
            "Asia/Tokyo",
        ),
        (
            1850147,
            "Tokyo",
            "Tokyo",
            "東京,Tōkyō",
            35.6762,
            139.6503,
            "P",
            "PPLC",
            "JP",
            "",
            "13", // Tokyo Prefecture
            "",
            "",
            "",
            13960000,
            0,
            0,
            "Asia/Tokyo",
        ),
        (
            2077456,
            "Australia",
            "Australia",
            "AU",
            -25.0,
            133.0,
            "A",
            "PCLI",
            "AU",
            "",
            "",
            "",
            "",
            "",
            25690000,
            0,
            0,
            "Australia/Sydney",
        ),
        (
            2147714,
            "Sydney",
            "Sydney",
            "Harbour City",
            -33.8688,
            151.2093,
            "P",
            "PPLA",
            "AU",
            "",
            "NSW", // New South Wales
            "",
            "",
            "",
            5312000,
            0,
            0,
            "Australia/Sydney",
        ),
    ];

    for (i, data) in base_data.iter().cycle().take(rows).enumerate() {
        let modified_id = data.0 + i as u32; // Ensure unique IDs
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t2023-01-01",
            modified_id,
            data.1,
            data.2,
            data.3,
            data.4,
            data.5,
            data.6,
            data.7,
            data.8,
            data.9,
            data.10,
            data.11,
            data.12,
            data.13,
            data.14,
            data.15,
            data.16,
            data.17
        )?;
    }
    Ok(())
}

fn write_realistic_country_info_data(file: &mut NamedTempFile, rows: usize) -> Result<()> {
    let base_data = [
        (
            "US",
            "USA",
            "840",
            "US",
            "United States",
            "Washington",
            "9629091",
            "331002651",
            "NA",
            ".us",
            "USD",
            "Dollar",
            "1",
            "#####-####",
            "^\\\\d{5}(-\\\\d{4})?$",
            "en-US,es-US",
            "6252001",
            "CA,MX",
            "",
        ),
        (
            "CA",
            "CAN",
            "124",
            "CA",
            "Canada",
            "Ottawa",
            "9984670",
            "37742154",
            "NA",
            ".ca",
            "CAD",
            "Dollar",
            "1",
            "",
            "",
            "en-CA,fr-CA",
            "6251999",
            "US",
            "",
        ),
        (
            "GB",
            "GBR",
            "826",
            "UK",
            "United Kingdom",
            "London",
            "243610",
            "67886011",
            "EU",
            ".uk",
            "GBP",
            "Pound",
            "44",
            "",
            "",
            "en-GB",
            "2635167",
            "IE",
            "",
        ),
        (
            "FR",
            "FRA",
            "250",
            "FR",
            "France",
            "Paris",
            "547030",
            "67390000",
            "EU",
            ".fr",
            "EUR",
            "Euro",
            "33",
            "",
            "",
            "fr-FR",
            "6295630",
            "ES,DE,BE,IT,CH,AD,LU,MC",
            "",
        ),
        (
            "JP",
            "JPN",
            "392",
            "JP",
            "Japan",
            "Tokyo",
            "377835",
            "125800000",
            "AS",
            ".jp",
            "JPY",
            "Yen",
            "81",
            "",
            "",
            "ja-JP",
            "1861060",
            "",
            "",
        ),
        (
            "AU",
            "AUS",
            "036",
            "AU",
            "Australia",
            "Canberra",
            "7686850",
            "25690000",
            "OC",
            ".au",
            "AUD",
            "Dollar",
            "61",
            "",
            "",
            "en-AU",
            "2077456",
            "",
            "",
        ),
    ];

    for data in base_data.iter().cycle().take(rows) {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            data.0,
            data.1,
            data.2,
            data.3,
            data.4,
            data.5,
            data.6,
            data.7,
            data.8,
            data.9,
            data.10,
            data.11,
            data.12,
            data.13,
            data.14,
            data.15,
            data.16,
            data.17,
            data.18
        )?;
    }
    Ok(())
}

fn write_realistic_feature_codes_data(file: &mut NamedTempFile, rows: usize) -> Result<()> {
    let base_data = [
        (
            "A.ADM1",
            "first-order administrative division",
            "a primary administrative division of a country",
        ),
        ("A.PCLI", "independent political entity", ""),
        (
            "P.PPLA2",
            "seat of a second-order administrative division",
            "",
        ),
        (
            "P.PPL",
            "populated place",
            "a city, town, village, or other agglomeration of buildings",
        ),
        (
            "T.MT",
            "mountain",
            "an elevation standing high above the surrounding area",
        ),
        ("H.LK", "lake", "a large inland body of standing water"),
        (
            "S.AIRP",
            "airport",
            "a place where aircraft regularly land and take off",
        ),
        ("P.PPLC", "capital of a political entity", ""),
        (
            "A.ADM2",
            "second-order administrative division",
            "county, district",
        ),
        (
            "A.ADM3",
            "third-order administrative division",
            "municipality",
        ),
        ("T.ISL", "island", "a tract of land surrounded by water"),
        ("R.RD", "road", "an open way with improved surface"),
        ("V.FRST", "forest", "an area covered with trees"),
        ("H.BAY", "bay", "a coastal indentation"),
        ("T.PK", "peak", "a pointed elevation"),
        ("L.CONT", "continent", "one of the seven large landmasses"),
        ("S.SCH", "school", "building where instruction is given"),
    ];

    for data in base_data.iter().cycle().take(rows) {
        writeln!(file, "{}\t{}\t{}", data.0, data.1, data.2)?;
    }
    Ok(())
}

// Minimal data generators (your existing test functions)
fn write_minimal_all_countries_data(file: &mut NamedTempFile, rows: usize) -> Result<()> {
    let data = [
        "6252001\tUnited States\tUnited States\tUS,USA,America\t39.5\t-98.35\tA\tPCLI\tUS\t\t\t\t\t\t331000000\t0\t0\tAmerica/New_York\t2023-01-01",
        "5332921\tCalifornia\tCalifornia\tCA,Calif\t36.17\t-119.75\tA\tADM1\tUS\t\tCA\t\t\t\t39538223\t0\t0\tAmerica/Los_Angeles\t2023-01-01",
        "5391959\tSan Francisco\tSan Francisco\tSF,San Fran\t37.7749\t-122.4194\tP\tPPLA2\tUS\t\tCA\t075\t\t\t873965\t16\t16\tAmerica/Los_Angeles\t2023-01-01",
    ];

    for line in data.iter().cycle().take(rows) {
        writeln!(file, "{line}")?;
    }
    Ok(())
}

fn write_minimal_country_info_data(file: &mut NamedTempFile, rows: usize) -> Result<()> {
    let data = [
        "US\tUSA\t840\tUS\tUnited States\tWashington\t9629091\t331002651\tNA\t.us\tUSD\tDollar\t1\t#####-####\t^\\\\d{5}(-\\\\d{4})?$\ten-US,es-US,haw,fr\t6252001\tCA,MX\t",
    ];

    for line in data.iter().cycle().take(rows) {
        writeln!(file, "{line}")?;
    }
    Ok(())
}

fn write_minimal_feature_codes_data(file: &mut NamedTempFile, rows: usize) -> Result<()> {
    let data = [
        "A.ADM1\tfirst-order administrative division\ta primary administrative division of a country",
        "A.PCLI\tindependent political entity\t",
        "P.PPLA2\tseat of a second-order administrative division\t",
    ];

    for line in data.iter().cycle().take(rows) {
        writeln!(file, "{line}")?;
    }
    Ok(())
}
