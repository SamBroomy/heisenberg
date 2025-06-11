- Heisenberg: A location enrichment crate/package named after Walter White from Breaking Bad
- Inspired by previous company's ML model for World Wide (WW) location enrichment
- Originally developed in Python using Polars and sentence embeddings with GADM dataset
- Rust reimplementation with key changes:
  - Switched to Tantivy full-text search instead of embeddings for improved speed and reliability
  - Changed data source to Geonames Dataset
  - Leverages Geonames' alternative names feature, eliminating the need for embeddings to handle name variations (e.g., "deuchland" � "germany")

- Input Vector Characteristics:
  - Requires input vector with locations in descending 'size' order
  - Largest locations should be first, smallest/most specific location last
  - Example input order: ['Country', 'State', 'County', 'Place']

- Location Enrichment Purpose:
  - Designed to fill in missing or inconsistent location information
  - Uses Geonames Dataset to backfill and standardize location data
  - Handles cases with partial or inconsistent location information
  - Example: ['Florida', 'Lakeland'] can be enriched to ['United States', 'Florida', 'Polk County', 'Lakeland']

- Data Source Caveat:
  - Uses Geonames Dataset for location hierarchies
  - Location accuracy depends on Geonames' understanding
  - Does not process specific address information
  - Focuses on location name standardization and enrichment
