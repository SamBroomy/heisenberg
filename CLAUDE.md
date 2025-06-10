- Heisenberg: A location enrichment crate/package named after Walter White from Breaking Bad
- Inspired by previous company's ML model for World Wide (WW) location enrichment
- Originally developed in Python using Polars and sentence embeddings with GADM dataset
- Rust reimplementation with key changes:
  * Switched to Tantivy full-text search instead of embeddings for improved speed and reliability
  * Changed data source to Geonames Dataset
  * Leverages Geonames' alternative names feature, eliminating the need for embeddings to handle name variations (e.g., "deuchland" ’ "germany")