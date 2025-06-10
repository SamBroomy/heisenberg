//! Full-text search indexing system for geographic location data.
//!
//! This module provides a flexible indexing framework built on top of Tantivy
//! for efficient text search across geographic datasets. It supports different
//! index types for administrative entities and places, with customizable
//! field weights and search parameters.

use std::path::{Path, PathBuf};

pub(crate) use error::IndexError;
use error::Result;
use itertools::izip;
use polars::prelude::{DataFrame, DataType, LazyFrame, col};
use tantivy::{
    Index, IndexWriter, TantivyDocument, Term,
    collector::TopDocs,
    query::{BooleanQuery, BoostQuery, FuzzyTermQuery, Occur, Query, QueryParser, TermSetQuery},
    schema::{
        FAST, Field, INDEXED, IndexRecordOption, STORED, Schema, SchemaBuilder, TextFieldIndexing,
        TextOptions, Value,
    },
};
use tracing::{debug, info, instrument, trace, warn};

use crate::data::LocationSearchData;
#[derive(Debug, Clone)]
pub struct LocationSearchIndex {
    pub admin: FTSIndex<AdminIndexDef>,
    pub places: FTSIndex<PlacesIndexDef>,
}

impl LocationSearchIndex {
    pub const INDEX_DIR: &str = "tantivy_indexes";

    /// Load existing if available and up-to-date, otherwise create new
    #[instrument(
        name = "Smart Initialize Location Search Index",
        skip(data),
        fields(data_source = ?data.data_source())
    )]
    pub fn initialize(data: &LocationSearchData) -> Result<Self> {
        // Try to load existing first
        if let Some(existing) = Self::load_existing(data.data_source())? {
            // Check if it's up to date
            if existing.is_up_to_date(data)? {
                info!("Using existing up-to-date indexes");
                return Ok(existing);
            } else {
                info!("Existing indexes are out of date, recreating");
            }
        } else {
            info!("No existing indexes found, creating new");
        }

        // Create new indexes
        Self::new(data, false)
    }

    /// Create or load the full-text search index for locations.
    ///
    /// This initializes both the administrative and places indexes, either loading
    /// existing indexes or creating new ones from the provided data.
    #[instrument(
        name = "Create Location Search Index",
        skip(data),
        fields(index_name = "location_search")
    )]
    pub fn new(data: &LocationSearchData, overwrite: bool) -> Result<Self> {
        info!("Initializing location search index");

        // Get the index path
        let base_index_path = data.data_source().data_source_dir().join(Self::INDEX_DIR);

        // Ensure base directory exists
        std::fs::create_dir_all(&base_index_path)?;

        let admin_index_path = base_index_path.join("admin_search");
        let places_index_path = base_index_path.join("places_search");

        // Get the data for each index
        let admin_data = data.admin_search_df();
        let places_data = data.place_search_df();

        // Initialize both indexes
        info!("Creating admin search index");
        let admin_index = FTSIndex::new(AdminIndexDef, admin_data, &admin_index_path, overwrite)?;

        info!("Creating places search index");
        let places_index =
            FTSIndex::new(PlacesIndexDef, places_data, &places_index_path, overwrite)?;

        info!("Location search index initialization complete");

        Ok(LocationSearchIndex {
            admin: admin_index,
            places: places_index,
        })
    }

    /// Load existing indexes without creating new ones
    #[instrument(
        name = "Load Location Search Index",
        skip(data_source),
        fields(data_source = ?data_source)
    )]
    pub fn load_existing(
        data_source: &heisenberg_data_processing::DataSource,
    ) -> Result<Option<Self>> {
        let base_index_path = data_source.data_source_dir().join(Self::INDEX_DIR);

        let admin_index_path = base_index_path.join("admin_search");
        let places_index_path = base_index_path.join("places_search");

        // Try to load both indexes
        let admin_index = FTSIndex::load_existing(AdminIndexDef, &admin_index_path)?;
        let places_index = FTSIndex::load_existing(PlacesIndexDef, &places_index_path)?;

        match (admin_index, places_index) {
            (Some(admin), Some(places)) => {
                info!("Successfully loaded both existing indexes");
                Ok(Some(LocationSearchIndex { admin, places }))
            }
            _ => {
                info!("One or both indexes missing, cannot load existing");
                Ok(None)
            }
        }
    }

    /// Check if existing indexes are up-to-date with the provided data
    pub fn is_up_to_date(&self, data: &LocationSearchData) -> Result<bool> {
        let admin_data = data.admin_search_df();
        let places_data = data.place_search_df();

        let admin_current = self.admin.is_up_to_date(&admin_data)?;
        let places_current = self.places.is_up_to_date(&places_data)?;

        Ok(admin_current && places_current)
    }

    /// Force recreation of both indexes
    pub fn recreate(data: &LocationSearchData) -> Result<Self> {
        Self::new(data, true)
    }

    /// Get the index directory for a given data source
    pub fn index_dir_for_source(data_source: &heisenberg_data_processing::DataSource) -> PathBuf {
        data_source.data_source_dir().join(Self::INDEX_DIR)
    }

    /// Check if indexes exist for a given data source
    pub fn exists_for_source(data_source: &heisenberg_data_processing::DataSource) -> bool {
        let base_path = Self::index_dir_for_source(data_source);
        let admin_path = base_path.join("admin_search").join("meta.json");
        let places_path = base_path.join("places_search").join("meta.json");

        admin_path.exists() && places_path.exists()
    }
}

/// Trait defining how to build and search a specific type of location index.
///
/// This trait allows different location datasets (admin entities vs places) to define
/// their own indexing strategies, field mappings, and search configurations while
/// sharing the same underlying search infrastructure.
pub trait IndexDefinition: std::fmt::Debug + Send + Sync + Clone + 'static {
    /// Returns the unique name for this index, used for directory and file naming.
    fn name(&self) -> &'static str;

    /// Defines the Tantivy schema for this index.
    fn schema(&self) -> Schema;

    /// Lists the columns to be selected from the source Parquet file for indexing.
    fn columns_for_indexing(&self) -> Vec<&'static str>;

    /// Populates the Tantivy index from the given DataFrame.
    /// This method is responsible for iterating through the DataFrame,
    /// creating TantivyDocuments, and adding them to the IndexWriter.
    fn index_data(&self, writer: &mut IndexWriter, data: DataFrame, schema: &Schema) -> Result<()>;

    /// Returns the default fields to be queried in a search.
    fn default_query_fields(&self, schema: &Schema) -> Vec<Field>;

    /// Returns a list of fields and their respective boost factors for querying.
    fn field_boosts(&self, schema: &Schema) -> Vec<(Field, f32)>;

    /// Returns a list of fields (typically ID codes) and their boost factors for exact matches.
    fn code_like_fields_with_boosts(&self, schema: &Schema) -> Vec<(Field, f32)>;
}

/// Index definition for administrative entities (countries, states, provinces, etc.).
///
/// Optimized for searching administrative hierarchies with support for official names,
/// alternate names, and various country/region codes (ISO, FIPS, etc.).
#[derive(Debug, Clone, Default)]
pub struct AdminIndexDef;

impl IndexDefinition for AdminIndexDef {
    fn name(&self) -> &'static str {
        "admin_search"
    }

    fn schema(&self) -> Schema {
        let mut schema_builder = SchemaBuilder::new();

        // Configure text indexing with stemming and position tracking
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default().set_indexing_options(text_indexing);

        // Configure exact matching for codes (no stemming)
        let code_options = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );

        schema_builder.add_u64_field("geonameId", STORED | INDEXED | FAST);
        schema_builder.add_text_field("name", text_options.clone());
        schema_builder.add_text_field("asciiname", text_options.clone());
        schema_builder.add_text_field("alternatenames", text_options.clone());
        schema_builder.add_text_field("official_name", text_options.clone());
        // Use text fields for codes to allow for easier querying, but with 'raw' tokenizer
        schema_builder.add_text_field("ISO", code_options.clone());
        schema_builder.add_text_field("ISO3", code_options.clone());
        schema_builder.add_text_field("fips", code_options);
        schema_builder.build()
    }

    fn columns_for_indexing(&self) -> Vec<&'static str> {
        vec![
            "geonameId",
            "name",
            "asciiname",
            "alternatenames",
            "official_name",
            "ISO",
            "ISO3",
            "fips",
        ]
    }

    fn index_data(&self, writer: &mut IndexWriter, df: DataFrame, schema: &Schema) -> Result<()> {
        let geoname_id_series = df.column("geonameId")?.cast(&DataType::UInt64)?;
        let geoname_id_series = geoname_id_series.u64()?;
        let name_series = df.column("name")?.str()?;
        let asciiname_series = df.column("asciiname")?.str()?;
        let alternatenames_series = df.column("alternatenames")?.list()?;
        let official_name_series = df.column("official_name")?.str()?;
        let iso_series = df.column("ISO")?.str()?;
        let iso3_series = df.column("ISO3")?.str()?;
        let fips_series = df.column("fips")?.str()?;

        let f_gid = schema.get_field("geonameId")?;
        let f_name = schema.get_field("name")?;
        let f_ascii = schema.get_field("asciiname")?;
        let f_alt = schema.get_field("alternatenames")?;
        let f_official = schema.get_field("official_name")?;
        let f_iso = schema.get_field("ISO")?;
        let f_iso3 = schema.get_field("ISO3")?;
        let f_fips = schema.get_field("fips")?;

        for (gid, name, asciiname, alternatenames, official_name, iso, iso3, fips) in izip!(
            geoname_id_series,
            name_series,
            asciiname_series,
            alternatenames_series,
            official_name_series,
            iso_series,
            iso3_series,
            fips_series
        ) {
            if let (Some(gid_val), Some(name_val)) = (gid, name) {
                let mut doc = TantivyDocument::default();
                doc.add_u64(f_gid, gid_val);
                doc.add_text(f_name, name_val);
                if let Some(val) = asciiname {
                    doc.add_text(f_ascii, val);
                }
                if let Some(val) = alternatenames {
                    for alt in val.str()?.iter().flatten() {
                        doc.add_text(f_alt, alt);
                    }
                }
                if let Some(val) = official_name {
                    doc.add_text(f_official, val);
                }
                if let Some(val) = iso {
                    doc.add_text(f_iso, val);
                }
                if let Some(val) = iso3 {
                    doc.add_text(f_iso3, val);
                }
                if let Some(val) = fips {
                    doc.add_text(f_fips, val);
                }
                writer.add_document(doc)?;
            }
        }
        Ok(())
    }

    fn default_query_fields(&self, schema: &Schema) -> Vec<Field> {
        vec![
            schema.get_field("name").unwrap(),
            schema.get_field("asciiname").unwrap(),
            schema.get_field("alternatenames").unwrap(),
            schema.get_field("official_name").unwrap(),
        ]
    }

    fn field_boosts(&self, schema: &Schema) -> Vec<(Field, f32)> {
        vec![
            (schema.get_field("name").unwrap(), 3.0),
            (schema.get_field("asciiname").unwrap(), 2.0),
            (schema.get_field("alternatenames").unwrap(), 1.0),
            (schema.get_field("official_name").unwrap(), 4.0),
        ]
    }

    fn code_like_fields_with_boosts(&self, schema: &Schema) -> Vec<(Field, f32)> {
        vec![
            (schema.get_field("ISO3").unwrap(), 1000.0),
            (schema.get_field("ISO").unwrap(), 800.0),
            (schema.get_field("fips").unwrap(), 400.0),
        ]
    }
}

/// Index definition for places (cities, towns, landmarks, points of interest).
///
/// Optimized for place name searching with focus on primary names, ASCII variants,
/// and alternate names. Lighter weight than admin index since places typically
/// don't have complex code systems.
#[derive(Debug, Clone, Default)]
pub struct PlacesIndexDef;

impl IndexDefinition for PlacesIndexDef {
    fn name(&self) -> &'static str {
        "places_search"
    }

    fn schema(&self) -> Schema {
        let mut schema_builder = SchemaBuilder::new();
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default().set_indexing_options(text_indexing);

        schema_builder.add_u64_field("geonameId", STORED | INDEXED | FAST);
        schema_builder.add_text_field("name", text_options.clone());
        schema_builder.add_text_field("asciiname", text_options.clone());
        schema_builder.add_text_field("alternatenames", text_options);
        schema_builder.build()
    }

    fn columns_for_indexing(&self) -> Vec<&'static str> {
        vec!["geonameId", "name", "asciiname", "alternatenames"]
    }

    fn index_data(&self, writer: &mut IndexWriter, df: DataFrame, schema: &Schema) -> Result<()> {
        let geoname_id_series = df.column("geonameId")?.cast(&DataType::UInt64)?;
        let geoname_id_series = geoname_id_series.u64()?;
        let name_series = df.column("name")?.str()?;
        let asciiname_series = df.column("asciiname")?.str()?;
        let alternatenames_series = df.column("alternatenames")?.list()?;

        let f_gid = schema.get_field("geonameId")?;
        let f_name = schema.get_field("name")?;
        let f_ascii = schema.get_field("asciiname")?;
        let f_alt = schema.get_field("alternatenames")?;

        for (gid, name, asciiname, alternatenames) in izip!(
            geoname_id_series,
            name_series,
            asciiname_series,
            alternatenames_series
        ) {
            if let (Some(gid_val), Some(name_val)) = (gid, name) {
                let mut doc = TantivyDocument::default();
                doc.add_u64(f_gid, gid_val);
                doc.add_text(f_name, name_val);
                if let Some(val) = asciiname {
                    doc.add_text(f_ascii, val);
                }
                if let Some(val) = alternatenames {
                    for alt in val.str()?.iter().flatten() {
                        doc.add_text(f_alt, alt);
                    }
                }
                writer.add_document(doc)?;
            }
        }
        Ok(())
    }

    fn default_query_fields(&self, schema: &Schema) -> Vec<Field> {
        vec![
            schema.get_field("name").unwrap(),
            schema.get_field("asciiname").unwrap(),
            schema.get_field("alternatenames").unwrap(),
        ]
    }

    fn field_boosts(&self, schema: &Schema) -> Vec<(Field, f32)> {
        vec![
            (schema.get_field("name").unwrap(), 3.0),
            (schema.get_field("asciiname").unwrap(), 2.0),
            (schema.get_field("alternatenames").unwrap(), 1.0),
        ]
    }

    fn code_like_fields_with_boosts(&self, _schema: &Schema) -> Vec<(Field, f32)> {
        vec![]
    }
}

/// Parameters controlling full-text search behavior and performance.
#[derive(Debug, Clone, Copy)]
pub struct FTSIndexSearchParams {
    /// The maximum number of results to return.
    pub limit: usize,
    /// Whether to enable fuzzy search for handling typos and variations.
    /// More expensive but useful for handling user input errors.
    pub fuzzy_search: bool,
}

impl Default for FTSIndexSearchParams {
    fn default() -> Self {
        Self {
            limit: 20,
            fuzzy_search: false,
        }
    }
}

/// Generic full-text search index supporting different location datasets.
///
/// Wraps Tantivy indexing with location-specific optimizations and provides
/// efficient search capabilities with support for:
/// - Multi-field text search with custom weights
/// - Fuzzy matching for typo tolerance
/// - Exact code matching for identifiers
/// - Subset searching within filtered document sets
#[derive(Debug, Clone)]
pub struct FTSIndex<D: IndexDefinition> {
    index: Index,
    definition: D,
}

impl<D: IndexDefinition> FTSIndex<D> {
    /// Try to load an existing index from the given path
    #[instrument(name = "Load Index", skip(definition), fields(index_name = definition.name()))]
    pub fn load_existing(definition: D, index_path: &Path) -> Result<Option<Self>> {
        let meta_path = index_path.join("meta.json");

        if !meta_path.exists() {
            debug!(path = ?index_path, "No meta.json found, index does not exist");
            return Ok(None);
        }

        info!(path = ?index_path, "Attempting to load existing index");

        match Index::open_in_dir(index_path) {
            Ok(existing_index) => {
                info!(path = ?index_path, "Successfully loaded existing index");
                Ok(Some(FTSIndex {
                    index: existing_index,
                    definition,
                }))
            }
            Err(e) => {
                warn!(path = ?index_path, error = ?e, "Failed to open existing index");
                Ok(None)
            }
        }
    }

    /// Check if an existing index is up-to-date with the provided data
    #[instrument(name = "Validate Index", skip(self, data), fields(index_name = self.definition.name()))]
    pub fn is_up_to_date(&self, data: &LazyFrame) -> Result<bool> {
        let expected_doc_count = data.clone().collect()?.shape().0;
        let actual_doc_count = self.index.reader()?.searcher().num_docs() as usize;

        let is_current = actual_doc_count == expected_doc_count;

        info!(
            index_name = self.definition.name(),
            actual_doc_count,
            expected_doc_count,
            is_up_to_date = is_current,
            "Index validation complete"
        );

        Ok(is_current)
    }

    /// Create a new index from scratch at the given path
    #[instrument(name = "Create Index", skip(definition, data), fields(index_name = definition.name()))]
    pub fn create_new(definition: D, data: LazyFrame, index_path: &Path) -> Result<Self> {
        info!(path = ?index_path, "Creating new FTS index");

        // Ensure directory exists and is clean
        Self::safely_recreate_dir(index_path)?;

        let schema = definition.schema();
        let index = Index::create_in_dir(index_path, schema.clone())?;

        let data_for_indexing = data
            .select(
                definition
                    .columns_for_indexing()
                    .into_iter()
                    .map(col)
                    .collect::<Vec<_>>(),
            )
            .collect()?;

        if !data_for_indexing.is_empty() {
            info!(
                index = definition.name(),
                num_rows = data_for_indexing.height(),
                "Populating index"
            );
            let mut index_writer: IndexWriter = index.writer(500_000_000)?;
            definition.index_data(&mut index_writer, data_for_indexing, &schema)?;
            index_writer.commit()?;
            info!(index = definition.name(), "Index creation complete");
        } else {
            warn!(
                index = definition.name(),
                "No data to index. Index will be empty."
            );
        }

        Ok(FTSIndex { index, definition })
    }

    /// Create or load a full-text search index.
    ///
    /// If the index exists and is up-to-date, loads the existing index.
    /// Otherwise, builds a new index from the provided data.
    #[instrument(name = "Initialize Index", skip(definition, data), fields(index_name = definition.name()))]
    pub fn new(definition: D, data: LazyFrame, index_path: &Path, overwrite: bool) -> Result<Self> {
        info!(path = ?index_path, overwrite, "Initializing FTS index");

        if overwrite {
            info!(path = ?index_path, "Overwrite requested, creating new index");
            return Self::create_new(definition, data, index_path);
        }

        // Try to load existing index
        if let Some(existing_index) = Self::load_existing(definition.clone(), index_path)? {
            // Check if it's up to date
            if existing_index.is_up_to_date(&data)? {
                info!(path = ?index_path, "Existing index is up-to-date");
                return Ok(existing_index);
            } else {
                info!(path = ?index_path, "Existing index is out of date, recreating");
                return Self::create_new(definition, data, index_path);
            }
        }

        // No existing index found, create new one
        info!(path = ?index_path, "No existing index found, creating new");
        Self::create_new(definition, data, index_path)
    }

    fn safely_recreate_dir(path: &std::path::Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_dir_all(path)?;
        }
        std::fs::create_dir_all(path)?;
        Ok(())
    }

    /// Search within a specific subset of documents.
    ///
    /// More efficient than general search when you know which documents are relevant,
    /// such as when filtering by administrative hierarchy or geographic bounds.
    pub fn search_in_subset(
        &self,
        query_str: &str,
        doc_ids: &[u64],
        params: &FTSIndexSearchParams,
    ) -> Result<Vec<(u64, f32)>> {
        self.search_inner(query_str, Some(doc_ids), params)
    }

    /// Build the complete search query combining text search, fuzzy matching, and exact code matching.
    ///
    /// Creates a sophisticated query that:
    /// - Searches across all configured text fields with appropriate weights
    /// - Adds fuzzy matching for longer terms to handle typos
    /// - Includes exact code matching for short terms (likely abbreviations/codes)
    /// - Filters to document subset if provided
    #[instrument(name = "Build Base Query", skip_all, level = "trace")]
    fn build_base_query(
        &self,
        query_str: &str,
        doc_ids: Option<&[u64]>,
        schema: &Schema,
        gid_field: Field,
        params: &FTSIndexSearchParams,
    ) -> Result<Box<dyn Query>> {
        let query_str = query_str.trim();
        if query_str.is_empty() {
            return Err(anyhow::anyhow!("Query string is empty.").into());
        }
        if params.limit == 0 {
            return Err(anyhow::anyhow!("Search limit must be greater than zero.").into());
        }

        let default_query_fields = self.definition.default_query_fields(schema);
        if default_query_fields.is_empty() {
            return Err(anyhow::anyhow!(
                "No default query fields defined for index '{}'",
                self.definition.name()
            )
            .into());
        }

        let mut general_query_parser = QueryParser::for_index(&self.index, default_query_fields);
        for (field, boost) in self.definition.field_boosts(schema) {
            general_query_parser.set_field_boost(field, boost);
        }
        let (general_fts_query, errors) = general_query_parser.parse_query_lenient(query_str);
        if errors.is_empty() {
            trace!(parsed_query = ?general_fts_query, "General FTS query parsed");
            trace!("Parsed query: {:?}", general_fts_query);
        } else {
            warn!(?errors, "Query parsing errors occurred");
        }

        let mut query_clauses: Vec<(Occur, Box<dyn Query>)> =
            vec![(Occur::Should, general_fts_query)];

        let is_single_short_token =
            !query_str.contains(char::is_whitespace) && query_str.len() <= 3;

        if params.fuzzy_search && !is_single_short_token {
            let query_terms: Vec<&str> = query_str.split_whitespace().collect();
            for term_str in query_terms {
                if term_str.len() > 2 {
                    let fuzzy_distance = 1;
                    let fuzzy_transpositions = true;
                    for field_name_for_fuzzy in ["name", "asciiname"] {
                        if let Ok(field_for_fuzzy) = schema.get_field(field_name_for_fuzzy) {
                            let term = Term::from_field_text(field_for_fuzzy, term_str);
                            let fuzzy_query =
                                FuzzyTermQuery::new(term, fuzzy_distance, fuzzy_transpositions);
                            query_clauses.push((
                                Occur::Should,
                                Box::new(BoostQuery::new(Box::new(fuzzy_query), 1.5)),
                            ));
                        }
                    }
                }
            }
        }

        let code_like_fields = self.definition.code_like_fields_with_boosts(schema);
        if is_single_short_token && !code_like_fields.is_empty() {
            let lower_case_code_query = query_str.to_lowercase();
            for (field, boost) in code_like_fields {
                let term = Term::from_field_text(field, &lower_case_code_query);
                let exact_query =
                    tantivy::query::TermQuery::new(term.clone(), IndexRecordOption::Basic);
                query_clauses.push((
                    Occur::Should,
                    Box::new(BoostQuery::new(Box::new(exact_query), boost)),
                ));
            }
        }

        let combined_search_query = BooleanQuery::new(query_clauses);

        let final_query = match doc_ids {
            Some(ids) if !ids.is_empty() => {
                let doc_id_terms: Vec<Term> = ids
                    .iter()
                    .map(|&id| Term::from_field_u64(gid_field, id))
                    .collect();
                if doc_id_terms.is_empty() {
                    // Should not happen if ids is not empty
                    Box::new(combined_search_query)
                } else {
                    let doc_id_filter = TermSetQuery::new(doc_id_terms);
                    Box::new(BooleanQuery::new(vec![
                        (Occur::Must, Box::new(combined_search_query)),
                        (Occur::Must, Box::new(doc_id_filter)),
                    ]))
                }
            }
            _ => Box::new(combined_search_query),
        };
        trace!(?final_query, "Final query constructed");
        Ok(final_query)
    }

    /// Internal search implementation handling both general and subset search.
    #[instrument(name="Search Text Index",
        skip_all, level = "debug", fields(index_name = self.definition.name(), query = query_str, limit = params.limit, has_subset = doc_ids.is_some()))]
    fn search_inner(
        &self,
        query_str: &str,
        doc_ids: Option<&[u64]>,
        params: &FTSIndexSearchParams,
    ) -> Result<Vec<(u64, f32)>> {
        let schema = self.index.schema();
        let gid_field = schema.get_field("geonameId")?;

        let query = self.build_base_query(query_str, doc_ids, &schema, gid_field, params)?;

        let reader = self.index.reader()?;
        let searcher = reader.searcher();

        let t_search = std::time::Instant::now();
        let top_docs = searcher.search(&*query, &TopDocs::with_limit(params.limit))?;
        let search_duration = t_search.elapsed();
        debug!(
            num_results = top_docs.len(),
            search_execution_seconds = search_duration.as_secs_f32(),
            "Tantivy search execution complete"
        );

        top_docs
            .into_iter()
            .map(|(score, doc_address)| {
                let received_doc = searcher.doc::<TantivyDocument>(doc_address)?;
                let doc_id_val = received_doc
                    .get_first(gid_field)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        anyhow::anyhow!("Failed to get geonameId from document: {:?}", received_doc)
                    })?;

                Ok((doc_id_val, score))
            })
            .collect::<Result<Vec<_>>>()
    }
}

mod error {
    use thiserror::Error;

    use super::super::data::error::HeisenbergDataError;

    #[derive(Error, Debug)]
    pub enum IndexError {
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),
        #[error("Tantivy error: {0}")]
        Tantivy(#[from] tantivy::TantivyError),
        #[error("DataFrame error: {0}")]
        DataFrame(#[from] polars::prelude::PolarsError),
        #[error(transparent)]
        Other(#[from] anyhow::Error),
        #[error("Heisenberg data processing error: {0}")]
        DataError(#[from] HeisenbergDataError),
    }
    pub type Result<T> = std::result::Result<T, IndexError>;
}
