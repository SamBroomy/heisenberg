use anyhow::{Context, Result};
use itertools::izip;
use polars::prelude::{col, LazyFrame};
use polars::prelude::{DataFrame, DataType};
use tantivy::schema::Field;
use tantivy::{
    collector::TopDocs,
    query::{BooleanQuery, BoostQuery, FuzzyTermQuery, Occur, Query, QueryParser, TermSetQuery},
    schema::{
        IndexRecordOption, Schema, SchemaBuilder, TextFieldIndexing, TextOptions, Value, FAST,
        INDEXED, STORED,
    },
    Index, IndexWriter, TantivyDocument, Term,
};
use tracing::{debug, info, instrument, trace, trace_span, warn};

pub trait IndexDefinition: std::fmt::Debug + Send + Sync + 'static {
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

#[derive(Debug, Clone, Default)]
pub struct AdminIndexDef;

impl IndexDefinition for AdminIndexDef {
    fn name(&self) -> &'static str {
        "admin_search"
    }

    fn schema(&self) -> Schema {
        let mut schema_builder = SchemaBuilder::new();
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default().set_indexing_options(text_indexing);
        let code_options = TextOptions::default() // For exact code matching, no stemming
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
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
            .set_index_option(IndexRecordOption::Basic);
        let text_options = TextOptions::default().set_indexing_options(text_indexing);

        schema_builder.add_u64_field("geonameId", STORED | INDEXED | FAST);
        schema_builder.add_text_field("name", text_options.clone());
        schema_builder.add_text_field("asciiname", text_options.clone());
        schema_builder.add_text_field("alternatenames", text_options);
        // Places search might not have official_name, ISO codes etc.
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

#[derive(Debug, Clone, Copy)]
pub struct FTSIndexSearchParams {
    /// The maximum number of results to return.
    pub limit: usize,
    /// Whether to enable fuzzy search, is more expensive, but can be useful for typos.
    /// Fuzzy search is only applied to the name and asciiname fields.
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

#[derive(Debug, Clone)]
pub struct FTSIndex<D: IndexDefinition> {
    index: Index,
    definition: D,
}

impl<D: IndexDefinition> FTSIndex<D> {
    #[instrument(name = "Create Index", skip(definition, data), fields(index_name = definition.name()))]
    pub fn new(definition: D, data: LazyFrame, overwrite: bool) -> Result<Self> {
        let index_path = crate::data::get_data_dir()
            .join("tantivy")
            .join(definition.name());

        if index_path.exists() && !overwrite {
            info!(index = definition.name(), "Already exists. Loading.",);
            let index = Index::open_in_dir(&index_path).with_context(|| {
                format!("Failed to open existing index '{}'", definition.name())
            })?;
            if index.reader().unwrap().searcher().num_docs() as usize
                == data.clone().collect()?.shape().0
            {
                info!(index = definition.name(), "Up to date");
                return Ok(FTSIndex { index, definition });
            }
            info!(index = definition.name(), "Not up to date, reindexing");
        }

        if index_path.exists() && overwrite {
            info!(index = definition.name(), "Deleting existing index");
            std::fs::remove_dir_all(&index_path).with_context(|| {
                format!("Failed to remove existing index '{}'", definition.name())
            })?;
        }

        std::fs::create_dir_all(index_path.parent().unwrap())
            .with_context(|| "Failed to create base directory for indexes".to_string())?;
        std::fs::create_dir_all(&index_path).with_context(|| {
            format!(
                "Failed to create directory for index '{}'",
                definition.name()
            )
        })?;

        let schema = definition.schema();
        let index = Index::create_in_dir(&index_path, schema.clone()).with_context(|| {
            format!(
                "Failed to create Tantivy index structure for '{}'",
                definition.name()
            )
        })?;

        let data = data
            .select(
                definition
                    .columns_for_indexing()
                    .into_iter()
                    .map(col)
                    .collect::<Vec<_>>(),
            )
            .collect()
            .with_context(|| "Failed to collect DataFrame".to_string())?;

        let mut index_writer: IndexWriter = index.writer(500_000_000).with_context(|| {
            format!("Failed to create index writer for '{}'", definition.name())
        })?;

        info!("Populating index '{}'...", definition.name());
        definition.index_data(&mut index_writer, data, &schema)?;

        info!("Committing index '{}'...", definition.name());
        index_writer.commit()?;
        Ok(FTSIndex { index, definition })
    }

    pub fn search(&self, query: &str, params: &FTSIndexSearchParams) -> Result<Vec<(u64, f32)>> {
        self.search_inner(query, None, params)
    }

    pub fn search_in_subset(
        &self,
        query_str: &str,
        doc_ids: &[u64],
        params: &FTSIndexSearchParams,
    ) -> Result<Vec<(u64, f32)>> {
        self.search_inner(query_str, Some(doc_ids), params)
    }

    #[instrument(name="Search Text Index",
        skip_all, level = "debug", fields(index_name = self.definition.name(), query = query_str, limit = params.limit, has_subset = doc_ids.is_some()))]
    fn search_inner(
        &self,
        query_str: &str,
        doc_ids: Option<&[u64]>,
        params: &FTSIndexSearchParams,
    ) -> Result<Vec<(u64, f32)>> {
        let query_str = query_str.trim();
        if query_str.is_empty() {
            return Err(anyhow::anyhow!("Query string cannot be empty."));
        }
        if params.limit == 0 {
            return Err(anyhow::anyhow!("Search limit must be greater than zero."));
        }

        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        let schema = self.index.schema();

        let gid_field = schema
            .get_field("geonameId")
            .context("geonameId field not in schema")?;
        // let name_field = schema
        //     .get_field("name")
        //     .context("name field not in schema")?;

        let default_query_fields = self.definition.default_query_fields(&schema);
        if default_query_fields.is_empty() {
            return Err(anyhow::anyhow!(
                "No default query fields defined for index '{}'",
                self.definition.name()
            ));
        }

        let final_query: Box<dyn Query> = {
            let _span = trace_span!("construct_tantivy_query").entered();

            let mut general_query_parser =
                QueryParser::for_index(&self.index, default_query_fields);
            for (field, boost) in self.definition.field_boosts(&schema) {
                general_query_parser.set_field_boost(field, boost);
            }

            let (general_fts_query, errors) = general_query_parser.parse_query_lenient(query_str);
            if errors.is_empty() {
                trace!(parsed_query = ?general_fts_query, "General FTS query parsed");
                trace!("Parsed query: {:?}", general_fts_query);
            } else {
                warn!(?errors, "Query parsing errors occurred");
            }

            // let general_fts_query = general_query_parser
            //     .parse_query(query_str)
            //     .with_context(|| format!("Failed to parse query: '{}'", query_str))?;

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

            let code_like_fields = self.definition.code_like_fields_with_boosts(&schema);
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

            match doc_ids {
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
            }
        };

        trace!(tantivy_query = ?final_query, "Executing Tantivy query");
        let t_search = std::time::Instant::now();
        let top_docs = searcher.search(&*final_query, &TopDocs::with_limit(params.limit))?;
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
                // let name_val = received_doc
                //     .get_first(name_field)
                //     .and_then(|v| v.as_str())
                //     .unwrap_or("")
                //     .to_string();
                let doc_id_val = received_doc
                    .get_first(gid_field)
                    .and_then(|v| v.as_u64())
                    .context("Failed to get geonameId as u64 from document")?;
                Ok((doc_id_val, score))
            })
            .collect::<Result<Vec<_>>>()
    }
}
