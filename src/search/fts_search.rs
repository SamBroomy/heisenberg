use anyhow::{Context, Result};
use polars::prelude::{col, DataType, LazyFrame};
use std::path::{Path, PathBuf};
use tracing::{debug, info, instrument, trace};

use itertools::izip;
use tantivy::{
    collector::TopDocs,
    query::{BooleanQuery, BoostQuery, FuzzyTermQuery, Occur, Query, QueryParser, TermSetQuery},
    schema::{
        IndexRecordOption, Schema, SchemaBuilder, TextFieldIndexing, TextOptions, Value, FAST,
        INDEXED, STORED,
    },
    Document, Index, IndexWriter, TantivyDocument, Term,
};

#[derive(Debug, Clone)]
pub enum FTSIndexes {
    AdminSearch,
    PlacesSearch,
}

impl FTSIndexes {
    fn get_index_name(&self) -> &'static str {
        match self {
            FTSIndexes::AdminSearch => "admin_search",
            FTSIndexes::PlacesSearch => "places_search",
        }
    }
    fn get_index_path(&self) -> PathBuf {
        let index_name = self.get_index_name();
        let path = Path::new("./data/indexes/tantivy").join(index_name);
        if !path.exists() {
            info!("Creating directory: {}", path.display());
            std::fs::create_dir_all(&path)
                .context("Failed to create directory")
                .unwrap();
        }
        path
    }
}
#[derive(Debug, Clone)]
pub struct FTSIndex {
    index: Index,
    index_name: FTSIndexes,
}
// Setting up the index
impl FTSIndex {
    pub fn new(index_name: FTSIndexes) -> Result<Self> {
        Self::new_inner(index_name, false)
    }

    pub fn overwrite_index(index_name: FTSIndexes) -> Result<Self> {
        Self::new_inner(index_name, true)
    }

    fn new_inner(index_name: FTSIndexes, overwrite: bool) -> Result<Self> {
        let index_path = index_name.get_index_path();

        // Check if the index already exists
        if index_path.exists() {
            if overwrite {
                info!("Deleting existing index...");
                std::fs::remove_dir_all(&index_path).context("Failed to remove existing index")?;
                std::fs::create_dir_all(&index_path).context("Failed to create directory")?;
            } else {
                info!("Index already exists. Skipping creation.");
                return Ok(FTSIndex {
                    index: Index::open_in_dir(index_path)
                        .context("Failed to open existing index")?,
                    index_name,
                });
            }
        }

        let index = Self::create_index(&index_name).context("Failed to create index")?;

        Ok(FTSIndex { index, index_name })
    }

    fn create_schema() -> Schema {
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::Basic);

        let text_options = TextOptions::default().set_indexing_options(text_indexing);

        let mut schema_builder = SchemaBuilder::new();

        schema_builder.add_u64_field("geonameId", STORED | INDEXED | FAST);
        schema_builder.add_text_field("name", text_options.clone().set_stored());
        schema_builder.add_text_field("asciiname", text_options.clone());
        schema_builder.add_text_field("alternatenames", text_options.clone());
        schema_builder.add_text_field("official_name", text_options.clone());
        schema_builder.add_text_field("ISO", text_options.clone());
        schema_builder.add_text_field("ISO3", text_options.clone());
        schema_builder.add_text_field("fips", text_options.clone());

        schema_builder.build()
    }

    fn create_index(index_name: &FTSIndexes) -> Result<Index> {
        let index_path = index_name.get_index_path();
        let schema = Self::create_schema();

        let index =
            Index::create_in_dir(index_path, schema.clone()).context("Failed to create index")?;

        let index_name = index_name.get_index_name();
        let df = LazyFrame::scan_parquet(
            format!("./data/processed/geonames/{}.parquet", index_name),
            Default::default(),
        )?
        .select([
            col("geonameId"),
            col("name"),
            col("asciiname"),
            col("alternatenames"),
            col("official_name"),
            col("ISO"),
            col("ISO3"),
            col("fips"),
        ])
        .collect()
        .context("Failed to collect DataFrame")?;

        let geoname_id = df.column("geonameId")?.cast(&DataType::UInt64)?;
        let geoname_id = geoname_id.u64()?;
        let name = df.column("name")?.str()?;
        let asciiname = df.column("asciiname")?.str()?;
        let alternatenames = df.column("alternatenames")?.str()?;
        let official_name = df.column("official_name")?.str()?;
        let iso = df.column("ISO")?.str()?;
        let iso3 = df.column("ISO3")?.str()?;
        let fips = df.column("fips")?.str()?;

        let gid_schema = schema.get_field("geonameId")?;
        let name_schema = schema.get_field("name")?;
        let asciiname_schema = schema.get_field("asciiname")?;
        let alternatives_schema = schema.get_field("alternatenames")?;
        let official_name_schema = schema.get_field("official_name")?;
        let iso_schema = schema.get_field("ISO")?;
        let iso3_schema = schema.get_field("ISO3")?;
        let fips_schema = schema.get_field("fips")?;

        let mut index_writer: IndexWriter = index.writer(500_000_000)?;

        info!("Creating index...");

        for (id, name, asciiname, alternative_name, official_name, iso, iso3, fips) in izip!(
            geoname_id,
            name,
            asciiname,
            alternatenames,
            official_name,
            iso,
            iso3,
            fips
        ) {
            if let (Some(id), Some(name)) = (id, name) {
                let mut doc = TantivyDocument::default();
                doc.add_u64(gid_schema, id);
                doc.add_text(name_schema, name);

                if let Some(asciiname) = asciiname {
                    doc.add_text(asciiname_schema, asciiname);
                }
                if let Some(altname) = alternative_name {
                    doc.add_text(alternatives_schema, altname);
                }
                if let Some(official_name) = official_name {
                    doc.add_text(official_name_schema, official_name);
                }
                if let Some(iso) = iso {
                    doc.add_text(iso_schema, iso);
                }
                if let Some(iso3) = iso3 {
                    doc.add_text(iso3_schema, iso3);
                }
                if let Some(fips) = fips {
                    doc.add_text(fips_schema, fips);
                }
                index_writer.add_document(doc)?;
            }
        }
        info!("Committing index...");
        index_writer.commit()?;
        Ok(index)
    }
}

// Search for a term in the index
impl FTSIndex {
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        fuzzy_search: bool,
    ) -> Result<Vec<(String, u64, f32)>> {
        self.search_inner(query, None, limit, fuzzy_search)
    }

    pub fn search_in_subset(
        &self,
        query_str: &str,
        doc_ids: &[u64],
        limit: usize,
        fuzzy_search: bool,
    ) -> Result<Vec<(String, u64, f32)>> {
        self.search_inner(query_str, Some(doc_ids), limit, fuzzy_search)
    }
    #[instrument(skip(self, doc_ids))]
    fn search_inner(
        &self,
        query_str: &str,
        doc_ids: Option<&[u64]>,
        limit: usize,
        fuzzy_search: bool,
    ) -> Result<Vec<(String, u64, f32)>> {
        let query_str = query_str.trim();
        debug_assert!(!query_str.is_empty(), "Query string is empty");
        debug_assert!(limit > 0, "Limit must be greater than zero, got: {}", limit);
        debug!("Searching for: {}", query_str);

        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        let schema = self.index.schema();
        let gid_field = schema.get_field("geonameId")?;
        let name_field = schema.get_field("name")?;
        let asciiname_field = schema.get_field("asciiname")?;
        let altname_field = schema.get_field("alternatenames")?;
        let official_name_field = schema.get_field("official_name")?;
        let iso_field = schema.get_field("ISO")?;
        let iso3_field = schema.get_field("ISO3")?;
        let fips_field = schema.get_field("fips")?;

        // Query parser for general text fields (name, asciiname, etc.)
        let mut general_query_parser = QueryParser::for_index(
            &self.index,
            vec![
                name_field,
                asciiname_field,
                altname_field,
                official_name_field,
            ],
        );
        //general_query_parser.set_conjunction_by_default(); // Terms in query are ANDed for these fields

        let (name_boost, asciiname_boost, altname_boost, official_name_boost) =
            match self.index_name {
                FTSIndexes::AdminSearch => (2.2, 1.8, 1.5, 2.2),
                FTSIndexes::PlacesSearch => (2.8, 1.8, 1.0, 1.5),
            };

        general_query_parser.set_field_boost(name_field, name_boost);
        general_query_parser.set_field_boost(asciiname_field, asciiname_boost);
        general_query_parser.set_field_boost(altname_field, altname_boost);
        general_query_parser.set_field_boost(official_name_field, official_name_boost);

        let general_fts_query = general_query_parser.parse_query(query_str)?;

        let mut query_clauses: Vec<(Occur, Box<dyn Query>)> = vec![];
        query_clauses.push((Occur::Should, general_fts_query));

        let is_single_short_token = !query_str.contains(char::is_whitespace)
            && !query_str.is_empty()
            && query_str.len() <= 3;

        if fuzzy_search {
            // Add Fuzzy Queries for general text fields
            // We might want to split the query_str into terms if it's multi-word
            // For simplicity, this example applies fuzzy to the whole query_str if it's a single token,
            // or you might iterate over tokens.
            let query_terms: Vec<&str> = query_str.split_whitespace().collect();

            // Only apply fuzzy matching if the query is not a short code candidate
            if !is_single_short_token {
                // Don't apply general fuzzy for things that might be codes
                for term_str in query_terms {
                    if term_str.len() > 2 {
                        // Apply fuzzy only to terms longer than 2 chars
                        let fuzzy_distance = 1; // Max 1 edit
                        let fuzzy_transpositions = true;

                        for (field, boost) in
                            [(name_field, 1.2_f32), (asciiname_field, 0.8)].into_iter()
                        {
                            let term = Term::from_field_text(field, term_str);
                            let fuzzy_query =
                                FuzzyTermQuery::new(term, fuzzy_distance, fuzzy_transpositions);
                            query_clauses.push((
                                Occur::Should,
                                Box::new(BoostQuery::new(Box::new(fuzzy_query), boost)),
                            ));
                        }
                    }
                }
            }
        }

        if is_single_short_token {
            let lower_case_code_query = query_str.to_lowercase(); // Explicitly lowercase "KEN" to "ken"

            for (field, boost) in [
                (iso3_field, 500.0_f32),
                (iso_field, 400.0),
                (fips_field, 200.0),
            ]
            .into_iter()
            {
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

        // Apply document ID filter if specified
        let final_query: Box<dyn Query> = match doc_ids {
            Some(ids) if !ids.is_empty() => {
                let doc_id_filter = TermSetQuery::new(
                    ids.iter()
                        .map(|&id| Term::from_field_u64(gid_field, id))
                        .collect::<Vec<_>>(),
                );
                Box::new(BooleanQuery::new(vec![
                    (Occur::Must, Box::new(combined_search_query)),
                    (Occur::Must, Box::new(doc_id_filter)),
                ]))
            }
            _ => Box::new(combined_search_query), // Handles None or empty doc_ids
        };

        let top_docs = searcher.search(&*final_query, &TopDocs::with_limit(limit))?;

        // Print documents for debugging
        debug!("Found {} results", top_docs.len());
        for (score, doc_address) in &top_docs {
            let doc = searcher.doc::<TantivyDocument>(*doc_address)?;
            trace!("Score: {}, Doc: {}", score, doc.to_json(&schema));
        }

        top_docs
            .into_iter()
            .map(|(score, doc_address)| {
                let received_doc = searcher
                    .doc::<TantivyDocument>(doc_address)
                    .context("Failed to retrieve document from searcher")?;
                let name = received_doc
                    .get_first(name_field)
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                // .and_then(Value::as_text) // Safely get text
                // .unwrap_or("") // Provide a default if None
                // .to_string();
                let doc_id = received_doc
                    .get_first(gid_field)
                    .and_then(|x| x.as_u64())
                    .context("Failed to get geonameId as u64 from document")?;

                Ok((name, doc_id, score))
            })
            .collect::<Result<Vec<_>>>()
    }
}
