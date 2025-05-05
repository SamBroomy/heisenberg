use anyhow::{Context, Result};
use std::{path::Path, vec};

use itertools::izip;
use polars::frame::DataFrame;
use polars::prelude::*;
//use polars::prelude::*;
use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    query::{BooleanQuery, Occur, Query, QueryParser, TermSetQuery},
    schema::*,
    Directory, Index, IndexWriter,
};

fn create_schema() -> tantivy::schema::Schema {
    // TODO: Dont store all fields, only store the
    let text_indexing = TextFieldIndexing::default()
        .set_tokenizer("default")
        .set_index_option(IndexRecordOption::Basic);

    let text_options = TextOptions::default().set_indexing_options(text_indexing);

    let mut schema_builder = SchemaBuilder::new();

    // Add fields with proper indexing settings
    schema_builder.add_u64_field("geonameId", STORED | INDEXED | FAST);

    // Use the custom analyzer for all text fields
    schema_builder.add_text_field("name", text_options.clone().set_stored());
    schema_builder.add_text_field("asciiname", text_options.clone());
    schema_builder.add_text_field("alternatenames", text_options.clone());
    schema_builder.add_text_field("official_name", text_options.clone());
    schema_builder.add_text_field("ISO", text_options.clone());
    schema_builder.add_text_field("ISO3", text_options.clone());

    schema_builder.build()
}

fn create_tantivy_index_from_df(df: &DataFrame, overwrite: bool) -> Result<Index> {
    let schema = create_schema();

    let path = Path::new("./data/indexes/tantivy");

    // First check if the path exists and create it if needed
    if !path.exists() {
        println!("Creating directory: {}", path.display());
        std::fs::create_dir_all(path).context("Failed to create directory")?;
    }

    // Now open the directory that we know exists
    let dir = tantivy::directory::MmapDirectory::open(path).context("Failed to open directory")?;

    if Index::exists(&dir)? {
        if overwrite {
            println!("Deleting existing index...");
            let path = Path::new("./data/indexes/tantivy");
            // Re-create the directory (safer than just deleting files)
            std::fs::remove_dir_all(path).context("Failed to remove existing index directory")?;
            std::fs::create_dir_all(path).context("Failed to create directory")?;
        } else {
            println!("Index already exists. Skipping creation.");
            return Index::open_in_dir(path).context("Failed to open existing index");
        }
    }

    let index = Index::create_in_dir(path, schema.clone()).context("Failed to create index")?;

    //let index = Index::create_in_ram(schema.clone());

    let geoname_id = df.column("geonameId")?.cast(&DataType::UInt64)?;
    let geoname_id = geoname_id.u64()?;
    let name = df.column("name")?.str()?;
    let asciiname = df.column("asciiname")?.str()?;
    let alternatenames = df.column("alternatenames")?.str()?;
    let official_name = df.column("official_name")?.str()?;
    let iso = df.column("ISO")?.str()?;
    let iso3 = df.column("ISO3")?.str()?;

    let gid_schema = schema.get_field("geonameId")?;
    let name_schema = schema.get_field("name")?;
    let asciiname_schema = schema.get_field("asciiname")?;
    let altname_schema = schema.get_field("alternatenames")?;
    let official_name_schema = schema.get_field("official_name")?;
    let iso_schema = schema.get_field("ISO")?;
    let iso3_schema = schema.get_field("ISO3")?;

    let mut index_writer: IndexWriter = index.writer(500_000_000)?;

    println!("Creating index...");

    for (id, name, asciiname, altname, official_name, iso, iso3) in izip!(
        geoname_id,
        name,
        asciiname,
        alternatenames,
        official_name,
        iso,
        iso3
    ) {
        if let (Some(id), Some(name)) = (id, name) {
            let mut doc = TantivyDocument::default();
            doc.add_u64(gid_schema, id);
            doc.add_text(name_schema, name);

            if let Some(asciiname) = asciiname {
                doc.add_text(asciiname_schema, asciiname);
            }
            if let Some(altname) = altname {
                doc.add_text(altname_schema, altname);
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
            index_writer.add_document(doc)?;
        }
    }
    println!("Committing index...");
    index_writer.commit()?;
    Ok(index)
}
fn search_in_subset(
    index: &Index,
    query: &str,
    doc_ids: Option<&[u64]>,
) -> Result<Vec<(String, u64, f32)>> {
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let schema = index.schema();
    let gid_field = schema.get_field("geonameId")?;
    let name_field = schema.get_field("name")?;
    let asciiname_field = schema.get_field("asciiname")?;
    let altname_field = schema.get_field("alternatenames")?;
    let official_name_field = schema.get_field("official_name")?;
    let iso_field = schema.get_field("ISO")?;
    let iso3_field = schema.get_field("ISO3")?;

    // Create a query parser that uses our custom analyzer
    let mut query_parser = QueryParser::for_index(
        index,
        vec![
            name_field,
            asciiname_field,
            altname_field,
            official_name_field,
            iso_field,
            iso3_field,
        ],
    );

    // Allow prefix queries (partial matching)
    query_parser.set_conjunction_by_default();
    query_parser.set_field_boost(name_field, 2.0); // Boost name matches

    let term_query = query_parser.parse_query(query)?;

    // Apply document ID filter if specified
    let final_query: Box<dyn Query> = match doc_ids {
        Some(ids) => {
            let doc_id_filter = TermSetQuery::new(
                ids.iter()
                    .map(|&id| Term::from_field_u64(gid_field, id))
                    .collect::<Vec<_>>(),
            );

            Box::new(BooleanQuery::new(vec![
                (Occur::Must, Box::new(term_query)),
                (Occur::Must, Box::new(doc_id_filter)),
            ]))
        }
        None => Box::new(term_query),
    };

    let top_docs = searcher.search(&*final_query, &TopDocs::with_limit(10))?;

    // Print documents for debugging
    println!("Found {} results", top_docs.len());
    for (score, doc_address) in &top_docs {
        let doc = searcher.doc::<TantivyDocument>(*doc_address)?;
        println!("Score: {}, Doc: {}", score, doc.to_json(&schema));
    }

    Ok(top_docs
        .into_iter()
        .map(|(score, doc_address)| {
            let received_doc = searcher.doc::<TantivyDocument>(doc_address).unwrap();
            let name = received_doc
                .get_first(name_field)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();
            let doc_id = received_doc.get_first(gid_field).unwrap().as_u64().unwrap();
            (name, doc_id, score)
        })
        .collect())
}

fn search_admin(
    term: &str,
    levels: &[u8],
    admin_data: &DataFrame,
    fts_index: &Index,
    previous_results: Option<&DataFrame>,
    limit: usize,
    all_cols: bool,
) -> Result<DataFrame> {
    todo!()
}

fn main() -> Result<()> {
    // print current working directory
    let current_dir = std::env::current_dir().unwrap();
    println!("{}", current_dir.display());
    let lf = LazyFrame::scan_parquet(
        "./data/processed/geonames/admin_search.parquet",
        Default::default(),
    )?
    .collect()?
    .lazy();

    let df = lf
        .clone()
        .select([
            col("geonameId"),
            col("name"),
            col("asciiname"),
            col("alternatenames"),
            col("official_name"),
            col("ISO"),
            col("ISO3"),
        ])
        .collect()
        .unwrap();
    println!("DataFrame: {:?}", df);

    println!("Creating tantivy index...");
    let index = create_tantivy_index_from_df(&df, false)?;

    println!("Index created.");

    let search_terms = ["Kenya", "Republic of Kenya", "Ceinia", "Chenia"];

    for term in search_terms {
        println!("\n--- Searching for: {} ---", term);
        let results = search_in_subset(&index, term, None)?;
        println!("Results: {:?}", results);
    }

    let query = "Kenya";
    let results = search_in_subset(&index, query, None)?;
    println!("Search results: {:?}", results);

    // Unzip results into separate vectors
    let (ids, scores): (Vec<_>, Vec<_>) =
        results.iter().map(|(_, id, score)| (*id, *score)).unzip();

    let lf = df!("geonameId" => ids,
                "fts_score" => scores,
    )?
    .lazy()
    .join(
        lf,
        [col("geonameId")],
        [col("geonameId")],
        JoinArgs::new(JoinType::Left),
    )
    .select([
        // Select all columns except fts_score
        col("*").exclude(["fts_score"]),
        // Then select fts_score to place it at the end
        col("fts_score"),
    ]);

    println!("{}", lf.to_dot(true)?);
    let df = lf.collect()?;

    println!("Joined DataFrame: {:?}", df);

    let doc_ids = vec![
        146669, 149590, 163843, 174982, 13297835, 13298761, 13298762, 13298880, 11237637, 192950,
    ];

    let results = search_in_subset(&index, query, Some(&doc_ids))?;

    println!("Search results with doc_ids: {:?}", results);

    let doc_ids = vec![
        146669, 149590, 163843, 174982, 13297835, 13298761, 13298762, 13298880, 11237637,
    ];

    let results = search_in_subset(&index, query, Some(&doc_ids))?;

    println!("Search results with doc_ids: {:?}", results);

    Ok(())
}
