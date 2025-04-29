#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
from loguru import logger
from typing import Literal
import polars.selectors as cs
import kuzu as kz
from pathlib import Path
from typing import Type, Callable
from usearch.index import Index, Matches
import geocoder
import numpy as np
from functools import partial
from numpy.typing import NDArray
import duckdb
from duckdb import DuckDBPyConnection
from time import time
import tantivy
from result import Result, Ok, Err
import json
import re
from sentence_transformers import SentenceTransformer


# In[2]:


duck_db_path = Path("./data/db/duck_db/data.db")
duck_db_path.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(database=duck_db_path.as_posix())

con.execute("SET enable_progress_bar = false;")
con.install_extension("spatial")
con.load_extension("spatial")

# Set DuckDB optimizations
con.execute("PRAGMA memory_limit='16GB'")  # Adjust based on your system
con.execute("PRAGMA threads=8")  # Adjust based on your CPU cores
con.execute("PRAGMA enable_object_cache=true")  # Improve query caching


# In[3]:


con.execute("SHOW TABLES").pl()


# In[4]:


SQL_FOLDER = Path("./sql")


def sql_file(sql_path: Path | str, **kwargs) -> str:
    if isinstance(sql_path, str):
        sql_path = Path(sql_path)
    if not sql_path.exists():
        sql_path = SQL_FOLDER / sql_path
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file {sql_path} not found")
    sql = sql_path.read_text()
    if kwargs:
        sql = sql.format(**kwargs)

    # Validate no {kwarg} left in string (regex)
    # if uninit_kwargs := re.findall(r"\{.*\}", sql):
    #     raise ValueError(
    #         f"SQL file {sql_path} still has unprocessed kwargs: {list(set(uninit_kwargs))} in:\n\n{sql}"
    #     )

    return sql


# In[5]:


GID = "geonameId"


def table_exists(con: DuckDBPyConnection, table_name: str) -> bool:
    return table_name in con.execute("SHOW TABLES").pl()["name"]


# Read and load 'allCountries.txt'
# Function to read and load other files with different schemas
def load_file(
    # con: DuckDBPyConnection,
    file_path: str,
    schema: dict[str, Type[pl.DataType]],
    table_name: str,
    table_definition: str | None = None,
    pipe: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    has_header: bool = False,
    skip_rows: int = 0,
    overwrite: bool = False,
    extra_expr: pl.Expr | None = None,
):
    if table_exists(con, table_name):
        logger.debug(f"Table '{table_name}' already exists")
        if not overwrite:
            return
        logger.debug(f"Overwriting table '{table_name}'")
        con.execute(f"DROP TABLE {table_name} CASCADE")
        logger.debug(f"Table '{table_name}' dropped")
    time_start = time()
    load = con.begin()
    try:
        logger.info(f"Loading '{file_path}'...")
        # Time scan
        time_scan = time()
        q = pl.scan_csv(
            file_path,
            separator="\t",
            has_header=has_header,
            schema=schema,
            skip_rows=skip_rows,
        )
        q = q.with_columns(
            pl.col(pl.Utf8).str.strip_chars().str.strip_chars("\"':").str.strip_chars()
        )
        if extra_expr is not None:
            q = q.with_columns(extra_expr)
        if pipe is not None:
            q = q.pipe(pipe)
        if GID in schema:
            q = q.sort(GID, nulls_last=True)
        logger.debug(f"Scan time: {time() - time_scan:.6f}s")

        q = q.with_columns(cs.by_dtype(pl.String).str.strip_chars().replace("", None))

        # Time collect
        time_collect = time()
        df = q.collect()
        logger.debug(f"Collect time: {time() - time_collect:.6f}s")

        # Time write
        time_write = time()
        save_path = Path(f"./data/processed/geonames/{table_name}.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(save_path.as_posix())
        logger.debug(f"Write time: {time() - time_write:.6f}s")

        # Time create
        time_create = time()
        # Create table with predefined schema if provided
        time_create = time()
        if table_definition:
            # Create the table with specified schema
            load.execute(table_definition)
            load.from_arrow(df.to_arrow()).insert_into(table_name)
        else:
            # Use automatic schema derivation (your current approach)
            load.from_arrow(df.to_arrow()).create(table_name)

        logger.debug(f"Create time: {time() - time_create:.6f}s")

        time_commit = time()
        load.commit()
        logger.debug(f"Commit time: {time() - time_commit:.6f}s")
        analyze_time = time()
        con.execute("VACUUM ANALYZE;")
        logger.debug(f"Analyze time: {time() - analyze_time:.6f}s")
    except Exception as e:
        logger.exception("Error loading '{file_path}'")
        logger.debug(e.with_traceback(None))
        # Time rollback
        time_rollback = time()
        load.rollback()
        logger.warning(f"Rollback time: {time() - time_rollback:.6f}s")
        raise e
    finally:
        logger.info(f"Total time: {time() - time_start:.6f}s")
    return df


def drop_duplicates(df: pl.LazyFrame) -> pl.LazyFrame:
    cols = [
        "name",
        "asciiname",
        "feature_class",
        "feature_code",
        "admin0_code",
        "admin1_code",
        "admin2_code",
        "admin3_code",
        "admin4_code",
        "timezone",
    ]
    return (
        df.sort("modification_date", descending=True)
        .unique(cols, keep="first")
        .filter(~pl.all_horizontal(pl.col(cols).is_null()))
        .sort("geonameId")
    )


schema_all_countries = {
    GID: pl.UInt32,
    "name": pl.Utf8,
    "asciiname": pl.Utf8,
    "alternatenames": pl.Utf8,
    "latitude": pl.Float32,
    "longitude": pl.Float32,
    "feature_class": pl.Categorical,
    "feature_code": pl.Categorical,
    "admin0_code": pl.Categorical,
    "cc2": pl.Utf8,
    "admin1_code": pl.Utf8,
    "admin2_code": pl.Utf8,
    "admin3_code": pl.Utf8,
    "admin4_code": pl.Utf8,
    "population": pl.Int64,
    "elevation": pl.Int32,
    "dem": pl.Int32,
    "timezone": pl.Categorical,
    "modification_date": pl.Date,
}


load_file(
    "./data/raw/geonames/allCountries.txt",
    schema_all_countries,
    "allCountries",
    table_definition=sql_file("create_table_allCountries.sql"),
    pipe=drop_duplicates,
)


load_file(
    "./data/raw/geonames/allCountriesPostCode.txt",
    {
        "admin_code0": pl.Categorical,
        "postal_code": pl.Utf8,
        "place_name": pl.Utf8,
        "admin_name1": pl.Utf8,
        "admin_code1": pl.Utf8,
        "admin_name2": pl.Utf8,
        "admin_code2": pl.Utf8,
        "admin_name3": pl.Utf8,
        "admin_code3": pl.Utf8,
        "latitude": pl.Float32,
        "longitude": pl.Float32,
        "accuracy": pl.Int32,
    },
    "allPostCodes",
)


# Load other files with respective schemas
load_file(
    "./data/raw/geonames/admin1CodesASCII.txt",
    {
        "code": pl.Utf8,
        "name": pl.Utf8,
        "name_ascii": pl.Utf8,
        GID: pl.UInt32,
    },
    "admin1CodesASCII",
    table_definition=sql_file("create_table_admin1CodesASCII.sql"),
)

load_file(
    "./data/raw/geonames/admin2Codes.txt",
    {
        "code": pl.Utf8,
        "name": pl.Utf8,
        "asciiname": pl.Utf8,
        GID: pl.UInt32,
    },
    "admin2Codes",
    table_definition=sql_file("create_table_admin2Codes.sql"),
)


def drop_invalid_gids(df: pl.LazyFrame, con: DuckDBPyConnection) -> pl.LazyFrame:
    ids = con.execute("SELECT geonameId FROM allCountries").pl().unique().to_series()
    return df.filter(pl.col(GID).is_in(ids))


load_file(
    "./data/raw/geonames/adminCode5.txt",
    {
        GID: pl.UInt32,
        "adm5code": pl.Utf8,
    },
    "adminCode5",
    table_definition=sql_file("create_table_adminCode5.sql"),
    pipe=partial(drop_invalid_gids, con=con),
)


load_file(
    "./data/raw/geonames/alternateNamesV2.txt",
    {
        "alternateNameId": pl.Int32,
        GID: pl.UInt32,
        "isolanguage": pl.Utf8,
        "alternate_name": pl.Utf8,
        "isPreferredName": pl.Int8,
        "isShortName": pl.Int8,
        "isColloquial": pl.Int8,
        "isHistoric": pl.Int8,
        "from": pl.Utf8,
        "to": pl.Utf8,
    },
    "alternateNamesV2",
    table_definition=sql_file("create_table_alternateNamesV2.sql"),
    extra_expr=cs.by_dtype(pl.Int8).cast(pl.Boolean).fill_null(False),
    pipe=partial(drop_invalid_gids, con=con),
)

load_file(
    "./data/raw/geonames/countryInfo.txt",
    {
        "ISO": pl.Categorical,
        "ISO3": pl.Categorical,
        "ISO_Numeric": pl.Int32,
        "fips": pl.Categorical,
        "Country": pl.Utf8,
        "Capital": pl.Utf8,
        "Area": pl.Float32,
        "Population": pl.Int32,
        "Continent": pl.Categorical,
        "tld": pl.Utf8,
        "CurrencyCode": pl.Utf8,
        "CurrencyName": pl.Utf8,
        "Phone": pl.Utf8,
        "Postal_Code_Format": pl.Utf8,
        "Postal_Code_Regex": pl.Utf8,
        "Languages": pl.Utf8,
        GID: pl.UInt32,
        "neighbours": pl.Utf8,
        "EquivalentFipsCode": pl.Utf8,
    },
    "countryInfo",
    table_definition=sql_file("create_table_countryInfo.sql"),
    skip_rows=51,
)

load_file(
    "./data/raw/geonames/featureCodes_en.txt",
    {
        "code": pl.Categorical,
        "name": pl.Utf8,
        "description": pl.Utf8,
    },
    "featureCodes",
    table_definition=sql_file("create_table_featureCodes.sql"),
)


def remove_old_ids(df: pl.LazyFrame, con: DuckDBPyConnection) -> pl.LazyFrame:
    ids = con.execute("SELECT geonameId FROM allCountries").pl().unique().to_series()
    return (
        df.filter(pl.col("parentId").is_in(ids) & pl.col("childId").is_in(ids))
        .with_columns(
            pl.when(pl.col("type").str.contains("adm", literal=True))
            .then(pl.col("type").str.to_uppercase())
            .otherwise(pl.col("type"))
        )
        .unique(["parentId", "childId"])
    )


load_file(
    "./data/raw/geonames/hierarchy.txt",
    {
        "parentId": pl.UInt32,
        "childId": pl.UInt32,
        "type": pl.Utf8,
    },
    "hierarchy",
    table_definition=sql_file("create_table_hierarchy.sql"),
    pipe=partial(remove_old_ids, con=con),
)
con.execute(sql_file("create_table_unique_ids.sql"))

load_file(
    "./data/raw/geonames/iso-languagecodes.txt",
    {
        "ISO_639_3": pl.Utf8,
        "ISO_639_2": pl.Utf8,
        "ISO_639_1": pl.Utf8,
        "Language_Name": pl.Utf8,
    },
    "iso_languagecodes",
    table_definition=sql_file("create_table_iso_languagecodes.sql"),
)


load_file(
    "./data/raw/geonames/timeZones.txt",
    {
        "CountryCode": pl.Utf8,
        "TimeZoneId": pl.Utf8,
        "GMT_offset_1_Jan_2024": pl.Float32,
        "DST_offset_1_Jul_2024": pl.Float32,
        "rawOffset": pl.Float32,
    },
    "timeZones",
    table_definition=sql_file("create_table_timeZones.sql"),
    skip_rows=1,
)
# # Ignore loading the geo data for now
if not table_exists(con, "shapes"):
    con.execute(sql_file("create_table_shapes.sql"))
    logger.debug("Table 'shapes' created")

# # File is corupted atm
# load_file(
#     "./data/raw/geonames/userTags.txt",
#     {
#         GID: pl.Int32,
#         "tag": pl.Utf8,
#     },
#     "userTags",
# )
con.execute(sql_file("create_table_equivalent.sql"))
con.execute(sql_file("create_view_cities.sql"))
con.execute(sql_file("create_view_locations_full.sql"))


# In[8]:


# Create country table
# con.execute(sql_file("create_table_equivalent.sql")).pl()
# con.execute(sql_file("create_table_admin0.sql")).execute("""PRAGMA create_fts_index(
#     admin0,
#     geonameId,
#     name,
#     asciiname,
#     official_name,
#     alternatenames,
#     admin0_code,
#     ISO3,
#     ISO_Numeric,
#     fips,
#     stemmer = 'none',
#     stopwords = 'none',
#     ignore = '(\\.|[^a-z0-9])+',
#     overwrite = 1
# );""")

# con.execute(sql_file("create_view_*_NODES.sql", table="admin0"))

# con.execute(sql_file("create_view_*_FTS.sql", table="admin0"))


# In[9]:


get_ipython().run_line_magic('skip', '')
# if (path := Path("./data/processed/latlon.index")).exists():
#     logger.debug("Loading index...")
#     index = Index.restore(path, view=True) or raise ValueError("Failed to load index")
# else:
#     logger.debug("Creating index...")
#     coordinates = df.select(["latitude", "longitude"]).to_numpy(order="c")
#     labels = df["geonameid"].to_numpy()
#     index: Index = Index(ndim=2, metric="haversine", dtype="f32")
#     index.add(keys=labels, vectors=coordinates, log=True)
#     index.save(path)


class VectorIndex:
    default_index_path = Path("./data/indexes/vector")

    def __init__(
        self,
        index_name: str,
        data: pl.DataFrame | None = None,
        id_column: str = "geonameId",
        main_column: str = "vectors",
        metric: str = "L2",
        embedder: SentenceTransformer | None = None,
    ):
        self._index_path = self.default_index_path / f"{index_name}.index"
        self._id_column = id_column
        self._main_column = main_column
        self._metric = metric
        index = self.get_or_build_index(data, metric)
        if isinstance(index, Err):
            logger.debug(
                f"Index does not exist at '{self.index_path}', build index with 'build_index' method."
            )
            self._index = None  # type: ignore
        else:
            self._index: Index = index.ok_value

    @property
    def index(self) -> Index:
        return self._index

    @property
    def id_column(self) -> str:
        return self._id_column

    @property
    def main_column(self) -> str:
        return self._main_column

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def metric(self) -> str:
        return self._metric

    def _build_index(
        self,
        df: pl.DataFrame,
        metric: str = "L2",  # TODO: Metric like
    ) -> Result[Index, str]:
        """Data passed should be an Id and a vector."""
        logger.debug("Creating index...")
        vectors = df[self.main_column].to_numpy()
        labels = df[self.id_column].to_numpy()
        ndims = vectors.shape[1]  # Find n dims
        index: Index = Index(ndim=ndims, metric=metric, dtype="f32")
        index.add(keys=labels, vectors=vectors, log=True)
        index.save(self.index_path)
        return Ok(index)

    def get_index(self) -> Result[Index, str]:
        if (path := self.index_path).exists():
            logger.debug(f"Opening index at '{self.index_path}'")
            index = Index.restore(path, view=True)
            if index is not None:
                return Ok(index)
        return Err(f"Index does not exist at '{self.index_path}'")

    def get_or_build_index(
        self,
        df: pl.DataFrame | None = None,
        metric: str = "L2",  # TODO: as above
    ) -> Result[Index, str]:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.index_path.exists():
            if df is None:
                return Err(
                    "Index does not exist. DataFrame is required to create index"
                )
            match self._build_index(df, metric):
                case Ok(index):
                    ...
                case Err(e):
                    return Err(e)
        else:
            match self.get_index():
                case Ok(index):
                    ...
                case Err(e):
                    return Err(e)

        self._ndims = index.ndim
        logger.debug("Opening index")
        return Ok(index)

    def search(
        self,
        query: NDArray[np.float32],
        limit: int = 10,
        include: list[int] | None = None,
        exclude: list[int] | None = None,
    ) -> Result[pl.DataFrame, str]:
        return self.vector_search(query, limit, include, exclude)

    def vector_search(
        self,
        query: NDArray[np.float32],
        limit: int = 10,
        include: list[int] | None = None,
        exclude: list[int] | None = None,
        exact: bool = False,
    ) -> Result[pl.DataFrame, str]:
        output = self.index.search(vectors=query, count=limit, log=True, exact=exact)

        logger.debug(f"Visited members: {output.visited_members}")
        logger.debug(f"Computed distances: {output.computed_distances}")

        # Extract keys (geonameids) and distances
        keys = output.keys
        distances = output.distances

        # Create a DataFrame from the search results
        results_df = pl.LazyFrame(
            data={self.id_column: keys, "score": distances},
            schema={self.id_column: pl.UInt32, "score": pl.Float32},
        )
        if self.metric == "haversine":
            results_df = results_df.with_columns(pl.col("score") * 6371.0)

        results_df = results_df.sort(
            "score"
        )  # TODO: ascending descending depending on metric.

        return Ok(results_df.collect())


class FTSIndex:
    default_index_path = Path("./data/indexes/fts")

    def __init__(
        self,
        index_name: str,
        data: pl.DataFrame | None = None,
        id_column: str = "geonameId",
        main_column: str = "name",
    ):
        self._index_path = self.default_index_path / index_name
        self._column_types = {}
        self._id_column = id_column
        self._main_column = main_column
        index = self.get_or_build_index(data)
        if isinstance(index, Err):
            logger.debug(
                f"Index does not exist at '{self.index_path}', build index with 'build_index' method."
            )
            self._index = None  # type: ignore
        else:
            self._index: tantivy.Index = index.ok_value

    @property
    def index(self) -> tantivy.Index:
        self._index.reload()
        return self._index

    @property
    def column_types(self) -> dict[str, str]:
        return self._column_types

    @property
    def id_column(self) -> str:
        return self._id_column

    @property
    def main_column(self) -> str:
        return self._main_column

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def columns_not_id(self) -> list[str]:
        return [col for col in self.column_types if col != self.id_column]

    def _build_index(
        self,
        df: pl.DataFrame,
        split_field: dict[str, list[str] | str] | None = None,
    ) -> Result[tantivy.Index, str]:
        """Only pass in data which you wish to build the ftx index with. split_field is a dictionary of fields to split by a delimiter. eg {",": ["field1", "field2"]} will split field1 and field2 by comma."""
        # TODO: this programmatically into tantivy schema
        schema_builder = tantivy.SchemaBuilder()

        if self.id_column not in df.columns:
            return Err(f"'{self.id_column}' column not found in DataFrame")

        col_types = {}
        for col in df.columns:
            if col == self.id_column:
                schema_builder.add_integer_field(
                    self.id_column, stored=True, indexed=True, fast=True
                )
            # TODO: ADD support for other types
            else:
                schema_builder.add_text_field(col)
            col_types[col] = df[col].dtype._string_repr()

        self._column_types = col_types

        schema = schema_builder.build()
        logger.debug(f"Creating index with columns:\n{json.dumps(col_types, indent=2)}")

        index = tantivy.Index(schema, path=self.index_path.as_posix(), reuse=False)
        writer = index.writer()
        for row in df.rows(named=True):
            if split_field:
                for splitter, fields in split_field.items():
                    if isinstance(fields, str):
                        fields = [fields]
                    for field in fields:
                        logger.debug(f"Splitting {field} by {splitter}...")
                        row[field] = row[field].split(splitter)
            writer.add_document(tantivy.Document(**row))
        writer.commit()
        writer.wait_merging_threads()
        return Ok(index)

    def get_index(self) -> Result[tantivy.Index, str]:
        if tantivy.Index.exists(self.index_path.as_posix()):
            logger.debug(f"Opening index at '{self.index_path}'")
            return Ok(tantivy.Index.open(self.index_path.as_posix()))
        return Err(f"Index does not exist at '{self.index_path}'")

    def get_or_build_index(
        self, df: pl.DataFrame | None = None
    ) -> Result[tantivy.Index, str]:
        if not self.index_path.exists() and df is None:
            return Err("Index does not exist. DataFrame is required to create index")

        self.index_path.mkdir(parents=True, exist_ok=True)

        if not tantivy.Index.exists(self.index_path.as_posix()):
            if df is None:
                return Err("DataFrame is required to create index")
            match self._build_index(df):
                case Ok(index):
                    ...
                case Err(e):
                    return Err(e)
        else:
            match self.get_index():
                case Ok(index):
                    ...
                case Err(e):
                    return Err(e)
        schema = json.loads((self.index_path / "meta.json").read_text())["schema"]
        sc = {}
        for v in schema:
            type_ = v["type"]
            if type_ == "text":
                type_ = pl.Utf8
            elif type_ == "i64":
                type_ = pl.UInt32
            sc[v["name"]] = type_

        self._column_types = sc
        logger.debug("Schema Loaded")
        logger.debug("Opening country index")
        return Ok(index)

    def convert_fts_results(
        self, hits: tantivy.SearchResult, searcher: tantivy.Searcher
    ) -> pl.DataFrame:
        logger.debug(f"FTS hits from search: {hits.count}")  # type: ignore

        scores, gids = zip(
            *[
                (score, searcher.doc(doc).get_first(self.id_column))
                for score, doc in hits.hits
            ]
        )

        return (
            pl.LazyFrame(
                {"geonameId": list(gids), "score": list(scores)},
                schema={"geonameId": pl.UInt32, "score": pl.Float32},
            )
            .sort("score", descending=True, maintain_order=True)
            .collect()
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        include: list[int] | None = None,
        exclude: list[int] | None = None,
    ) -> Result[pl.DataFrame, str]:
        return self.fts_search(
            query,
            limit=limit,
            include=include,
            exclude=exclude,
        )

    def fts_search(
        self,
        query: str,
        limit: int = 10,
        include: list[int] | None = None,
        exclude: list[int] | None = None,
        main_term_query_boost: float = 3.0,
        fuzzy_term_query_boost: float = 2.0,
        max_fuzzy_distance: int = 2,
        phrase: bool = True,
    ) -> Result[pl.DataFrame, str]:
        # Create for list of queries (batch search)
        if phrase:
            query = f"'{query}'"
        else:
            query = query.strip("\"'")
        query = query.strip()
        index = self.index

        searcher = index.searcher()

        bool_query_list: list[tuple[tantivy.Occur, tantivy.Query]] = []

        # Calculate fuzzy distance based on query length
        fuzzy_distance = min(max(0, len(query) - 2), max_fuzzy_distance)

        if self.main_column in self.columns_not_id:
            main_term_query = tantivy.Query.term_query(
                index.schema, self.main_column, query
            )
            bool_query_list.append(
                (
                    tantivy.Occur.Should,
                    tantivy.Query.boost_query(main_term_query, main_term_query_boost),
                )
            )

            if fuzzy_distance > 0:
                main_fuzzy_query = tantivy.Query.fuzzy_term_query(
                    index.schema, self.main_column, query, distance=fuzzy_distance
                )

                bool_query_list.append(
                    (
                        tantivy.Occur.Should,
                        tantivy.Query.boost_query(
                            main_fuzzy_query, fuzzy_term_query_boost
                        ),
                    )
                )

            rest_of_query = index.parse_query(
                query, list(set(self.columns_not_id) - {self.main_column})
            )
            bool_query_list.append((tantivy.Occur.Should, rest_of_query))

        if include:
            bool_query_list.append(
                (
                    tantivy.Occur.Must,
                    tantivy.Query.term_set_query(index.schema, self.id_column, include),
                )
            )
        if exclude:
            bool_query_list.append(
                (
                    tantivy.Occur.MustNot,
                    tantivy.Query.term_set_query(index.schema, self.id_column, exclude),
                )
            )
        if bool_query_list:
            final_query = tantivy.Query.boolean_query(bool_query_list)

        else:
            final_query: tantivy.Query = index.parse_query(
                query, default_field_names=self.columns_not_id
            )

        logger.debug(final_query)

        hits: tantivy.SearchResult = searcher.search(final_query, limit=limit)

        if hits.count == 0:  # type: ignore
            if phrase:
                logger.debug("No results found, retrying without phrase search...")
                return self.fts_search(
                    query,
                    limit,
                    include,
                    exclude,
                    main_term_query_boost,
                    fuzzy_term_query_boost,
                    max_fuzzy_distance,
                    phrase=False,
                )
            return Err("No results found")

        return Ok(self.convert_fts_results(hits, searcher))


class HybridIndex:
    def __init__(self, fts_idx: FTSIndex, vidx: VectorIndex):
        self._fts_idx = fts_idx
        self._vidx = vidx

    @property
    def vector_index(self) -> VectorIndex:
        return self._vidx

    @property
    def fts_index(self) -> FTSIndex:
        return self._fts_idx

    def search(
        self,
        query: str,
        limit: int = 10,
        include: list[int] | None = None,
        exclude: list[int] | None = None,
        main_term_query_boost: float = 3.0,
        fuzzy_term_query_boost: float = 2.0,
        max_fuzzy_distance: int = 2,
        phrase: bool = True,
    ) -> Result[pl.DataFrame, str]:
        v_search = self.vector_index.vector_search


country_index = FTSIndex("admin0", con.table("admin0_FTS").pl())
country_index.fts_search("An Danmhairg").unwrap().join(
    con.table("admin0").pl(), "geonameId", "left"
)


# In[10]:


entities_df = con.execute(f"""
    SELECT {GID}, name, feature_class, feature_code
    FROM unique_ids
""").pl()

hierarchy_df = con.execute("""
    SELECT parentId, childId, type
    FROM hierarchy
""").pl()
logger.debug(
    f"Loaded {len(entities_df)} entities and {len(hierarchy_df)} hierarchical relationships"
)
logger.debug("Entity columns:", entities_df.columns)
logger.debug("Hierarchy columns:", hierarchy_df.columns)

# 2. Setup Kuzu database connection
gdb_path = Path("./data/db/graph_db")
gdb_path.mkdir(parents=True, exist_ok=True)
gdb = kz.Database(gdb_path.as_posix())
conn = kz.Connection(gdb)

# 3. Create the schema in Kuzu if needed
if "Entity" not in conn.execute("CALL SHOW_TABLES() RETURN *;").get_as_pl().get_column(
    "name"
):
    conn.execute(sql_file("create_node_entity.sql"))
    logger.debug("Created Entity table")

if "IsIn" not in conn.execute("CALL SHOW_TABLES() RETURN *;").get_as_pl().get_column(
    "name"
):
    conn.execute(sql_file("create_relation_IsIn.sql"))
    logger.debug("Created IsIn table")

    # 4. Check if tables already have data
are_nodes = (
    conn.execute("MATCH (e:Entity) RETURN count(e) > 0 AS HasData")
    .get_as_pl()
    .get_column("HasData")[0]
)
are_edges = (
    conn.execute("MATCH ()-[r:IsIn]->() RETURN count(r) > 0 AS HasData")
    .get_as_pl()
    .get_column("HasData")[0]
)

if not are_nodes:
    conn.execute(
        f"COPY Entity FROM (LOAD FROM entities_df RETURN {GID}, name, feature_class, feature_code)"
    )
    logger.debug("Loaded Entity")

if not are_edges:
    conn.execute(
        "COPY IsIn FROM (LOAD FROM hierarchy_df RETURN parentId, childId, type)"
    )
    logger.debug("Loaded IsIn")


# In[11]:


def get_children_query(geoname_id: int) -> str:
    query = f"""MATCH (p:Entity {{geonameId: {geoname_id}}})-[:IsIn]->(c:Entity)
    RETURN c.{GID} AS {GID}, c.name AS name, c.feature_class AS feature_class, c.feature_code AS feature_code;"""
    return query


def get_parents_query(geoname_id):
    query = f"""MATCH (c:Entity {{geonameId: {geoname_id}}})<-[:IsIn]-(p:Entity)
    RETURN p.{GID} AS {GID}, p.name AS name, p.feature_class AS feature_class, p.feature_code AS feature_code;"""
    return query


# MATCH (c:Entity) WHERE CAST(c.geonameId, "INT64") IN list_creation({formatted_ids}) RETURN *;
def get_children_querys(
    geoname_ids: list[int] | pl.Series, traverse: bool = False
) -> str:
    query = f"""MATCH (p:Entity)-[:IsIn{"*" if traverse else ""}]->(c:Entity)
    WHERE p.geonameId IN CAST({geoname_ids}, "UINT32[]")
    RETURN DISTINCT c.{GID} AS {GID}, c.name AS name, c.feature_class AS feature_class, c.feature_code AS feature_code;"""
    return query


def get_parents_querys(
    geoname_ids: list[int] | pl.Series, traverse: bool = False
) -> str:
    query = f"""MATCH (c:Entity)<-[:IsIn{"*" if traverse else ""}]-(p:Entity)
    WHERE c.geonameId IN CAST({geoname_ids}, "UINT32[]")
    RETURN DISTINCT p.{GID} AS {GID}, p.name AS name, p.feature_class AS feature_class, p.feature_code AS feature_code;"""
    return query


def get_highest_parent_query():
    query = f"""
    MATCH (entity:Entity)
    WHERE NOT (entity)<-[:IsIn]-(:Entity)
    RETURN entity.{GID} AS {GID}, entity.name AS name, entity.feature_class AS feature_class, entity.feature_code AS feature_code;"""
    return query


conn.execute(get_parents_query(49518)).get_as_pl()
conn.execute(get_children_query(6252001)).get_as_pl()
conn.execute(get_children_querys([49518, 51537])).get_as_pl()
conn.execute(get_parents_querys([49518, 51537])).get_as_pl()


# In[12]:


def handle_orphaned_admin_entities(con, conn):
    """Fix orphaned admin entities using both admin codes and graph traversal."""

    # Find entities with admin level feature codes that aren't in any admin table
    orphans_query = """
    SELECT a.geonameId, a.name, a.feature_code, a.admin0_code, a.admin1_code, a.admin2_code, a.admin3_code
    FROM allCountries a
    WHERE a.feature_code LIKE 'ADM%'
    AND a.geonameId NOT IN (
        SELECT geonameId FROM admin0
        UNION ALL SELECT geonameId FROM admin1
        UNION ALL SELECT geonameId FROM admin2
        UNION ALL SELECT geonameId FROM admin3
        UNION ALL SELECT geonameId FROM admin4
    )
    """

    orphans_df = con.execute(orphans_query).pl()
    if len(orphans_df) == 0:
        logger.debug("No orphaned admin entities found!")
        return

    logger.debug(f"Found {len(orphans_df)} orphaned admin entities.")

    # Try to find parents using graph database first
    orphan_ids = orphans_df["geonameId"].to_list()
    parents_query = f"""
    MATCH (c:Entity)<-[:IsIn]-(p:Entity)
    WHERE c.geonameId IN CAST({orphan_ids}, "UINT32[]")
    RETURN c.geonameId AS geonameId, p.geonameId AS parent_id
    """

    parent_links = conn.execute(parents_query).get_as_pl()

    # For remaining orphans without graph parents, try to infer level and parent using admin codes
    code_linked = 0
    for orphan in orphans_df.filter(
        ~pl.col("geonameId").is_in(parent_links["geonameId"])
    ).iter_rows(named=True):
        level = None
        if "ADM1" in orphan["feature_code"]:
            level = 1
        elif "ADM2" in orphan["feature_code"]:
            level = 2
        elif "ADM3" in orphan["feature_code"]:
            level = 3
        elif "ADM4" in orphan["feature_code"]:
            level = 4

        if level:
            # Based on level, try to add to appropriate admin table using code-based parent
            con.execute(f"""
            INSERT INTO admin{level} (
                SELECT
                    a.geonameId,
                    a.name,
                    a.asciiname,
                    a.admin0_code,
                    {"a.admin1_code" if level >= 1 else "NULL AS admin1_code"},
                    {"a.admin2_code" if level >= 2 else "NULL AS admin2_code"},
                    {"a.admin3_code" if level >= 3 else "NULL AS admin3_code"},
                    {"a.admin4_code" if level >= 4 else "NULL AS admin4_code"},
                    a.feature_class,
                    a.feature_code,
                    a.population,
                    -- Try to find parent ID from previous level
                    (
                        SELECT parent.geonameId FROM admin{level - 1} parent
                        WHERE parent.admin0_code = a.admin0_code
                        {"AND parent.admin1_code = a.admin1_code" if level >= 2 else ""}
                        {"AND parent.admin2_code = a.admin2_code" if level >= 3 else ""}
                        {"AND parent.admin3_code = a.admin3_code" if level >= 4 else ""}
                        LIMIT 1
                    ) AS parent_id,
                    c.name AS country_name,
                    parent.name AS parent_name,
                    a.alternatenames
                FROM
                    allCountries a
                LEFT JOIN
                    admin0 c ON a.admin0_code = c.admin0_code
                LEFT JOIN
                    admin{level - 1} parent ON
                        parent.admin0_code = a.admin0_code
                        {"AND parent.admin1_code = a.admin1_code" if level >= 2 else ""}
                        {"AND parent.admin2_code = a.admin2_code" if level >= 3 else ""}
                        {"AND parent.admin3_code = a.admin3_code" if level >= 4 else ""}
                WHERE
                    a.geonameId = {orphan["geonameId"]}
            )
            """)
            code_linked += 1

    logger.debug(
        f"Fixed {len(parent_links)} orphans using graph relationships and {code_linked} using admin codes"
    )


def create_admin_search_view(con):
    """Create a unified view for searching across all admin levels."""

    logger.debug("Creating unified admin search view...")
    con.execute("""
    CREATE OR REPLACE VIEW admin_search AS

    -- Countries (admin0)
    SELECT
        geonameId,
        name,
        asciiname,
        0 AS admin_level,
        admin0_code,
        NULL AS admin1_code,
        NULL AS admin2_code,
        NULL AS admin3_code,
        NULL AS admin4_code,
        NULL AS parent_name,
        NULL AS parent_id,
        feature_class,
        feature_code,
        Population AS population,
        alternatenames,
        'admin0' AS source_table
    FROM
        admin0

    UNION ALL

    -- Admin1 entities
    SELECT
        geonameId,
        name,
        asciiname,
        1 AS admin_level,
        admin0_code,
        admin1_code,
        NULL AS admin2_code,
        NULL AS admin3_code,
        NULL AS admin4_code,
        parent_name,
        parent_id,
        feature_class,
        feature_code,
        population,
        alternatenames,
        'admin1' AS source_table
    FROM
        admin1

    UNION ALL

    -- Admin2 entities
    SELECT
        geonameId,
        name,
        asciiname,
        2 AS admin_level,
        admin0_code,
        admin1_code,
        admin2_code,
        NULL AS admin3_code,
        NULL AS admin4_code,
        parent_name,
        parent_id,
        feature_class,
        feature_code,
        population,
        alternatenames,
        'admin2' AS source_table
    FROM
        admin2

    UNION ALL

    -- Admin3 entities
    SELECT
        geonameId,
        name,
        asciiname,
        3 AS admin_level,
        admin0_code,
        admin1_code,
        admin2_code,
        admin3_code,
        NULL AS admin4_code,
        parent_name,
        parent_id,
        feature_class,
        feature_code,
        population,
        alternatenames,
        'admin3' AS source_table
    FROM
        admin3

    UNION ALL

    -- Admin4 entities
    SELECT
        geonameId,
        name,
        asciiname,
        4 AS admin_level,
        admin0_code,
        admin1_code,
        admin2_code,
        admin3_code,
        admin4_code,
        parent_name,
        parent_id,
        feature_class,
        feature_code,
        population,
        alternatenames,
        'admin4' AS source_table
    FROM
        admin4
    """)

    logger.debug("Admin search view created!")


# In[13]:


def build_admin_tables_hybrid(con, conn, overwrite=True):
    """Build admin tables using graph database for relationships and DuckDB for storage."""

    logger.debug("Starting hybrid admin table construction...")

    # Define admin level feature code patterns
    level_codes = {
        0: ["PCL", "PCLI", "PCLD", "PCLF", "PCLS", "TERR"],
        1: ["ADM1", "ADM1H"],
        2: ["ADM2", "ADM2H"],
        3: ["ADM3", "ADM3H"],
        4: ["ADM4", "ADM4H"],
    }

    # Track entities at each admin level
    admin_entities = {}

    # Build admin tables one by one
    for level in range(0, 5):
        table_name = f"admin{level}"

        if table_exists(con, table_name) and not overwrite:
            logger.debug(f"Table {table_name} already exists. Skipping.")
            continue

        logger.info(f"Building {table_name} table...")

        # Step 1: Identify entities of this admin level by feature code
        feature_patterns = "', '".join([code for code in level_codes[level]])
        base_entities_query = f"""
            SELECT geonameId
            FROM allCountries
            WHERE feature_code IN ('{feature_patterns}')
            OR feature_code LIKE '{level_codes[level][0]}%'
        """

        level_entities = set(con.execute(base_entities_query).pl()["geonameId"])
        logger.debug(f"Found {len(level_entities)} initial entities for level {level}")

        # Step 2: For levels 1-4, use graph DB to validate hierarchical placement
        # This ensures entities are correctly placed in the admin hierarchy
        if level > 0 and level - 1 in admin_entities:
            # Find all direct children of the previous level's entities
            previous_level_ids = list(admin_entities[level - 1])

            # Use graph DB to get valid children
            children_query = f"""
            MATCH (p:Entity)-[:IsIn]->(c:Entity)
            WHERE p.geonameId IN CAST({previous_level_ids}, "UINT32[]")
            RETURN DISTINCT c.geonameId AS geonameId
            """

            valid_children = set(conn.execute(children_query).get_as_pl()["geonameId"])
            logger.debug(
                f"Found {len(valid_children)} valid children from graph relationships"
            )

            # Combine feature-based and relationship-based entities
            level_entities = level_entities.union(valid_children)

        admin_entities[level] = level_entities

        # Step 3: Create the admin table with appropriate fields
        if level == 0:  # Special case for countries
            create_query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT
                a.geonameId,
                a.name,
                a.asciiname,
                a.admin0_code,
                a.feature_class,
                a.feature_code,
                c.ISO,
                c.ISO3,
                c.ISO_Numeric,
                c.Country AS official_name,
                c.fips,
                c.Population,
                c.Area,
                a.alternatenames,
                NULL AS parent_id  -- No parent for countries
            FROM
                allCountries a
            LEFT JOIN
                countryInfo c ON a.geonameId = c.geonameId
            WHERE
                a.geonameId IN ({",".join(map(str, level_entities))})
            ORDER BY
                a.geonameId
            """
        else:
            # For admin levels 1-4, include parent relationship
            create_query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            WITH parent_links AS (
                SELECT
                    childId AS geonameId,
                    parentId AS parent_id
                FROM
                    hierarchy
                WHERE
                    type = 'ADM' AND
                    childId IN ({",".join(map(str, level_entities))})
            )
            SELECT
                a.geonameId,
                a.name,
                a.asciiname,
                a.admin0_code
                {", a.admin1_code" if level >= 1 else ", NULL AS admin1_code"}
                {", a.admin2_code" if level >= 2 else ", NULL AS admin2_code"}
                {", a.admin3_code" if level >= 3 else ", NULL AS admin3_code"}
                {", a.admin4_code" if level >= 4 else ", NULL AS admin4_code"},
                a.feature_class,
                a.feature_code,
                a.population,
                p.parent_id,
                c.name AS country_name,
                CASE
                    WHEN p.parent_id IS NOT NULL THEN parent.name
                    ELSE NULL
                END AS parent_name,
                a.alternatenames
            FROM
                allCountries a
            LEFT JOIN
                parent_links p ON a.geonameId = p.geonameId
            LEFT JOIN
                allCountries parent ON p.parent_id = parent.geonameId
            LEFT JOIN
                admin0 c ON a.admin0_code = c.admin0_code
            WHERE
                a.geonameId IN ({",".join(map(str, level_entities))})
            ORDER BY
                a.geonameId
            """

        # Execute the query to create the table
        con.execute(create_query)

        # Step 4: Add indexes and FTS
        con.execute(f"CREATE INDEX idx_{table_name}_gid ON {table_name} (geonameId)")

        if level > 0:
            # Create indexes for parent relationships
            con.execute(
                f"CREATE INDEX idx_{table_name}_parent ON {table_name} (parent_id)"
            )
            con.execute(
                f"CREATE INDEX idx_{table_name}_admin0 ON {table_name} (admin0_code)"
            )

            # Create admin code index specific to this level
            if level <= 4:
                con.execute(
                    f"CREATE INDEX idx_{table_name}_code ON {table_name} (admin{level}_code)"
                )

        # Create FTS index for searching
        fts_fields = "geonameId, name, asciiname, alternatenames"
        if level == 0:
            fts_fields += ", official_name, ISO, ISO3"

        con.execute(f"""
        PRAGMA create_fts_index(
            {table_name},
            {fts_fields},
            stemmer = 'none',
            stopwords = 'none',
            ignore = '(\\.|[^a-z0-9])+',
            overwrite = 1
        )
        """)

        # Report count
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.debug(f"Created {table_name} with {count} entities")

    # Step 5: Handle orphaned entities
    logger.debug("\nChecking for orphaned administrative entities...")
    handle_orphaned_admin_entities(con, conn)

    # Step 6: Create a unified admin search view
    create_admin_search_view(con)
    # Admin1 composite index
    con.execute(
        "CREATE INDEX IF NOT EXISTS admin1_codes ON admin1 (admin0_code, admin1_code)"
    )

    # Admin2 composite index
    con.execute(
        "CREATE INDEX IF NOT EXISTS admin2_codes ON admin2 (admin0_code, admin1_code, admin2_code)"
    )

    # Admin3 composite index
    con.execute(
        "CREATE INDEX IF NOT EXISTS admin3_codes ON admin3 (admin0_code, admin1_code, admin2_code, admin3_code)"
    )

    # Admin4 composite index
    con.execute(
        "CREATE INDEX IF NOT EXISTS admin4_codes ON admin4 (admin0_code, admin1_code, admin2_code, admin3_code, admin4_code)"
    )

    logger.debug("Created composite indexes on admin code columns")

    logger.debug("Hybrid admin table construction complete!")


# In[14]:


build_admin_tables_hybrid(con, conn, overwrite=True)


# In[17]:


con.table("admin1").pl()


# In[ ]:


def country_score(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.with_columns(
            pop_multiplier=(1 + (pl.col("Population").add(1).log10() / 10)),
            area_multiplier=(1 + (pl.col("Area").add(1).log10() / 20)),
            feature_multiplier=pl.when(pl.col("feature_code") == "PCLI")
            .then(0.5)
            .when(pl.col("feature_code").str.contains("^PCL.*$"))
            .then(0.2)
            .otherwise(0),
        )
        .with_columns(
            adjusted_score_0=pl.col("score")
            * (
                pl.col("pop_multiplier")
                + pl.col("area_multiplier")
                + pl.col("feature_multiplier")
            )
        )
        .sort("adjusted_score_0", descending=True)
        .select(
            "geonameId",
            "name",
            cs.starts_with("admin"),
            cs.starts_with("adjusted_score"),
        )
    )


def admin_score(
    df: pl.LazyFrame, level: int, parent_weight: float = 0.3
) -> pl.LazyFrame:
    assert level in [1, 2, 3, 4], "Level must be between 1 and 4"
    score_column = f"adjusted_score_{level}"
    parent_score_column = f"adjusted_score_{level - 1}"

    df = df.with_columns(
        pop_multiplier=(1 + (pl.col("population").add(1).log10() / 10)),
    )
    # Check all possible parent score columns, from highest to lowest
    # This allows using any available parent score, not just the immediate parent
    available_parent_scores = []
    for parent_level in range(level - 1, -1, -1):
        parent_score_column = f"adjusted_score_{parent_level}"
        if parent_score_column in df.collect_schema().keys():
            logger.debug(f"Found parent score column: {parent_score_column}")
            available_parent_scores.append((parent_level, parent_score_column))
            break  # Use the highest available parent level
    if available_parent_scores:
        # Use the highest available parent score
        parent_level, parent_score_column = available_parent_scores[0]
        logger.debug(
            f"Using parent score from level {parent_level}: {parent_score_column}"
        )

        # Apply scoring with parent influence
        df = df.with_columns(
            (
                (pl.col("score") * pl.col("pop_multiplier")).pow(1 - parent_weight)
                * pl.col(parent_score_column).pow(parent_weight)
            ).alias(score_column),
        )
    else:
        # No parent scores available, use only the current score
        logger.debug("No parent score columns found. Using only current level score.")
        df = df.with_columns(
            (pl.col("score") * pl.col("pop_multiplier")).alias(score_column),
        )
    # Return sorted and selected columns
    return df.sort(score_column, descending=True).select(
        "geonameId",
        "name",
        cs.starts_with("admin"),
        cs.starts_with("adjusted_score"),
    )



def search_country(term: str, con: DuckDBPyConnection, limit: int = 20) -> pl.DataFrame:
    """Search for a country by name, ISO code, etc."""
    query = """SELECT *,
    -- High fixed score for exact matches
    CASE
        WHEN LOWER(ISO) = LOWER($term) THEN 10.0
        WHEN LOWER(ISO3) = LOWER($term) THEN 8.0
        WHEN LOWER(fips) = LOWER($term) THEN 4.0
    END AS score
    FROM admin0
    WHERE
        -- First exact match priority
        LOWER(ISO) = LOWER($term) OR
        LOWER(ISO3) = LOWER($term) OR
        LOWER(fips) = LOWER($term)
    UNION ALL
    -- Then fall back to fuzzy search for anything not exact
    SELECT * FROM (
        SELECT *, fts_main_admin0.match_bm25(geonameId, $term) AS score
        FROM admin0
        WHERE
            LOWER(ISO) != LOWER($term) AND
            LOWER(ISO3) != LOWER($term) AND
            LOWER(fips) != LOWER($term)
    ) sq
    WHERE score IS NOT NULL
    ORDER BY score DESC
    LIMIT $limit;
    """

    results = con.execute(query, {"term": term, "limit": limit}).pl()

    if not results.is_empty():
        # Apply country-specific scoring
        return results.lazy().pipe(country_score).collect()

    return results


def build_path_conditions(df: pl.DataFrame, admin_cols: list[str]) -> str:
    """
    Build SQL path conditions from unique admin code combinations.
    """
    if not admin_cols:
        return ""

    # Extract only the relevant columns and remove rows with nulls
    paths_df = df.select(admin_cols).drop_nulls()

    if paths_df.is_empty():
        return ""

    # Get unique path combinations
    unique_paths = paths_df.unique()

    # Build OR conditions for each path
    path_conditions = []
    for row in unique_paths.iter_rows(named=True):
        conditions = []
        for col, val in row.items():
            if val is None:
                conditions.append(f"{col} IS NULL")
            elif isinstance(val, str):
                conditions.append(f"{col} = '{val}'")
            else:
                conditions.append(f"{col} = {val}")

        if conditions:
            path_conditions.append(f"({' AND '.join(conditions)})")

    if path_conditions:
        return " OR ".join(path_conditions)

    return ""


def search_admin_level(
    term: str,
    level: int,
    con: DuckDBPyConnection,
    previous_results: pl.DataFrame | None = None,
    limit: int = 20,
) -> pl.DataFrame:
    """
    Search for a term with path-aware hierarchical filtering - simplified approach.
    """
    assert level in [1, 2, 3, 4], "Level must be between 1 and 4"
    table_name = f"admin{level}"

    # Base query for finding matches
    base_query = f"""
    SELECT *, fts_main_{table_name}.match_bm25(geonameId, $term) AS score
    FROM {table_name}
    WHERE score IS NOT NULL
    """

    # No filtering needed if no previous results
    if previous_results is None or previous_results.is_empty():
        query = f"{base_query} ORDER BY score DESC LIMIT $limit"
        results = con.execute(query, {"term": term, "limit": limit}).pl()
        return (
            results.lazy().pipe(admin_score, level).collect()
            if not results.is_empty()
            else results
        )

    # Find the highest level we've searched so far
    highest_level = -1
    for i in range(level):
        score_col = f"adjusted_score_{i}"
        if score_col in previous_results.columns:
            highest_level = i

    if highest_level >= 0:
        print(
            f"Highest searched level: {highest_level} (with '{f'adjusted_score_{highest_level}'}')"
        )

        # Extract the admin columns up to the highest searched level
        admin_cols = [f"admin{i}_code" for i in range(highest_level + 1)]

        try:
            # Extract unique combinations as path filters
            path_conditions = build_path_conditions(previous_results, admin_cols)

            if path_conditions:
                print(f"Filtering with path conditions: {path_conditions}")

                # Add path filtering to query
                query = f"""
                SELECT * FROM ({base_query})
                WHERE {path_conditions}
                ORDER BY score DESC
                LIMIT $limit
                """

                results = con.execute(query, {"term": term, "limit": limit}).pl()

                # If filtering returned results, process them
                if not results.is_empty():
                    # Add parent score if available
                    parent_score_col = f"adjusted_score_{highest_level}"
                    if parent_score_col in previous_results.columns:
                        # For simplicity, use the highest score for each path
                        # This is a simplification but avoids complex joins
                        max_scores = {}
                        for row in (
                            previous_results.select(admin_cols + [parent_score_col])
                            .unique()
                            .iter_rows(named=True)
                        ):
                            path_key = tuple(row[col] for col in admin_cols)
                            max_scores[path_key] = row[parent_score_col]

                        # Add parent scores to results
                        if max_scores:
                            results = results.with_columns(
                                pl.lit(list(max_scores.values())[0]).alias(
                                    parent_score_col
                                )
                            )
                            print(f"Added parent score from {parent_score_col}")
                else:
                    print("No results with path filtering, trying unfiltered search")
                    # Fallback to unfiltered search
                    results = con.execute(
                        f"{base_query} ORDER BY score DESC LIMIT $limit",
                        {"term": term, "limit": limit},
                    ).pl()
            else:
                print("No valid paths found, using unfiltered search")
                # No valid paths, use unfiltered search
                results = con.execute(
                    f"{base_query} ORDER BY score DESC LIMIT $limit",
                    {"term": term, "limit": limit},
                ).pl()
        except Exception as e:
            print(f"Error building path filters: {e}")
            # Error fallback
            results = con.execute(
                f"""
            SELECT * FROM {table_name}
            WHERE LOWER(name) LIKE '%' || LOWER($term) || '%'
            ORDER BY name
            LIMIT $limit
            """,
                {"term": term, "limit": limit},
            ).pl()
    else:
        # No previous levels searched, use basic search
        print("No previous levels searched, using basic search")
        results = con.execute(
            f"{base_query} ORDER BY score DESC LIMIT $limit",
            {"term": term, "limit": limit},
        ).pl()

    # Apply scoring
    if not results.is_empty():
        return results.lazy().pipe(admin_score, level).collect()

    return results


def hierarchical_search(
    search_terms: list[str | None], con: DuckDBPyConnection, limit: int = 10
) -> dict[str, pl.DataFrame]:
    """
    Perform hierarchical geographic search across admin levels.

    Args:
        search_terms: List of search terms, ordered by admin level (country, admin1, admin2, etc.)
        con: Database connection
        limit: Maximum results to return per level

    Returns:
        Dictionary mapping level names to search results
    """
    results = {}
    last_results = None
    # Ensure we have enough search terms
    if not search_terms:
        return results

    # Pad search terms if fewer than 5 are provided
    search_terms = search_terms + [None] * (5 - len(search_terms))

    # Level 0: Country search
    if search_terms[0]:
        logger.debug(f"Searching for country: '{search_terms[0]}'")
        country_results = search_country(search_terms[0], con, limit)
        if not country_results.is_empty():
            results["country"] = country_results
            last_results = country_results

    # Level 1: Admin1 search
    if search_terms[1]:
        logger.debug(f"Searching for admin1: '{search_terms[1]}'")
        admin1_results = search_admin_level(
            search_terms[1], 1, con, last_results, limit
        )
        if not admin1_results.is_empty():
            results["admin1"] = admin1_results
            last_results = admin1_results

    # Level 2: Admin2 search
    if search_terms[2]:
        logger.debug(f"Searching for admin2: '{search_terms[2]}'")
        admin2_results = search_admin_level(
            search_terms[2], 2, con, last_results, limit
        )
        if not admin2_results.is_empty():
            results["admin2"] = admin2_results
            last_results = admin2_results

    # Level 3: Admin3 search
    if search_terms[3]:
        logger.debug(f"Searching for admin3: '{search_terms[3]}'")
        admin3_results = search_admin_level(
            search_terms[3], 3, con, last_results, limit
        )
        if not admin3_results.is_empty():
            results["admin3"] = admin3_results
            last_results = admin3_results

    # Level 4: Admin4 search
    if search_terms[4]:
        logger.debug(f"Searching for admin4: '{search_terms[4]}'")
        admin4_results = search_admin_level(
            search_terms[4], 4, con, last_results, limit
        )
        if not admin4_results.is_empty():
            results["admin4"] = admin4_results
            last_results = admin4_results

    return results


# Search through the admin hierarchy
results = hierarchical_search(
    search_terms=["FR", "Provence-Alpes-Côte d'Azur", None, None, "Le Lavandou"],
    con=con,
)

# Access results for each level
if "country" in results:
    logger.debug("Country results:")
    print(results["country"])
if "admin1" in results:
    logger.debug("Admin1 results:")
    print(results["admin1"])
if "admin2" in results:
    logger.debug(
        "Admin2 results:",
    )
    print(results["admin2"])
if "admin3" in results:
    logger.debug(
        "Admin3 results:",
    )
    print(results["admin3"])
if "admin4" in results:
    logger.debug(
        "Admin4 results:",
    )
    print(results["admin4"])


# In[32]:


def backfill_hierarchy(row: dict, con: DuckDBPyConnection) -> dict:
    def get_where_clause(codes: list[str | None]) -> str:
        return "WHERE " + " AND ".join(
            [
                f"admin{i}_code = '{code}'"
                if code is not None
                else f"admin{i}_code IS NULL"
                for i, code in enumerate(codes)
            ]
        )

    hierarchy = {}
    codes = []
    for i in range(5):
        code = row.get(f"admin{i}_code")
        codes.append(code)
        if code is not None:
            df = con.execute(f"""
                SELECT geonameId, name FROM admin{i}
                {get_where_clause(codes)}
                LIMIT 1
            """).pl()
            if not df.is_empty():
                hierarchy[f"admin{i}"] = df.to_dicts()[0]
    return hierarchy


# In[34]:


results["admin4"]


# In[33]:


from pprint import pprint


results = hierarchical_search(
    search_terms=["FR", None, None, None, "Le Lavandou"], con=con
)
row = results["admin4"].row(0, named=True)
pprint(row)

pprint(backfill_hierarchy(row, con))


# In[30]:


results = hierarchical_search(
    search_terms=[None, "FL", None, "Lakeland", None], con=con
)
row = results["admin3"].row(0, named=True)


pprint(row)

pprint(backfill_hierarchy(row, con))


# In[29]:


# Search through the admin hierarchy
results = hierarchical_search(
    search_terms=[
        "FR",
        "Provence-Alpes-Côte d'Azur",
        "Var",
        "Arrondissement de Toulon",
        "Le Lavandou",
    ],
    con=con,
)

# Access results for each level
if "country" in results:
    logger.debug("Country results:", results["country"])
if "admin1" in results:
    logger.debug("Admin1 results:", results["admin1"])
if "admin2" in results:
    logger.debug("Admin2 results:", results["admin2"])
if "admin3" in results:
    logger.debug("Admin3 results:", results["admin3"])
if "admin4" in results:
    logger.debug("Admin4 results:", results["admin4"])
results["admin4"]


# In[ ]:


data = (
    con.execute("SELECT geonameId, latitude, longitude FROM allCountries")
    .pl()
    .select(
        pl.col("geonameId"),
        pl.concat_list(pl.col("latitude"), pl.col("longitude"))
        .cast(pl.Array(pl.Float32, 2))
        .alias("vectors"),
    )
)


# In[ ]:


my_coordinates1 = np.array([51.549902, -0.121696], dtype=np.float32)
my_coordinates2 = np.array([37.77493, -122.41942], dtype=np.float32)

vidx = VectorIndex("latlon", data, metric="haversine")


# In[ ]:


if (path := Path("./data/processed/latlon.index")).exists():
    logger.debug("Loading index...")
    index = Index.restore(path, view=True)
    if index is None:
        raise ValueError("Failed to load index")
else:
    logger.debug("Creating index...")
    coordinates = df.select(["latitude", "longitude"]).to_numpy(order="c")
    labels = df["geonameId"].to_numpy()
    index: Index = Index(ndim=2, metric="haversine", dtype="f32")
    index.add(keys=labels, vectors=coordinates, log=True)
    index.save(path)


# In[ ]:


# Example function to search and return results with distances
def search_with_distances(
    index: Index,
    my_coordinates: NDArray[np.float32],
    original_df: pl.LazyFrame,
    k=10,
    exact=False,
):
    # Perform the search
    output = index.search(vectors=my_coordinates, count=k, log=True, exact=exact)

    logger.debug(f"Visited members: {output.visited_members}")
    logger.debug(f"Computed distances: {output.computed_distances}")

    # Extract keys (geonameids) and distances
    keys = output.keys
    distances = output.distances

    # Create a DataFrame from the search results
    results_df = pl.LazyFrame(
        data={"geonameId": keys, "distance": distances},
        schema={"geonameId": pl.UInt32, "distance": pl.Float32},
    ).with_columns(pl.col("distance") * 6371.0)

    # Join the results with the original DataFrame to get detailed information
    detailed_results_df = results_df.join(original_df, on="geonameId", how="left")

    # Sort by distance
    sorted_results_df = detailed_results_df.sort("distance")

    return sorted_results_df.collect()


# In[ ]:


search_with_distances(index, my_coordinates2, df.lazy())


# In[ ]:


output: Matches = index.search(vectors=my_coordinates1, count=10, log=True)
logger.debug(f"{output.computed_distances=}")
logger.debug(f"{output.visited_members=}")
df.filter(pl.col("geonameId").is_in(output.keys))

