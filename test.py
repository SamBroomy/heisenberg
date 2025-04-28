#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
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
        print(f"Table '{table_name}' already exists")
        if not overwrite:
            return
        print(f"Overwriting table '{table_name}'")
        con.execute(f"DROP TABLE {table_name} CASCADE")
        print(f"Table '{table_name}' dropped")
    time_start = time()
    load = con.begin()
    try:
        print(f"Loading '{file_path}'...")
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
        print(f"Scan time: {time() - time_scan:.6f}s")

        q = q.with_columns(cs.by_dtype(pl.String).str.strip_chars().replace("", None))

        # Time collect
        time_collect = time()
        df = q.collect()
        print(f"Collect time: {time() - time_collect:.6f}s")

        # Time write
        time_write = time()
        save_path = Path(f"./data/processed/geonames/{table_name}.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(save_path.as_posix())
        print(f"Write time: {time() - time_write:.6f}s")

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

        print(f"Create time: {time() - time_create:.6f}s")

        time_commit = time()
        load.commit()
        print(f"Commit time: {time() - time_commit:.6f}s")
        analyze_time = time()
        con.execute("VACUUM ANALYZE;")
        print(f"Analyze time: {time() - analyze_time:.6f}s")
    except Exception as e:
        print(f"Error loading '{file_path}'")
        print(e.with_traceback(None))
        # Time rollback
        time_rollback = time()
        load.rollback()
        print(f"Rollback time: {time() - time_rollback:.6f}s")
        raise e
    finally:
        print(f"Total time: {time() - time_start:.6f}s")
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
    print("Table 'shapes' created")

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


# In[6]:


con.table("locations_full").pl()


# In[7]:


a = con.table("countryInfo").pl().join(con.table("allCountries").pl(), on="geonameId")


# In[8]:


a.select(pl.col("feature_code").value_counts(sort=True)).unnest("feature_code")


# In[9]:


a.filter(feature_code="PCLH")


# In[10]:


# Create country table
con.execute(sql_file("create_table_equivalent.sql")).pl()
con.execute(sql_file("create_table_admin0.sql")).execute("""PRAGMA create_fts_index(
    admin0,
    geonameId,
    name,
    asciiname,
    official_name,
    alternatenames,
    admin0_code,
    ISO3,
    ISO_Numeric,
    fips,
    stemmer = 'none',
    stopwords = 'none',
    ignore = '(\\.|[^a-z0-9])+',
    overwrite = 1
);""")

con.execute(sql_file("create_view_*_NODES.sql", table="admin0"))

con.execute(sql_file("create_view_*_FTS.sql", table="admin0"))


# In[11]:


# if (path := Path("./data/processed/latlon.index")).exists():
#     print("Loading index...")
#     index = Index.restore(path, view=True) or raise ValueError("Failed to load index")
# else:
#     print("Creating index...")
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
            print(
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
        print("Creating index...")
        vectors = df[self.main_column].to_numpy()
        labels = df[self.id_column].to_numpy()
        ndims = vectors.shape[1]  # Find n dims
        index: Index = Index(ndim=ndims, metric=metric, dtype="f32")
        index.add(keys=labels, vectors=vectors, log=True)
        index.save(self.index_path)
        return Ok(index)

    def get_index(self) -> Result[Index, str]:
        if (path := self.index_path).exists():
            print(f"Opening index at '{self.index_path}'")
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
        print("Opening index")
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

        print(f"Visited members: {output.visited_members}")
        print(f"Computed distances: {output.computed_distances}")

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
            print(
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
        print(f"Creating index with columns:\n{json.dumps(col_types, indent=2)}")

        index = tantivy.Index(schema, path=self.index_path.as_posix(), reuse=False)
        writer = index.writer()
        for row in df.rows(named=True):
            if split_field:
                for splitter, fields in split_field.items():
                    if isinstance(fields, str):
                        fields = [fields]
                    for field in fields:
                        print(f"Splitting {field} by {splitter}...")
                        row[field] = row[field].split(splitter)
            writer.add_document(tantivy.Document(**row))
        writer.commit()
        writer.wait_merging_threads()
        return Ok(index)

    def get_index(self) -> Result[tantivy.Index, str]:
        if tantivy.Index.exists(self.index_path.as_posix()):
            print(f"Opening index at '{self.index_path}'")
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
        print("Schema Loaded")
        print("Opening country index")
        return Ok(index)

    def convert_fts_results(
        self, hits: tantivy.SearchResult, searcher: tantivy.Searcher
    ) -> pl.DataFrame:
        print(f"FTS hits from search: {hits.count}")  # type: ignore

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

        print(final_query)

        hits: tantivy.SearchResult = searcher.search(final_query, limit=limit)

        if hits.count == 0:  # type: ignore
            if phrase:
                print("No results found, retrying without phrase search...")
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


# In[12]:


# 1. PREPARE THE BASE DATA
# Start with a comprehensive temporary table containing all admin entities
con.execute("""
    CREATE OR REPLACE TEMP TABLE admin_entities AS
    SELECT
        a.geonameId,
        a.name,
        a.asciiname,
        a.alternatenames,
        a.latitude,
        a.longitude,
        a.feature_class,
        a.feature_code,
        a.admin0_code,
        a.admin1_code,
        a.admin2_code,
        a.admin3_code,
        a.admin4_code,
        CASE
            WHEN a.feature_code IN ('PCLI', 'PCLD', 'PCLF', 'PCLS', 'PCL', 'TERR') THEN 0
            WHEN a.feature_code LIKE 'ADM1%' THEN 1
            WHEN a.feature_code LIKE 'ADM2%' THEN 2
            WHEN a.feature_code LIKE 'ADM3%' THEN 3
            WHEN a.feature_code LIKE 'ADM4%' THEN 4
            ELSE NULL
        END AS admin_level,
        h.parentId
    FROM
        allCountries a
    LEFT JOIN
        hierarchy h ON a.geonameId = h.childId AND h.type = 'ADM'
    WHERE
        a.feature_class = 'A' AND
        (a.feature_code IN ('PCLI', 'PCLD', 'PCLF', 'PCLS', 'PCL', 'TERR') OR
         a.feature_code LIKE 'ADM%')
    """)
con.table("admin_entities").pl()


# In[13]:


# 2. BUILD THE HIERARCHY USING RECURSIVE CTE
# This is a key optimization - using a recursive CTE to build the full hierarchy
# in a single SQL operation instead of multiple database calls
con.execute("""
CREATE OR REPLACE TABLE admin_hierarchy_full AS
WITH RECURSIVE hierarchy_path(geonameId, name, admin_level, id_path, name_path, depth) AS (
    -- Base case: Start with top-level entities (countries)
    SELECT
        geonameId,
        name,
        admin_level,
        [geonameId]::INTEGER[], -- Start with array containing just this ID
        [name]::VARCHAR[],      -- Start with array containing just this name
        0                        -- Depth starts at 0
    FROM
        admin_entities
    WHERE
        admin_level = 0

    UNION ALL

    -- Recursive case: Add children to the path
    SELECT
        c.geonameId,
        c.name,
        c.admin_level,
        p.id_path || [c.geonameId]::INTEGER[],
        p.name_path || [c.name]::VARCHAR[], -- Append child name to name array
        p.depth + 1                 -- Increment depth
    FROM
        admin_entities c
    JOIN
        hierarchy_path p ON c.parentId = p.geonameId
    WHERE
        c.admin_level > 0 AND p.depth < 5 -- Prevent infinite recursion
)
SELECT
    geonameId,
    name,
    admin_level,
    id_path,
    name_path,
    depth,
    -- Parent ID is the second-to-last element in the path array
    CASE
        WHEN array_length(id_path) > 1 THEN
            id_path[array_length(id_path) - 1]
        ELSE NULL
    END AS parent_id,
    -- Root ID is always the first element in the path array
    id_path[1] AS root_id,
    -- Create a simple string representation for display
    array_to_string(list_reverse(name_path), ', ') AS path_display
FROM
    hierarchy_path
ORDER BY
    id_path, admin_level;

    """)
con.table("admin_hierarchy_full").pl()


# In[14]:


# 3. HANDLE ORPHANED ENTITIES (Those without proper hierarchy linkage)
con.execute("""
    CREATE OR REPLACE TEMP TABLE orphaned_entities AS
    SELECT a.*
    FROM admin_entities a
    LEFT JOIN admin_hierarchy_full h ON a.geonameId = h.geonameId
    WHERE h.geonameId IS NULL
    """)
con.execute("ANALYZE admin_hierarchy_full")
con.execute("ANALYZE orphaned_entities")
con.table("orphaned_entities").pl()


# In[15]:


con.execute("""
CREATE OR REPLACE TEMP TABLE fixed_orphans AS

-- Level 1 orphans (join with countries)
SELECT
    o.geonameId,
    o.name,
    o.admin_level,
    h.geonameId AS parent_id
FROM
    orphaned_entities o
JOIN
    admin_entities c ON o.admin0_code = c.admin0_code
JOIN
    admin_hierarchy_full h ON c.geonameId = h.geonameId AND h.admin_level = 0
WHERE
    o.admin_level = 1

UNION ALL

-- Level 2 orphans (join with admin1)
SELECT
    o.geonameId,
    o.name,
    o.admin_level,
    h.geonameId AS parent_id
FROM
    orphaned_entities o
JOIN
    admin_entities a1 ON o.admin0_code = a1.admin0_code
                      AND o.admin1_code = a1.admin1_code
JOIN
    admin_hierarchy_full h ON a1.geonameId = h.geonameId AND h.admin_level = 1
WHERE
    o.admin_level = 2

UNION ALL

-- Level 3 orphans (join with admin2)
SELECT
    o.geonameId,
    o.name,
    o.admin_level,
    h.geonameId AS parent_id
FROM
    orphaned_entities o
JOIN
    admin_entities a2 ON o.admin0_code = a2.admin0_code
                      AND o.admin1_code = a2.admin1_code
                      AND o.admin2_code = a2.admin2_code
JOIN
    admin_hierarchy_full h ON a2.geonameId = h.geonameId AND h.admin_level = 2
WHERE
    o.admin_level = 3

UNION ALL

-- Level 4 orphans (join with admin3)
SELECT
    o.geonameId,
    o.name,
    o.admin_level,
    h.geonameId AS parent_id
FROM
    orphaned_entities o
JOIN
    admin_entities a3 ON o.admin0_code = a3.admin0_code
                      AND o.admin1_code = a3.admin1_code
                      AND o.admin2_code = a3.admin2_code
                      AND o.admin3_code = a3.admin3_code
JOIN
    admin_hierarchy_full h ON a3.geonameId = h.geonameId AND h.admin_level = 3
WHERE
    o.admin_level = 4;
    """)
con.table("fixed_orphans").pl()


# In[16]:


# 4. CREATE ADMIN TABLES WITH UNIFIED APPROACH
# Now create each admin level table with consistent structure
admin_levels = range(0, 5)  # levels 0-4
overwrite = True

for level in admin_levels:
    table_name = f"admin{level}"
    if table_exists(con, table_name) and not overwrite:
        print(f"Table {table_name} exists, skipping (use overwrite=True to replace)")
        continue

    print(f"Creating {table_name} table...")

    # Create the base table with rich information
    if level == 0:  # Special case for admin0 (countries)
        con.execute(f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT
        a.geonameId,
        a.name,
        a.asciiname,
        a.admin0_code,
        a.cc2,
        c.ISO,
        c.ISO3,
        c.ISO_Numeric,
        c.Country AS official_name,
        c.fips,
        c.Population,
        c.Area,
        a.alternatenames,
        a.feature_class,
        a.feature_code,
        -- Simple reference to root ID (same as own ID for countries)
        a.geonameId AS root_id,
        -- We don't need full path arrays for countries
        NULL AS parent_id
    FROM
        admin_hierarchy_full h
    JOIN
        allCountries a ON h.geonameId = a.geonameId
    JOIN
        countryInfo c ON a.geonameId = c.geonameId
    WHERE
        h.admin_level = 0
    ORDER BY
        a.geonameId
    """)
    else:
        # For admin levels 1-4
        parent_level = level - 1
        con.execute(f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT
        a.geonameId,
        a.name,
        a.asciiname,
        -- Admin codes (only include what's needed)
        a.admin0_code,
        a.admin1_code
        {', a.admin2_code' if level >= 2 else ', NULL AS admin2_code'}
        {', a.admin3_code' if level >= 3 else ', NULL AS admin3_code'}
        {', a.admin4_code' if level >= 4 else ', NULL AS admin4_code'},
        -- Feature classification
        a.feature_class,
        a.feature_code,
        a.population,
        -- Hierarchy references (just the essential IDs, not full arrays)
        h.root_id AS country_id,
        h.parent_id,
        -- Names for display and search
        r.name AS country_name,
        p.name AS parent_name,
        a.alternatenames
    FROM
        admin_hierarchy_full h
    JOIN
        allCountries a ON h.geonameId = a.geonameId
    LEFT JOIN
        allCountries p ON h.parent_id = p.geonameId
    JOIN
        allCountries r ON h.root_id = r.geonameId
    WHERE
        h.admin_level = {level}

    UNION ALL

    -- Add any fixed orphans with slim structure
    SELECT
        a.geonameId,
        a.name,
        a.asciiname,
        -- Admin codes
        a.admin0_code,
        a.admin1_code
        {', a.admin2_code' if level >= 2 else ', NULL AS admin2_code'}
        {', a.admin3_code' if level >= 3 else ', NULL AS admin3_code'}
        {', a.admin4_code' if level >= 4 else ', NULL AS admin4_code'},
        -- Feature classification
        a.feature_class,
        a.feature_code,
        p.population,
        -- Links to country
        (SELECT geonameId FROM allCountries
         WHERE admin0_code = a.admin0_code AND feature_code = 'PCLI' LIMIT 1) AS country_id,
        f.parent_id,
        -- Names for display and search
        c.name AS country_name,
        p.name AS parent_name,
        a.alternatenames
    FROM
        fixed_orphans f
    JOIN
        admin_entities a ON f.geonameId = a.geonameId
    LEFT JOIN
        allCountries p ON f.parent_id = p.geonameId
    JOIN
        allCountries c ON a.admin0_code = c.admin0_code AND c.feature_code LIKE 'PCL%'
    WHERE
        a.admin_level = {level}
    ORDER BY
        geonameId
    """)
    # Create primary key and other indexes
    con.execute(f"CREATE INDEX idx_{table_name}_gid ON {table_name} (geonameId)")

    if level > 0:
        # Create indexes for parent relationships
        con.execute(f"CREATE INDEX idx_{table_name}_parent ON {table_name} (parent_id)")
        con.execute(f"CREATE INDEX idx_{table_name}_admin0_code ON {table_name} (admin0_code)")

        # Create admin code index specific to this level
        con.execute(f"CREATE INDEX idx_{table_name}_code ON {table_name} (admin{level}_code)")

    # Create FTS index with optimized fields for search
    create_fts_fields = "geonameId, name, asciiname, alternatenames"
    if level == 0:
        create_fts_fields += ", official_name, ISO, ISO3, fips"


    con.execute(f"""
    PRAGMA create_fts_index(
        {table_name},
        {create_fts_fields},
        stemmer = 'none',
        stopwords = 'none',
        ignore = '(\\.|[^a-z0-9])+',
        overwrite = 1
    )
    """)

    # Report count
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"Created {table_name} with {count} entities")


# In[17]:


print("Creating unified admin search view...")
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
    NULL AS parent_name,  -- Add NULL placeholder for admin0
    parent_id,
    feature_class,
    feature_code,
    population,
    alternatenames,
    'admin0' AS source_table
FROM
    admin0


    UNION ALL

    -- Admin1 entities (states/provinces)
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

    -- Admin2 entities (counties/districts)

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

    -- Admin3 entities (municipalities)

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

    -- Admin4 entities (neighborhoods/villages)
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

    # Clean up temporary tables


print("Admin hierarchy construction complete!")


# In[18]:


con.execute("DROP TABLE IF EXISTS admin_entities")
con.execute("DROP TABLE IF EXISTS admin_hierarchy_full")
con.execute("DROP TABLE IF EXISTS orphaned_entities")
con.execute("DROP TABLE IF EXISTS fixed_orphans")


# In[19]:


entities_df = con.execute(f"""
    SELECT {GID}, name, feature_class, feature_code
    FROM unique_ids
""").pl()

hierarchy_df = con.execute("""
    SELECT parentId, childId, type
    FROM hierarchy
""").pl()
# Optional: Print some diagnostics
print(
    f"Loaded {len(entities_df)} entities and {len(hierarchy_df)} hierarchical relationships"
)
print("Entity columns:", entities_df.columns)
print("Hierarchy columns:", hierarchy_df.columns)

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
    print("Created Entity table")

if "IsIn" not in conn.execute("CALL SHOW_TABLES() RETURN *;").get_as_pl().get_column(
    "name"
):
    conn.execute(sql_file("create_relation_IsIn.sql"))
    print("Created IsIn table")

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
    print("Loaded Entity")

if not are_edges:
    conn.execute(
        "COPY IsIn FROM (LOAD FROM hierarchy_df RETURN parentId, childId, type)"
    )
    print("Loaded IsIn")


# In[49]:


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


# In[24]:


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
        print("No orphaned admin entities found!")
        return

    print(f"Found {len(orphans_df)} orphaned admin entities.")

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
    for orphan in orphans_df.filter(~pl.col("geonameId").is_in(parent_links["geonameId"])).iter_rows(named=True):
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
                        SELECT parent.geonameId FROM admin{level-1} parent
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
                    admin{level-1} parent ON
                        parent.admin0_code = a.admin0_code
                        {"AND parent.admin1_code = a.admin1_code" if level >= 2 else ""}
                        {"AND parent.admin2_code = a.admin2_code" if level >= 3 else ""}
                        {"AND parent.admin3_code = a.admin3_code" if level >= 4 else ""}
                WHERE
                    a.geonameId = {orphan["geonameId"]}
            )
            """)
            code_linked += 1


    print(f"Fixed {len(parent_links)} orphans using graph relationships and {code_linked} using admin codes")

def create_admin_search_view(con):
    """Create a unified view for searching across all admin levels."""

    print("Creating unified admin search view...")
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

    print("Admin search view created!")


# In[25]:


def build_admin_tables_hybrid(con, conn, overwrite=True):
    """Build admin tables using graph database for relationships and DuckDB for storage."""

    print("Starting hybrid admin table construction...")

    # Define admin level feature code patterns
    level_codes = {
        0: ["PCL", "PCLI", "PCLD", "PCLF", "PCLS", "TERR"],
        1: ["ADM1", "ADM1H"],
        2: ["ADM2", "ADM2H"],
        3: ["ADM3", "ADM3H"],
        4: ["ADM4", "ADM4H"]
    }

    # Track entities at each admin level
    admin_entities = {}

    # Build admin tables one by one
    for level in range(0, 5):
        table_name = f"admin{level}"

        if table_exists(con, table_name) and not overwrite:
            print(f"Table {table_name} already exists. Skipping.")
            continue

        print(f"Building {table_name} table...")

        # Step 1: Identify entities of this admin level by feature code
        feature_patterns = "', '".join([code for code in level_codes[level]])
        base_entities_query = f"""
            SELECT geonameId
            FROM allCountries
            WHERE feature_code IN ('{feature_patterns}')
            OR feature_code LIKE '{level_codes[level][0]}%'
        """

        level_entities = set(con.execute(base_entities_query).pl()["geonameId"])
        print(f"Found {len(level_entities)} initial entities for level {level}")

        # Step 2: For levels 1-4, use graph DB to validate hierarchical placement
        # This ensures entities are correctly placed in the admin hierarchy
        if level > 0 and level-1 in admin_entities:
            # Find all direct children of the previous level's entities
            previous_level_ids = list(admin_entities[level-1])

            # Use graph DB to get valid children
            children_query = f"""
            MATCH (p:Entity)-[:IsIn]->(c:Entity)
            WHERE p.geonameId IN CAST({previous_level_ids}, "UINT32[]")
            RETURN DISTINCT c.geonameId AS geonameId
            """

            valid_children = set(conn.execute(children_query).get_as_pl()["geonameId"])
            print(f"Found {len(valid_children)} valid children from graph relationships")

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
                a.geonameId IN ({','.join(map(str, level_entities))})
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
                    childId IN ({','.join(map(str, level_entities))})
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
                a.geonameId IN ({','.join(map(str, level_entities))})
            ORDER BY
                a.geonameId
            """

        # Execute the query to create the table
        con.execute(create_query)

        # Step 4: Add indexes and FTS
        con.execute(f"CREATE INDEX idx_{table_name}_gid ON {table_name} (geonameId)")

        if level > 0:
            # Create indexes for parent relationships
            con.execute(f"CREATE INDEX idx_{table_name}_parent ON {table_name} (parent_id)")
            con.execute(f"CREATE INDEX idx_{table_name}_admin0 ON {table_name} (admin0_code)")

            # Create admin code index specific to this level
            if level <= 4:
                con.execute(f"CREATE INDEX idx_{table_name}_code ON {table_name} (admin{level}_code)")

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
        print(f"Created {table_name} with {count} entities")

    # Step 5: Handle orphaned entities
    print("\nChecking for orphaned administrative entities...")
    handle_orphaned_admin_entities(con, conn)

    # Step 6: Create a unified admin search view
    create_admin_search_view(con)

    print("Hybrid admin table construction complete!")


# In[26]:


build_admin_tables_hybrid(con, conn, overwrite=True)


# In[29]:


con.table("admin1").pl()


# In[ ]:


def build_clean_admin_hierarchy(create_views=False, create_fts=True, overwrite=False):
    """Build admin hierarchy with strict level separation and clean boundaries."""
    print("Building clean administrative hierarchy...")

    # Check if country table exists
    if not table_exists(con, "admin0"):
        print("Error: admin0 table must exist first")
        return

    # Define the distinct feature codes for each level
    level_codes = {
        1: ["ADM1", "ADM1H"],
        2: ["ADM2", "ADM2H"],
        3: ["ADM3", "ADM3H"],
        4: ["ADM4", "ADM4H"],
    }

    # Get all country IDs first to exclude them from all admin tables
    country_ids = set(con.execute("SELECT geonameId FROM admin0").pl().to_series())
    admin_ids = {0: country_ids}  # Level 0 = countries

    # Process each level in sequence
    for level in range(1, 5):
        print(f"\n=== Building admin{level} table ===")

        # Skip if exists and not overwriting
        if table_exists(con, f"admin{level}") and not overwrite:
            print(f"Table admin{level} already exists. Skipping.")
            admin_ids[level] = set(
                con.execute(f"SELECT geonameId FROM admin{level}").pl().to_series()
            )
            continue

        # Get candidates with the correct feature code
        feature_code_pattern = ", ".join([f"'{code}'" for code in level_codes[level]])
        code_candidates = set(
            con.execute(f"""
            SELECT geonameId
            FROM allCountries
            WHERE feature_code IN ({feature_code_pattern})
        """)
            .pl()
            .to_series()
        )

        # Get candidates from graph database parentage (direct children of previous level)
        graph_candidates = set()
        if level > 1 and level - 1 in admin_ids:
            parent_ids = admin_ids[level - 1]
            if parent_ids:
                children_df = conn.execute(
                    get_children_querys(list(parent_ids))
                ).get_as_pl()
                if not children_df.is_empty():
                    graph_candidates = set(children_df["geonameId"])

        # Combine candidates
        all_candidates = code_candidates.union(graph_candidates)

        # Build exclusion set - exclude all entities from other admin levels
        excluded_ids = set().union(*[ids for ids in admin_ids.values()])

        # Get final set of IDs for this level
        final_ids = all_candidates - excluded_ids

        if not final_ids:
            print(f"Warning: No entities found for admin{level} table!")
            admin_ids[level] = set()
            continue

        # Create strict level condition
        level_condition = f"a.feature_code LIKE 'ADM{level}%' AND"
        if level == 1:
            # For admin1, ensure no admin2, admin3, or admin4 codes
            level_condition += """
                (admin1_code IS NOT NULL) AND
                (admin2_code IS NULL) AND
                (admin3_code IS NULL) AND
                (admin4_code IS NULL)
            """
        elif level == 2:
            # For admin2, ensure admin1_code exists, but no admin3 or admin4 codes
            level_condition += """
                (admin2_code IS NOT NULL) AND
                (admin3_code IS NULL) AND
                (admin4_code IS NULL)
            """
        elif level == 3:
            # For admin3, ensure admin1 and admin2 codes exist, but no admin4
            level_condition += """
                (admin3_code IS NOT NULL) AND
                (admin4_code IS NULL)
            """
        else:  # level == 4
            # For admin4, ensure admin1, admin2, and admin3 codes exist
            level_condition = """
                (admin4_code IS NOT NULL)
            """

        # Create the table with SQL
        print(f"Creating admin{level} table with strict level constraints...")
        con.execute(f"""
        CREATE OR REPLACE TABLE admin{level} AS
        SELECT a.*
        FROM allCountries a
        WHERE
            a.geonameId IN ({",".join(map(str, final_ids))})
            AND {level_condition}
        ORDER BY a.geonameId
        """)

        # Store IDs for next level
        admin_ids[level] = set(
            con.execute(f"SELECT geonameId FROM admin{level}").pl().to_series()
        )

        # Create indexes and FTS
        print(f"Creating index on admin{level}...")
        con.execute(
            f"CREATE INDEX IF NOT EXISTS admin{level}_gid ON admin{level} (geonameId)"
        )

        if create_views:
            con.execute(sql_file("create_view_*_NODES.sql", table=f"admin{level}"))
            con.execute(sql_file("create_view_*_FTS.sql", table=f"admin{level}"))

        if create_fts:
            con.execute(f"""
            PRAGMA create_fts_index(
                admin{level},
                geonameId, name, asciiname, alternatenames, admin{level}_code,
                stemmer='none', stopwords='none', ignore='(\\.|[^a-z0-9])+', overwrite=1
            )
            """)

        # Report results
        count = con.execute(f"SELECT COUNT(*) FROM admin{level}").fetchone()[0]
        print(f"Completed admin{level} table with {count} entities")

    # Print final statistics
    print("\nAdministrative hierarchy successfully built!")
    for level in range(0, 5):
        table = "country" if level == 0 else f"admin{level}"
        if table_exists(con, table):
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count} entities")


def create_admin_indexes(con: DuckDBPyConnection):
    # Admin0 (already has indexes on ISO codes from FTS)

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

    print("Created composite indexes on admin code columns")


#build_clean_admin_hierarchy(overwrite=True)
create_admin_indexes(con)


# In[64]:


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
            "adjusted_score_0",
            "admin0_code",
            "feature_class",
            "feature_code",
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
            print(f"Found parent score column: {parent_score_column}")
            available_parent_scores.append((parent_level, parent_score_column))
            break  # Use the highest available parent level
    if available_parent_scores:
        # Use the highest available parent score
        parent_level, parent_score_column = available_parent_scores[0]
        print(f"Using parent score from level {parent_level}: {parent_score_column}")

        # Apply scoring with parent influence
        df = df.with_columns(
            (
                (pl.col("score") * pl.col("pop_multiplier")).pow(1 - parent_weight)
                * pl.col(parent_score_column).pow(parent_weight)
            ).alias(score_column),
        )
    else:
        # No parent scores available, use only the current score
        print("No parent score columns found. Using only current level score.")
        df = df.with_columns(
            (pl.col("score") * pl.col("pop_multiplier")).alias(score_column),
        )
    # Return sorted and selected columns
    return df.sort(score_column, descending=True).select(
        "geonameId",
        "name",
        "feature_class",
        "feature_code",
        cs.starts_with("admin"),
        cs.starts_with("adjusted_score"),
    )

    if parent_score_column in df.collect_schema().keys():
        print(f"Parent score column: {parent_score_column}")
        df = df.with_columns(
            (
                ((1 - parent_weight) * pl.col("score") * pl.col("pop_multiplier"))
                + (parent_weight * pl.col(parent_score_column))
            ).alias(score_column),
        )
    else:
        print(f"Parent score column: {parent_score_column} not found")
        df = df.with_columns(
            (pl.col("score") * pl.col("pop_multiplier")).alias(score_column),
        )

    return df.sort(score_column, descending=True).select(
        "geonameId",
        "name",
        "feature_class",
        "feature_code",
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


def search_admin_level(
    term: str,
    level: int,
    con: DuckDBPyConnection,
    previous_results: pl.DataFrame | None = None,
    limit: int = 20,
) -> pl.DataFrame:
    """
    Search for a term at the specified admin level with built-in resilience.

    Args:
        term: Search term
        level: Admin level (1=admin1, 2=admin2, etc.)
        con: Database connection
        previous_results: Optional results from previous level search
        limit: Maximum results to return
    """
    assert level in [1, 2, 3, 4], "Level must be between 1 and 4"
    table_name = f"admin{level}"
    parent_level = level - 1
    parent_table = f"admin{parent_level}"
    parent_code_col = f"admin{parent_level}_code"
    parent_score_col = f"adjusted_score_{parent_level}"

    # Check if previous results exist and have data
    has_previous = previous_results is not None and not previous_results.is_empty()

    # Base query - always needed
    base_query = f"""
    SELECT *, fts_main_{table_name}.match_bm25(geonameId, $term) AS score
    FROM {table_name}
    WHERE score IS NOT NULL
    """

    if has_previous:
        # Register previous results
        temp_table = f"temp_previous_{level}"
        con.register(temp_table, previous_results)
        # Determine which admin level the previous results are from
        parent_level = None

        # Check available columns to determine parent level
        if "admin0_code" in previous_results.columns:
            # Previous results include country info
            parent_level = 0

        for i in range(1, level):
            if f"adjusted_score_{i}" in previous_results.columns:
                parent_level = i

        if parent_level is not None:
            # We found a valid parent level
            if parent_level == 0:
                # If parent is country, filter by admin0_code
                parent_code_col = "admin0_code"
                parent_score_col = "adjusted_score_0"
            else:
                # Otherwise use the appropriate admin code
                parent_code_col = f"admin{parent_level}_code"
                parent_score_col = f"adjusted_score_{parent_level}"

            # Check if parent scores exist
            has_parent_scores = parent_score_col in previous_results.columns

            # With previous results, we can filter by parent and include parent scores
            if has_parent_scores:
                print(
                    f"Filtering by parent level {parent_level} using {parent_code_col}"
                )
                query = f"""
                WITH parent_data AS (
                    SELECT DISTINCT {parent_code_col}, {parent_score_col}
                    FROM {temp_table}
                    WHERE {parent_score_col} IS NOT NULL
                )
                SELECT a.*, p.{parent_score_col}
                FROM ({base_query}) a
                LEFT JOIN parent_data p ON a.{parent_code_col} = p.{parent_code_col}
                WHERE a.{parent_code_col} IN (SELECT {parent_code_col} FROM {temp_table})
                ORDER BY a.score DESC
                LIMIT $limit
                """
            else:
                # No parent scores available, but we can still filter by parent
                print(
                    f"Filtering by parent level {parent_level} using {parent_code_col} (no scores)"
                )
                # No parent scores available, but we can still filter by parent
                query = f"""
                SELECT a.*
                FROM ({base_query}) a
                WHERE a.{parent_code_col} IN (SELECT {parent_code_col} FROM {temp_table})
                ORDER BY a.score DESC
                LIMIT $limit
                """
        else:
            # Couldn't determine parent level, fall back to unfiltered search
            print("Could not determine parent level for filtering")
            query = f"""
            {base_query}
            ORDER BY score DESC
            LIMIT $limit
            """
    else:
        # Simple search without parent filtering
        query = f"""
        {base_query}
        ORDER BY score DESC
        LIMIT $limit
        """

    # Execute the query
    results = con.execute(query, {"term": term, "limit": limit}).pl()

    # If we got no results with filtering, try without filtering
    if has_previous and results.is_empty():
        print("No results found with parent filtering. Trying unfiltered search...")
        fallback_query = f"""
        {base_query}
        ORDER BY score DESC
        LIMIT $limit
        """
        results = con.execute(fallback_query, {"term": term, "limit": limit}).pl()

    # Apply admin-specific scoring
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
        print(f"Searching for country: '{search_terms[0]}'")
        country_results = search_country(search_terms[0], con, limit)
        if not country_results.is_empty():
            results["country"] = country_results
            last_results = country_results

    # Level 1: Admin1 search
    if search_terms[1]:
        print(f"Searching for admin1: '{search_terms[1]}'")
        admin1_results = search_admin_level(
            search_terms[1], 1, con, last_results, limit
        )
        if not admin1_results.is_empty():
            results["admin1"] = admin1_results
            last_results = admin1_results

    # Level 2: Admin2 search
    if search_terms[2]:
        print(f"Searching for admin2: '{search_terms[2]}'")
        admin2_results = search_admin_level(
            search_terms[2], 2, con, last_results, limit
        )
        if not admin2_results.is_empty():
            results["admin2"] = admin2_results
            last_results = admin2_results

    # Level 3: Admin3 search
    if search_terms[3]:
        print(f"Searching for admin3: '{search_terms[3]}'")
        admin3_results = search_admin_level(
            search_terms[3], 3, con, last_results, limit
        )
        if not admin3_results.is_empty():
            results["admin3"] = admin3_results
            last_results = admin3_results

    # Level 4: Admin4 search
    if search_terms[4]:
        print(f"Searching for admin4: '{search_terms[4]}'")
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
    print("Country results:", results["country"])
if "admin1" in results:
    print("Admin1 results:", results["admin1"])
if "admin2" in results:
    print("Admin2 results:", results["admin2"])
if "admin3" in results:
    print("Admin3 results:", results["admin3"])
if "admin4" in results:
    print("Admin4 results:", results["admin4"])


# In[ ]:


def backfill_hierarchy(row: dict, con: DuckDBPyConnection) -> dict:
    """
    Backfill complete hierarchical information for a given location.

    Args:
        row: A dictionary or DataFrame row containing admin codes
        con: Database connection

    Returns:
        Dictionary mapping admin levels to their corresponding entity information
    """

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


# In[79]:


from pprint import pprint



results = hierarchical_search(
    search_terms=["FR", None, None, None, "Le Lavandou"], con=con
)
row = results["admin4"].row(0, named=True)
pprint(row)

pprint(backfill_hierarchy(row, con))


# In[80]:


results = hierarchical_search(
    search_terms=[None, "FL", None, "Lakeland", None], con=con
)
row = results["admin3"].row(0, named=True)
from pprint import pprint

pprint(row)

pprint(backfill_hierarchy(row, con))


# In[68]:


con.table("admin2").pl().filter(pl.col("admin2_code") == "105")


# In[ ]:


def backfill_hierarchy(row: dict, con: DuckDBPyConnection) -> dict:
    """
    Backfill complete hierarchical information for a given location.

    Args:
        row: A dictionary or DataFrame row containing admin codes
        con: Database connection

    Returns:
        Dictionary mapping admin levels to their corresponding entity information
    """
    hierarchy = {}

    # Extract admin codes from the row
    admin0_code = row.get("admin0_code")
    admin1_code = row.get("admin1_code")
    admin2_code = row.get("admin2_code")
    admin3_code = row.get("admin3_code")
    admin4_code = row.get("admin4_code")

    # Try to find country (admin0)
    if admin0_code:
        admin0_df = con.execute(f"""
            SELECT * FROM admin0
            WHERE admin0_code = '{admin0_code}'
            LIMIT 1
        """).pl()
        if not admin0_df.is_empty():
            hierarchy["admin0"] = admin0_df[0].to_dict()

    # Try to find state/province (admin1)
    if admin0_code and admin1_code:
        admin1_df = con.execute(f"""
            SELECT * FROM admin1
            WHERE admin0_code = '{admin0_code}'
            AND admin1_code = '{admin1_code}'
            LIMIT 1
        """).pl()
        if not admin1_df.is_empty():
            hierarchy["admin1"] = admin1_df[0].to_dict()

    # Try to find county/district (admin2)
    if admin0_code and admin1_code and admin2_code:
        admin2_df = con.execute(f"""
            SELECT * FROM admin2
            WHERE admin0_code = '{admin0_code}'
            AND admin1_code = '{admin1_code}'
            AND admin2_code = '{admin2_code}'
            LIMIT 1
        """).pl()
        if not admin2_df.is_empty():
            hierarchy["admin2"] = admin2_df[0].to_dict()

    # Try to find sub-district (admin3)
    if admin0_code and admin1_code and admin2_code and admin3_code:
        admin3_df = con.execute(f"""
            SELECT * FROM admin3
            WHERE admin0_code = '{admin0_code}'
            AND admin1_code = '{admin1_code}'
            AND admin2_code = '{admin2_code}'
            AND admin3_code = '{admin3_code}'
            LIMIT 1
        """).pl()
        if not admin3_df.is_empty():
            hierarchy["admin3"] = admin3_df[0].to_dict()

    # Try to find locality (admin4)
    if admin0_code and admin1_code and admin2_code and admin3_code and admin4_code:
        admin4_df = con.execute(f"""
            SELECT * FROM admin4
            WHERE admin0_code = '{admin0_code}'
            AND admin1_code = '{admin1_code}'
            AND admin2_code = '{admin2_code}'
            AND admin3_code = '{admin3_code}'
            AND admin4_code = '{admin4_code}'
            LIMIT 1
        """).pl()
        if not admin4_df.is_empty():
            hierarchy["admin4"] = admin4_df[0].to_dict()

    return hierarchy


# In[ ]:


con.table("admin1").pl().filter(pl.col("admin0_code") == "US").filter(
    pl.col("admin1_code") == "FL"
)


# In[ ]:


con.table("admin0").pl().filter(pl.col("admin0_code") == "US")


# In[ ]:


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
    print("Country results:", results["country"])
if "admin1" in results:
    print("Admin1 results:", results["admin1"])
if "admin2" in results:
    print("Admin2 results:", results["admin2"])
if "admin3" in results:
    print("Admin3 results:", results["admin3"])
if "admin4" in results:
    print("Admin4 results:", results["admin4"])
results["admin4"]


# In[ ]:


con.execute("""SELECT *
FROM (
    SELECT *, fts_main_admin0.match_bm25(
        geonameId,
        'EN'
    ) AS score
    FROM admin0
) sq
WHERE score IS NOT NULL
ORDER BY score DESC;
""").pl()  # .filter(pl.col("score")>1).select("ISO3")


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
            "adjusted_score_0",
            "admin0_code",
            "feature_class",
            "feature_code",
        )
    )


def admin_score(
    df: pl.LazyFrame, level: int, parent_weight: float = 0.3
) -> pl.LazyFrame:
    assert level in [1, 2, 3, 4], "Level must be between 1 and 4"
    score_column = f"adjusted_score_{level}"
    parent_score_column = f"adjusted_score_{level - 1}"
    parent_weight = 1 - parent_weight

    df = df.with_columns(
        pop_multiplier=(1 + (pl.col("population").add(1).log10() / 10)),
    )
    print(df.collect_schema().keys())

    if parent_score_column in df.collect_schema().keys():
        print(f"Parent score column: {parent_score_column}")
        df = df.with_columns(
            (
                ((1 - parent_weight) * pl.col("score") * pl.col("pop_multiplier"))
                + (parent_weight * pl.col(parent_score_column))
            ).alias(score_column),
        )
    else:
        print(f"Parent score column: {parent_score_column} not found")
        df = df.with_columns(
            (pl.col("score") * pl.col("pop_multiplier")).alias(score_column),
        )

    return df.sort(score_column, descending=True).select(
        "geonameId",
        "name",
        "feature_class",
        "feature_code",
        cs.starts_with("admin"),
        cs.starts_with("adjusted_score"),
    )


SEARCH_TERM = [
    "FR",
    "Provence-Alpes-Côte d'Azur",
    "Var",
    "Arrondissement de Toulon",
    "Le Lavandou",
]

# df_con = (
#     con.execute(
#         """SELECT *,
#     -- High fixed score for exact matches
#     CASE
#         WHEN LOWER(ISO) = LOWER($term) THEN 10.0
#         WHEN LOWER(ISO3) = LOWER($term) THEN 8.0
#         WHEN LOWER(fips) = LOWER($term) THEN 4.0
#     END AS score
# FROM admin0
# WHERE
#     -- First exact match priority
#     LOWER(ISO) = LOWER($term) OR
#     LOWER(ISO3) = LOWER($term) OR
#     LOWER(fips) = LOWER($term)
# UNION ALL
# -- Then fall back to fuzzy search for anything not exact
# SELECT * FROM (
#     SELECT *, fts_main_admin0.match_bm25(geonameId,
# $term
#     ) AS score
#     FROM admin0
#     WHERE
#         LOWER(ISO) != LOWER($term) AND
#         LOWER(ISO3) != LOWER($term) AND
#         LOWER(fips) != LOWER($term)
# ) sq
# WHERE score IS NOT NULL
# ORDER BY score DESC;
# """,
#         {"term": SEARCH_TERM[0]},
#     )
#     .pl()
#     .lazy()
#     .pipe(country_score)
#     .collect()
# )
# df_con


def search(
    term: str,
    con: DuckDBPyConnection,
    level: Literal[0, 1, 2, 3, 4],
    previous_results: pl.DataFrame | None = None,
) -> pl.DataFrame:
    table_name = f"admin{level}"
    print(f"Searching for '{term}' in '{table_name}' hierarchy")

    if level == 0:
        return (
            con.execute(
                f"""SELECT *,
    -- High fixed score for exact matches
    CASE
        WHEN LOWER(ISO) = LOWER($term) THEN 10.0
        WHEN LOWER(ISO3) = LOWER($term) THEN 8.0
        WHEN LOWER(fips) = LOWER($term) THEN 4.0
    END AS score
FROM admin{level}
WHERE
    -- First exact match priority
    LOWER(ISO) = LOWER($term) OR
    LOWER(ISO3) = LOWER($term) OR
    LOWER(fips) = LOWER($term)
UNION ALL
-- Then fall back to fuzzy search for anything not exact
SELECT * FROM (
    SELECT *, fts_main_admin{level}.match_bm25(geonameId,
$term
    ) AS score
    FROM admin{level}
    WHERE
        LOWER(ISO) != LOWER($term) AND
        LOWER(ISO3) != LOWER($term) AND
        LOWER(fips) != LOWER($term)
) sq
WHERE score IS NOT NULL
ORDER BY score DESC;
""",
                {"term": SEARCH_TERM[0]},
            )
            .pl()
            .lazy()
            .pipe(country_score)
            .collect()
        )

    has_previous = previous_results is not None and not previous_results.is_empty()
    join_type = "LEFT JOIN" if has_previous else "LEFT JOIN"

    with_clause = ""
    join_clause = ""
    parent_filter = ""
    if has_previous:
        con.register("previous_results", previous_results)
        parent_level = level - 1
        parent_code_col = f"admin{parent_level}_code"
        score_col = f"adjusted_score_{parent_level}"

        parent_filter = f"""
        AND a.{parent_code_col} IN (
            SELECT {parent_code_col} FROM previous_results
        ) """

        with_clause = f"""
        WITH parent_scores AS (
            SELECT {parent_code_col}, {score_col}
            FROM previous_results
        )
        """
        join_clause = (
            f"{join_type} parent_scores p ON a.{parent_code_col} = p.{parent_code_col}"
        )

    query = f"""
        {with_clause}
        SELECT a.*
        FROM (
            SELECT *, fts_main_{table_name}.match_bm25(
                geonameId,
                $term
            ) AS score
            FROM {table_name}
        ) a
        {join_clause}
        WHERE score IS NOT NULL
        {parent_filter}
        ORDER BY score DESC;"""

    print(f"Searching {table_name} for '{term}'")
    results = con.execute(query, {"term": term}).pl()

    # If we got no results with filtering, try again without filtering
    if has_previous and results.is_empty() and parent_filter:
        print("No results found with parent filtering. Trying unfiltered search...")
        # Simple query with no filtering
        fallback_query = f"""
        SELECT *
        FROM (
            SELECT *, fts_main_{table_name}.match_bm25(
                geonameId,
                $term
            ) AS score
            FROM {table_name}
        ) a
        WHERE score IS NOT NULL
        ORDER BY score DESC;
        """
        results = con.execute(fallback_query, {"term": term}).pl()

    # Process the results with the appropriate scoring function
    return results.lazy().pipe(admin_score, level).collect()

    return None

    print(f"Searching for '{search_term}' in admin{level} hierarchy")
    if level < 0 or level > 4:
        raise ValueError("Level must be between 0 and 4")
    """Search for a term in the admin hierarchy."""
    if level == 0:
        return (
            con.execute(
                f"""SELECT *,
    -- High fixed score for exact matches
    CASE
        WHEN LOWER(ISO) = LOWER($term) THEN 10.0
        WHEN LOWER(ISO3) = LOWER($term) THEN 8.0
        WHEN LOWER(fips) = LOWER($term) THEN 4.0
    END AS score
FROM admin{level}
WHERE
    -- First exact match priority
    LOWER(ISO) = LOWER($term) OR
    LOWER(ISO3) = LOWER($term) OR
    LOWER(fips) = LOWER($term)
UNION ALL
-- Then fall back to fuzzy search for anything not exact
SELECT * FROM (
    SELECT *, fts_main_admin{level}.match_bm25(geonameId,
$term
    ) AS score
    FROM admin{level}
    WHERE
        LOWER(ISO) != LOWER($term) AND
        LOWER(ISO3) != LOWER($term) AND
        LOWER(fips) != LOWER($term)
) sq
WHERE score IS NOT NULL
ORDER BY score DESC;
""",
                {"term": SEARCH_TERM[0]},
            )
            .pl()
            .lazy()
            .pipe(country_score)
            .collect()
        )

        return (
            con.execute(
                f"""SELECT *
        FROM (
            SELECT *, fts_main_admin{level}.match_bm25(
                geonameId,
                $term
            ) AS score
            FROM admin{level}
        ) sq
        WHERE score IS NOT NULL
        ORDER BY score DESC;
        """,
                {"term": search_term},
            )
            .pl()
            .lazy()
            .pipe(country_score)
            .collect()
        )
    else:
        if previous_search is None:
            raise ValueError("previous_search must be provided for levels 1-4")
        return (
            con.execute(
                f"""SELECT *
        FROM (
            SELECT *, fts_main_admin{level}.match_bm25(
                geonameId,
                $term
            ) AS score
            FROM admin{level}
        ) sq
        WHERE score IS NOT NULL
            AND admin{level - 1}_code IN (
                SELECT admin{level - 1}_code FROM previous_search
            )
        ORDER BY score DESC;
        """,
                {"term": search_term},
            )
            .pl()
            .lazy()
            .pipe(admin_score, level)
            .collect()
        )


a = search(SEARCH_TERM[0], con, 0)
print(a)
b = search(SEARCH_TERM[1], con, 1, a)
print(b)
c = search(SEARCH_TERM[2], con, 2, b)
print(c)
d = search(SEARCH_TERM[3], con, 3, c)
print(d)
e = search(SEARCH_TERM[4], con, 4, d)
e


# In[ ]:


d = search(SEARCH_TERM[3], con, 3, c)
print(d)
e = search(SEARCH_TERM[4], con, 4, d)
e


# In[ ]:


conn.execute(get_parents_query(8354456)).get_as_pl()


# In[ ]:


con.table("allCountries").pl().filter(pl.col("geonameId") == 8354456)


# In[ ]:


conn.execute(get_children_querys([8378478])).get_as_pl()


# In[ ]:


con.table("country").pl().select(pl.col("feature_class").value_counts()).unnest(
    "feature_class"
)


# In[ ]:


con.execute("SELECT geonameId FROM country").pl()


# In[ ]:


query = f"""

CREATE OR REPLACE TABLE

"""


# In[ ]:


df = con.table("allCountries").pl()


# In[ ]:


con.table("area").pl()


# In[ ]:


df


# In[ ]:


df.sample(10)


# In[ ]:


df.filter(pl.col("country_code") == "US").filter(pl.col("admin1_code") == "GA").filter(
    pl.col("name").str.contains("Tabernacle")
)


# In[ ]:


df.filter(pl.col(GID) == 4225563)


# In[ ]:


(
    df.filter(pl.col("admin1_code") == "GA")
    .filter(pl.col("admin2_code") == "279")
    .filter(pl.col("feature_class").is_in(["A", "P"]))
    .filter(pl.col("feature_code").is_in(["ADM1", "ADM2"]))
)


# In[ ]:


(
    df.filter(pl.col("feature_class").is_in(["P", "A"]))
    # .filter(pl.col("name") == "Islington")
    # .filter(pl.col("admin1_code") == "NY")
    .filter(pl.col(GID).is_in([2545, 43702, 124193, 124271, 7147616]))
)


# In[ ]:


con.table("allCountries").pl().filter(pl.col("geonameId") == 3038832)


# In[ ]:


df.filter((pl.col("admin4_code").is_not_null())).sample(10).filter(
    pl.col("feature_class").is_in(["P", "A"])
)


# In[ ]:


df.filter(pl.col("admin3_code") == "77152")


# In[ ]:


conn.execute(get_parents_querys([6556042], traverse=False)).get_as_pl()


# In[ ]:


conn.execute(get_parents_querys([6255148], traverse=False)).get_as_pl()


# In[ ]:


con.table("area").pl()


# In[ ]:





# In[ ]:


country_ids = con.execute("SELECT geonameId FROM country").pl().to_series()
output_df = (
    conn.execute(get_children_querys(country_ids.to_list()))
    .get_as_pl()
    .lazy()
    .unique("geonameId")
    .select("geonameId")
    .collect()
    .to_series()
)
admin1_ids = set(
    con.execute("SELECT geonameId FROM allCountries WHERE feature_code LIKE 'ADM1%'")
    .pl()
    .to_series()
    .append(output_df)
    .unique()
) - set(country_ids)
con.execute(
    f"CREATE OR REPLACE TABLE admin1 AS SELECT * FROM allCountries WHERE allCountries.geonameId IN ({','.join(map(str, admin1_ids))}) ORDER BY geonameId;"
)


# In[ ]:


con.table("country").pl().filter(pl.col("feature_code") == "PCLH")


# In[ ]:


con.table("country").pl().select(pl.col("feature_code").value_counts()).unnest(
    "feature_code"
)


# In[ ]:


admin1_ids = con.execute("select geonameid from admin1").pl().to_series()
output_df = (
    conn.execute(get_children_querys(admin1_ids.to_list()))
    .get_as_pl()
    .lazy()
    .unique("geonameId")
    .select("geonameId")
    .collect()
    .to_series()
)
admin2_ids = (
    set(
        con.execute(
            "SELECT geonameId FROM allCountries WHERE feature_code LIKE 'ADM2%'"
        )
        .pl()
        .to_series()
        .append(output_df)
        .unique()
    )
    - set(admin1_ids)
    - set(country_ids)
)
con.execute(
    f"CREATE OR REPLACE TABLE admin2 AS SELECT * FROM allCountries WHERE allCountries.geonameId IN ({','.join(map(str, admin2_ids))}) ORDER BY geonameId;"
)
con.table("admin2").pl()


# In[ ]:


con.table("admin2").pl()


# In[ ]:


admin2_ids = con.execute("select geonameid from admin2").pl().to_series()
output_df = (
    conn.execute(get_children_querys(admin2_ids.to_list()))
    .get_as_pl()
    .lazy()
    .unique("geonameId")
    .select("geonameId")
    .collect()
    .to_series()
)
admin3_ids = (
    set(
        con.execute(
            "SELECT geonameId FROM allCountries WHERE feature_code LIKE 'ADM3%'"
        )
        .pl()
        .to_series()
        .append(output_df)
        .unique()
    )
    - set(admin2_ids)
    - set(admin1_ids)
    - set(country_ids)
)
con.execute(
    f"CREATE OR REPLACE TABLE admin3 AS SELECT * FROM allCountries WHERE allCountries.geonameId IN ({','.join(map(str, admin3_ids))}) ORDER BY geonameId;"
)
con.table("admin3").pl()


# In[ ]:


con.table("admin3").pl()


# In[ ]:


admin3_ids = con.execute("select geonameid from admin3").pl().to_series()
output_df = (
    conn.execute(get_children_querys(admin3_ids.to_list()))
    .get_as_pl()
    .lazy()
    .unique("geonameId")
    .select("geonameId")
    .collect()
    .to_series()
)
admin4_ids = (
    set(
        con.execute(
            "SELECT geonameId FROM allCountries WHERE feature_code LIKE 'ADM4%'"
        )
        .pl()
        .to_series()
        .append(output_df)
        .unique()
    )
    - set(admin3_ids)
    - set(admin2_ids)
    - set(admin1_ids)
    - set(country_ids)
)
con.execute(
    f"CREATE OR REPLACE TABLE admin4 AS SELECT * FROM allCountries WHERE allCountries.geonameId IN ({','.join(map(str, admin4_ids))}) ORDER BY geonameId;"
)


# In[ ]:


con.table("admin4").pl()


# In[ ]:


admin4_ids = con.execute("select geonameid from admin4").pl().to_series()
output_df = (
    conn.execute(get_children_querys(admin4_ids.to_list()))
    .get_as_pl()
    .lazy()
    .unique("geonameId")
    .select("geonameId")
    .collect()
    .to_series()
)
admin5_ids = (
    set(
        con.execute(
            "SELECT geonameId FROM allCountries WHERE feature_code LIKE 'ADM5%'"
        )
        .pl()
        .to_series()
        .append(output_df)
        .unique()
    )
    - set(admin4_ids)
    - set(admin3_ids)
    - set(admin2_ids)
    - set(admin1_ids)
    - set(country_ids)
)
con.execute(
    f"CREATE OR REPLACE TABLE admin5 AS SELECT allCountries.*, adminCode5.adm5code AS admin5_code FROM allCountries LEFT JOIN adminCode5 ON allCountries.geonameId = adminCode5.geonameId WHERE allCountries.geonameId IN ({','.join(map(str, admin5_ids))}) ORDER BY allCountries.geonameId;"
)


# In[ ]:


con.table("admin5").pl()


# ## Plan
# 
# Some larger teretory one. -> Country -> State -> District -> Subdistrict -> City -> Some generic search field for everything else.
# 
# ### Country, use country table. (Historic filter?)
# 
# - From country info table
# 
# ### Larger teretorys.
# 
# - Get ids from country table, get all parents (recursive?) from country table, remove any duplicates (american samoa to USA) (Could be filters?)
# 
# ### State
# 
# - Get all states (ADM1) (historic filter?) and then get all sub regions of countries? (potential country duplicate? If found in country table, when orchestrating change state found to country?)
# 

# In[ ]:


df = con.table("allCountries").pl().lazy()


# In[ ]:


df.collect()


# In[ ]:


df.sort("modification_date", descending=True).collect()


# In[ ]:


# Write a query to find gids where (gid.a & gid.b) where, name, asciiname, country_code, admin1_code, admin2_code, admin3_code, admin4_code, timezone timezone are the same but feature class are A and P respectively
q = (
    df.filter(pl.col("feature_class") == "A")
    .join(
        df.filter(pl.col("feature_class") == "P"),
        on=[
            "name",
            "asciiname",
            "country_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "timezone",
        ],
        how="inner",
        suffix="_p",
        nulls_equal=True,
    )
    .rename({"geonameId": "geonameId_a"})
    .select("geonameId_a", "geonameId_p")
    .sort("geonameId_a", "geonameId_p")
)
ab = q.collect()


# In[ ]:


# LEFT JOIN P Equivalents to A tables

df.filter(pl.col("feature_class") == "P").collect()


# In[ ]:


con.table("equivalent").pl()


# In[ ]:


my_coordinates = np.array(geocoder.ip("me").latlng, dtype=np.float32)


# In[ ]:


df = con.table("allCountries").pl()


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


vidx.vector_search(my_coordinates1).unwrap().join(df, "geonameId", "left")


# In[ ]:


if (path := Path("./data/processed/latlon.index")).exists():
    print("Loading index...")
    index = Index.restore(path, view=True)
    if index is None:
        raise ValueError("Failed to load index")
else:
    print("Creating index...")
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

    print(f"Visited members: {output.visited_members}")
    print(f"Computed distances: {output.computed_distances}")

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
print(f"{output.computed_distances=}")
print(f"{output.visited_members=}")
df.filter(pl.col("geonameId").is_in(output.keys))


# In[ ]:


output: Matches = index.search(vectors=my_coordinates1, count=10, log=True)
df.filter(pl.col("geonameId").is_in(output.keys))


# In[ ]:


output.distances


# In[ ]:


df.filter(pl.col("feature_code") == "LTER")


# # Country
# 
# feature code is `PCL*` in the `geoname` table.
# 
# #
# 

# In[ ]:


df.filter(pl.col("geonameId") == 15904)


# In[ ]:


hi_q = pl.scan_csv(
    "./data/geonames/hierarchy.txt",
    separator="\t",
    has_header=False,
    schema={
        "from": pl.Int32,
        "to": pl.Int32,
        "type": pl.Utf8,
    },
)


# In[ ]:


hi_df = hi_q.head(500).collect()


# In[ ]:


hi_df


# In[ ]:





# In[ ]:


hi_df = hi_q.collect()


# In[ ]:


hi_df.write_parquet("./data/processed/hierarchy.parquet")


# In[ ]:


hi_df.filter(pl.col("from") == 2635167)


# In[ ]:


df.filter(pl.col("geonameid") == 2635167)


# In[ ]:


gdb = kz.Database("./data/graph_db")

conn = kz.Connection(gdb)


# In[ ]:


conn.execute(
    "CREATE NODE TABLE Entity(geonameid INT32, name STRING, feature_class STRING, feature_code STRING, country_code STRING, population INT64, PRIMARY KEY(geonameid))"
)
conn.execute("CREATE REL TABLE IsIn(FROM Entity TO Entity, type STRING)")


# In[ ]:


conn.execute("Copy Entity FROM './data/processed/geonames.parquet'")


# In[ ]:


conn.execute("Copy IsIn FROM './data/processed/hierarchy.parquet'")


# In[ ]:


res = conn.execute("MATCH (a)-[b]->(c) RETURN *;")
G = res.get_as_networkx(directed=False)


# In[ ]:


import networkx as nx

pageranks = nx.pagerank(G)


# In[ ]:


pageranks


# In[ ]:


data = {"geonameid": list(pageranks.keys()), "pagerank": list(pageranks.values())}
pageranks_df = pl.DataFrame(
    data, schema={"geonameid": pl.Utf8, "pagerank": pl.Float64}
).with_columns(pl.col("geonameid").str.strip_chars_start("Entity_").cast(pl.Int32))
del data


# In[ ]:


pageranks_df.sort("pagerank", descending=True).head(10)


# In[ ]:


df.filter(pl.col("geonameid") == 3169070)


# In[ ]:




