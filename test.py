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
con.install_extension("nanoarrow", repository="community")
con.load_extension("nanoarrow")
con.install_extension("spatial")
con.load_extension("spatial")


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
    file_path: str,
    schema: dict[str, Type[pl.DataType]],
    table_name: str,
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
        con.execute(f"DROP TABLE {table_name}")

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
        if extra_expr is not None:
            q = q.with_columns(extra_expr)

        q = q.with_columns(
            pl.col(pl.Utf8).str.strip_chars().str.strip_chars("\"':").str.strip_chars()
        )
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
        load.from_arrow(df.to_arrow()).create(table_name)
        print(f"Create time: {time() - time_create:.6f}s")
        # load.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        # Time commit
        if GID in schema:
            time_index = time()
            load.execute(f"CREATE INDEX {table_name}_gid ON {table_name} ({GID})")
            print(f"Index time: {time() - time_index:.6f}s")
        load.commit()
        time_commit = time()
        print(f"Commit time: {time() - time_commit:.6f}s")
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


def drop_duplicates(df: pl.LazyFrame) -> pl.LazyFrame:
    col = [
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
        .unique(col, keep="first")
        .filter(~pl.all_horizontal(pl.col(col).is_null()))
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
)

load_file(
    "./data/raw/geonames/adminCode5.txt",
    {
        GID: pl.UInt32,
        "adm5code": pl.Utf8,
    },
    "adminCode5",
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
    extra_expr=cs.by_dtype(pl.Int8).cast(pl.Boolean).fill_null(False),
)
# ISO	ISO3	ISO-Numeric	fips	Country	Capital	Area(in sq km)	Population	Continent	tld	CurrencyCode	CurrencyName	Phone	Postal Code Format	Postal Code Regex	Languages	geonameid	neighbours	EquivalentFipsCode
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
)


def remove_old_ids(df: pl.LazyFrame, con: DuckDBPyConnection) -> pl.LazyFrame:
    ids = con.execute("SELECT geonameId FROM allCountries").pl().unique().to_series()
    return df.filter(pl.col("parentId").is_in(ids) & pl.col("childId").is_in(ids))


load_file(
    "./data/raw/geonames/hierarchy.txt",
    {
        "parentId": pl.UInt32,
        "childId": pl.UInt32,
        "type": pl.Utf8,
    },
    "hierarchy",
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
    skip_rows=1,
)
# # Ignore loading the geo data for now
# if not table_exists(con, "shapes"):
#     con.execute(sql_file("create_table_shapes.sql"))
#     print("Table 'shapes' created")

# # File is corupted atm
# load_file(
#     "./data/raw/geonames/userTags.txt",
#     {
#         gid: pl.Int32,
#         "tag": pl.Utf8,
#     },
#     "userTags",
# )
con.execute("""CREATE OR REPLACE TABLE equivalent AS
SELECT
    A.geonameId AS geonameId_a,
    P.geonameId AS geonameId_p
FROM
    (SELECT * FROM allCountries WHERE feature_class = 'A') AS A
INNER JOIN
    (SELECT * FROM allCountries WHERE feature_class = 'P') AS P
ON
    COALESCE(A.name, 'N/A') = COALESCE(P.name, 'N/A') AND
    COALESCE(A.asciiname, 'N/A') = COALESCE(P.asciiname, 'N/A') AND
    COALESCE(A.admin0_code, 'N/A') = COALESCE(P.admin0_code, 'N/A') AND
    COALESCE(A.admin1_code, 'N/A') = COALESCE(P.admin1_code, 'N/A') AND
    COALESCE(A.admin2_code, 'N/A') = COALESCE(P.admin2_code, 'N/A') AND
    COALESCE(A.admin3_code, 'N/A') = COALESCE(P.admin3_code, 'N/A') AND
    COALESCE(A.admin4_code, 'N/A') = COALESCE(P.admin4_code, 'N/A') AND
    COALESCE(A.timezone, 'N/A') = COALESCE(P.timezone, 'N/A')
ORDER BY
    geonameId_a,
    geonameId_p;""").pl()


# In[8]:


con.table("allCountries").pl()


# In[9]:


# Create country table
con.execute(sql_file("create_table_equivalent.sql")).pl()

con.sql(sql_file("create_table_admin0.sql"))

con.install_extension("fts")
con.load_extension("fts")
con.execute("""
PRAGMA create_fts_index(
    admin0,
    geonameId,
    name,
    asciiname,
    alternatenames,
    Country,
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


# In[10]:


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


# In[11]:


con.table("admin0").pl().rows_by_key("geonameId", named=True)


# In[12]:


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


# In[13]:


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
        # Current + historical codes for each level
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


build_clean_admin_hierarchy(overwrite=True)
create_admin_indexes(con)

# Old hierarchy
# admin1: 4501 entities
# admin2: 47553 entities
# admin3: 178449 entities
# admin4: 224229 entities
# def build_admin_hierarchy(create_views=False, create_fts=True, overwrite=False):
#     """Build the complete administrative hierarchy from admin1 to admin4."""
#     print("Building administrative hierarchy...")

#     # Check if country table exists
#     if not table_exists(con, "country"):
#         print("Error: Country table must exist first")
#         return

#     # Define the hierarchy building process
#     def build_level(level: int, parent_table: str, exclude_tables: list[str]):
#         print(f"\n=== Building admin{level} table ===")

#         # Skip if exists and not overwriting
#         if table_exists(con, f"admin{level}") and not overwrite:
#             print(f"Table admin{level} already exists. Skipping.")
#             return

#         # 1. Get parent IDs
#         parent_ids = con.execute(f"SELECT geonameId FROM {parent_table}").pl()

#         # 2. Get direct children from graph database
#         print(
#             f"Fetching children of {len(parent_ids)} {parent_table} entities from graph database..."
#         )
#         children_df = conn.execute(
#             get_children_querys(parent_ids["geonameId"].to_list())
#         ).get_as_pl()
#         print(f"Found {len(children_df)} children in graph database")

#         # 3. Register as temporary table
#         temp_table = f"temp_children_{level}"
#         con.register(temp_table, children_df)

#         # 4. Build exclusion clause
#         exclusion_clause = " AND ".join(
#             [f"a.geonameId NOT IN (SELECT geonameId FROM {t})" for t in exclude_tables]
#         )

#         # 5. Create the table with SQL
#         print(f"Creating admin{level} table...")
#         con.execute(f"""
#         CREATE OR REPLACE TABLE admin{level} AS
#         SELECT a.*
#         FROM allCountries a
#         WHERE (
#             -- Direct children from graph DB
#             a.geonameId IN (SELECT geonameId FROM {temp_table})
#             -- Feature code pattern matching
#             OR a.feature_code LIKE 'ADM{level}%'
#         )
#         -- Exclude higher admin levels
#         AND {exclusion_clause}
#         ORDER BY a.geonameId
#         """)

#         # 6. Create indexes
#         print(f"Creating index on admin{level}...")
#         con.execute(
#             f"CREATE INDEX IF NOT EXISTS admin{level}_gid ON admin{level} (geonameId)"
#         )

#         # 7. Create views if requested
#         if create_views:
#             print(f"Creating NODES and FTS views for admin{level}...")
#             con.execute(sql_file("create_view_*_NODES.sql", table=f"admin{level}"))
#             con.execute(sql_file("create_view_*_FTS.sql", table=f"admin{level}"))

#         # 8. Create FTS index if requested
#         if create_fts:
#             print(f"Creating FTS index for admin{level}...")
#             con.execute(f"""
#             PRAGMA create_fts_index(
#                 admin{level},
#                 geonameId, name, asciiname, alternatenames, admin{level}_code, stemmer='none', stopwords='none', ignore='(\\.|[^a-z0-9])+', overwrite=1
#             )
#             """)

#         # 9. Report results
#         count = con.execute(f"SELECT COUNT(*) FROM admin{level}").fetchone()[0]  # type: ignore
#         print(f"Completed admin{level} table with {count} entities")

#         return f"admin{level}"

#     # Build each level in sequence
#     admin1 = build_level(1, "country", ["country"])
#     admin2 = build_level(2, "admin1", ["country", "admin1"])
#     admin3 = build_level(3, "admin2", ["country", "admin1", "admin2"])
#     admin4 = build_level(4, "admin3", ["country", "admin1", "admin2", "admin3"])

#     print("\nAdministrative hierarchy successfully built!")

#     # Return counts for all tables
#     for table in ["country", "admin1", "admin2", "admin3", "admin4"]:
#         if table_exists(con, table):
#             count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
#             print(f"{table}: {count} entities")


# # Execute the hierarchy building
# build_admin_hierarchy()


# In[18]:


output_df = (
    conn.execute(
        get_parents_querys(
            con.execute("SELECT geonameId FROM admin0").pl().to_series().to_list(),
            True,
        )
    )
    .get_as_pl()
    .lazy()
    .unique("geonameId")
    .select("geonameId")
    .collect()
    .to_series()
)

id_set = set(output_df) - set(
    con.execute("SELECT geonameId FROM admin0").pl().to_series()
)
query = f"""
CREATE OR REPLACE TABLE area AS
SELECT * FROM allCountries WHERE geonameId IN ({",".join(map(str, id_set))})
ORDER BY geonameId;

CREATE INDEX area_geonameId ON area (geonameId);
"""
print(query)
con.execute(query).pl()


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
                query = f"""
                SELECT a.*
                FROM ({base_query}) a
                WHERE a.{parent_code_col} IN (SELECT {parent_code_col} FROM {temp_table})
                ORDER BY a.score DESC
                LIMIT $limit
                """
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


# In[106]:


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
                SELECT * FROM admin{i}
                {get_where_clause(codes)}
                LIMIT 1
            """).pl()
            if not df.is_empty():
                hierarchy[f"admin{i}"] = df.select("geonameId", "name").row(
                    0, named=True
                )

    return hierarchy


# In[107]:


results = hierarchical_search(
    search_terms=["FR", None, None, None, "Le Lavandou"], con=con
)
results["admin4"].select("geonameId", "name").row(0, named=True)


# In[108]:


results = hierarchical_search(
    search_terms=[None, "FL", None, "Lakeland", None], con=con
)
row = results["admin3"].row(0, named=True)
from pprint import pprint
pprint(row)

pprint(backfill_hierarchy(row, con))


# In[72]:


con.table("admin2").pl().filter(pl.col("admin2_code") == "105")


# In[ ]:


con.table("admin2").pl().filter(
    (pl.col("admin0_code") == "US")
    & (pl.col("admin1_code") == "FL")
    & (pl.col("admin2_code") == "105")
)


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


# In[61]:


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


# In[40]:


d = search(SEARCH_TERM[3], con, 3, c)
print(d)
e = search(SEARCH_TERM[4], con, 4, d)
e


# In[210]:


conn.execute(get_parents_query(8354456)).get_as_pl()


# In[209]:


con.table("allCountries").pl().filter(pl.col("geonameId") == 8354456)


# In[206]:


conn.execute(get_children_querys([8378478])).get_as_pl()


# In[ ]:


con.table("country").pl().select(pl.col("feature_class").value_counts()).unnest(
    "feature_class"
)


# In[22]:


con.execute("SELECT geonameId FROM country").pl()


# In[ ]:


query = f"""

CREATE OR REPLACE TABLE

"""


# In[64]:


df = con.table("allCountries").pl()


# In[20]:


con.table("area").pl()


# In[27]:


df


# In[36]:


df.sample(10)


# In[ ]:


df.filter(pl.col("country_code") == "US").filter(pl.col("admin1_code") == "GA").filter(
    pl.col("name").str.contains("Tabernacle")
)


# In[ ]:


df.filter(pl.col(GID) == 4225563)


# In[41]:


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


# In[25]:


con.table("allCountries").pl().filter(pl.col("geonameId") == 3038832)


# In[ ]:


df.filter((pl.col("admin4_code").is_not_null())).sample(10).filter(
    pl.col("feature_class").is_in(["P", "A"])
)


# In[102]:


df.filter(pl.col("admin3_code") == "77152")


# In[1]:


conn.execute(get_parents_querys([6556042], traverse=False)).get_as_pl()


# In[71]:


conn.execute(get_parents_querys([6255148], traverse=False)).get_as_pl()


# In[83]:


con.table("area").pl()


# In[ ]:





# In[17]:


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


# In[19]:


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


# In[20]:


con.table("admin2").pl()


# In[21]:


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


# In[23]:


con.table("admin3").pl()


# In[24]:


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


# In[29]:


con.table("admin4").pl()


# In[30]:


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


# In[31]:


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

# In[32]:


df = con.table("allCountries").pl().lazy()


# In[33]:


df.collect()


# In[34]:


df.sort("modification_date", descending=True).collect()


# In[36]:


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


# In[37]:


# LEFT JOIN P Equivalents to A tables

df.filter(pl.col("feature_class") == "P").collect()


# In[38]:


con.table("equivalent").pl()


# In[39]:


my_coordinates = np.array(geocoder.ip("me").latlng, dtype=np.float32)


# In[40]:


df = con.table("allCountries").pl()


# In[41]:


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


# In[42]:


my_coordinates1 = np.array([51.549902, -0.121696], dtype=np.float32)
my_coordinates2 = np.array([37.77493, -122.41942], dtype=np.float32)

vidx = VectorIndex("latlon", data, metric="haversine")


# In[43]:


vidx.vector_search(my_coordinates1).unwrap().join(df, "geonameId", "left")


# In[50]:


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


# In[54]:


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


# In[55]:


search_with_distances(index, my_coordinates2, df.lazy())


# In[57]:


output: Matches = index.search(vectors=my_coordinates1, count=10, log=True)
print(f"{output.computed_distances=}")
print(f"{output.visited_members=}")
df.filter(pl.col("geonameId").is_in(output.keys))


# In[59]:


output: Matches = index.search(vectors=my_coordinates1, count=10, log=True)
df.filter(pl.col("geonameId").is_in(output.keys))


# In[60]:


output.distances


# In[61]:


df.filter(pl.col("feature_code") == "LTER")


# # Country
# 
# feature code is `PCL*` in the `geoname` table.
# 
# #
# 

# In[63]:


df.filter(pl.col("geonameId") == 15904)


# In[64]:


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


# In[67]:


hi_df = hi_q.head(500).collect()


# In[66]:


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




