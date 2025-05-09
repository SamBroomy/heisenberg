# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import polars as pl
import polars_distance as pld
from pprint import pprint
from loguru import logger
import polars.selectors as cs
import kuzu as kz
from pathlib import Path
from typing import Type, Callable
from usearch.index import Index, Matches
import numpy as np
from typing import NamedTuple, Self, TypedDict
from functools import partial
from numpy.typing import NDArray
import duckdb
from duckdb import DuckDBPyConnection
from time import time


pl.Config.set_tbl_rows(20)


# %%
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
# con.execute("PRAGMA profiling_mode = 'standard'")  # Set profiling mode
# con.execute("PRAGMA enable_profiling = 'json'")  # Enable profiling
# con.execute("PRAGMA profiling_output = './profile.json'")  # Set profiling output


# %%
con.execute(
    """SELECT * , fts_main_admin_search.match_bm25(geonameId, $term) AS fts_score

        FROM admin_search
        WHERE fts_score IS NOT NULL
            """,
    {"term": "Kenya"},
).pl()


# %%
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


# %%
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


# %%
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


# %%

# %%
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


# %%
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


# %%
def build_unified_admin_table(con, conn=None, overwrite=True):
    """Build a simplified admin_search table focusing on admin codes for hierarchy."""

    logger.debug("Starting simplified admin search table construction...")

    # Check if table exists
    if table_exists(con, "admin_search") and not overwrite:
        logger.debug("Table admin_search already exists. Skipping.")
        con.execute("VACUUM ANALYZE;")
        return

    # Create the table with proper schema
    con.execute(sql_file("create_unified_admin_table.sql"))

    # Define admin level feature code patterns
    level_codes = {
        0: ["PCL", "PCLI", "PCLD", "PCLF", "PCLS", "TERR"],
        1: ["ADM1", "ADM1H"],
        2: ["ADM2", "ADM2H"],
        3: ["ADM3", "ADM3H"],
        4: ["ADM4", "ADM4H"],
    }

    # Process each admin level
    for level in range(0, 5):
        logger.info(f"Processing admin level {level} entities...")

        # Identify entities of this admin level by feature code
        feature_patterns = "', '".join([code for code in level_codes[level]])

        # Direct insert of entities with the matching feature codes
        if level == 0:  # Countries (admin0)
            insert_query = f"""
            INSERT INTO admin_search
            SELECT
                a.geonameId,
                a.name,
                a.asciiname,
                0 AS admin_level,
                a.admin0_code,
                NULL AS admin1_code,
                NULL AS admin2_code,
                NULL AS admin3_code,
                NULL AS admin4_code,
                a.feature_class,
                a.feature_code,
                c.ISO,
                c.ISO3,
                c.ISO_Numeric,
                c.Country AS official_name,
                c.fips,
                a.latitude,
                a.longitude,
                c.population,
                c.area,
                a.alternatenames,
                c.Country AS country_name
            FROM
                allCountries a
            LEFT JOIN
                countryInfo c ON a.geonameId = c.geonameId
            WHERE
                a.feature_code IN ('{feature_patterns}')
                OR a.feature_code LIKE '{level_codes[level][0]}%'
            """
        else:
            # For admin levels 1-4
            insert_query = f"""
            INSERT INTO admin_search
            SELECT
                a.geonameId,
                a.name,
                a.asciiname,
                {level} AS admin_level,
                a.admin0_code,
                {("a.admin1_code" if level >= 1 else "NULL::VARCHAR AS admin1_code")},
                {("a.admin2_code" if level >= 2 else "NULL::VARCHAR AS admin2_code")},
                {("a.admin3_code" if level >= 3 else "NULL::VARCHAR AS admin3_code")},
                {("a.admin4_code" if level >= 4 else "NULL::VARCHAR AS admin4_code")},
                a.feature_class,
                a.feature_code,
                NULL AS ISO,
                NULL AS ISO3,
                NULL AS ISO_Numeric,
                NULL AS official_name,
                NULL AS fips,
                a.latitude,
                a.longitude,
                a.population,
                NULL AS area,
                a.alternatenames,
                c.name AS country_name
            FROM
                allCountries a
            LEFT JOIN
                allCountries c ON a.admin0_code = c.admin0_code AND c.feature_code = 'PCLI'
            WHERE
                (a.feature_code IN ('{feature_patterns}')
                OR a.feature_code LIKE '{level_codes[level][0]}%')
                AND a.admin0_code IS NOT NULL
            """

        # Execute the query to insert data
        con.execute(insert_query)

        # Report count
        count = con.execute(
            f"SELECT COUNT(*) FROM admin_search WHERE admin_level = {level}"
        ).fetchone()[0]
        logger.debug(f"Added {count} entities for admin level {level}")

    # Create FTS index for the unified table
    logger.debug("Creating FTS index for admin_search table...")
    con.execute("""
    PRAGMA create_fts_index(
        admin_search,
        geonameId,
        name, asciiname, alternatenames, official_name, ISO, ISO3,
        stemmer = 'none',
        stopwords = 'none',
        ignore = '(\\.|[^a-z0-9])+',
        overwrite = 1
    )
    """)

    # Final optimization
    logger.debug("Running VACUUM ANALYZE to optimize the database...")
    con.execute("VACUUM ANALYZE;")

    logger.debug("Admin search table construction complete!")
    logger.debug("Writing admin_search table to parquet...")
    con.table("admin_search").pl().write_parquet(
        Path("./data/processed/geonames/admin_search.parquet").as_posix(),
        partition_by=["admin_level"],
    )
    logger.debug("Admin search table written to parquet.")


build_unified_admin_table(con, conn, overwrite=False)


# %%
def build_places_search_table(con, overwrite=True):
    """Build places_search table with balanced importance scoring."""

    logger.debug("Starting places search table construction...")

    if table_exists(con, "places_search") and not overwrite:
        logger.debug("Table places_search already exists. Skipping.")
        return

    # Create the table with physical importance_tier column
    con.execute(sql_file("create_places_search_table.sql"))

    # Define feature categories with more nuanced scoring
    feature_categories = {
        "major_populated": {
            "codes": [
                "PPLA",
                "PPLA2",
                "PPLA3",
                "PPLA4",
                "PPLC",
                "PPLF",
                "PPLG",
                "PPLR",
                "PPLS",
            ],
            "min_population": 0,
            "base_score": 0.6,
            "pop_weight": 0.7,
            "feature_weight": 0.3,
        },
        "landmarks": {
            "codes": [
                "CSTL",
                "MNMT",
                "RUIN",
                "TOWR",
                "ARCH",
                "HSTS",
                "CAVE",
                "ANS",
                "THTR",
                "AMTH",
                "MUS",
                "LIBR",
                "OPRA",
                "PAL",
                "PGDA",
                "TMPL",
                "SHRN",
                "CH",
                "MSQE",
                "SYG",
                "CVNT",
                "MTRO",
                "AIRP",
                "PRT",
                "RSTN",
                "BUSTN",
                "MAR",
            ],
            "min_population": 0,
            "base_score": 0.4,
            "pop_weight": 0.3,
            "feature_weight": 0.7,
        },
        "natural_features": {
            "codes": [
                "MT",
                "PK",
                "PASS",
                "VLC",
                "ISL",
                "BCH",
                "BAY",
                "CAPE",
                "LK",
                "FLLS",
                "CNYN",
                "VAL",
                "DSRT",
                "GLCR",
                "RSV",
            ],
            "min_population": 0,
            "base_score": 0.3,
            "pop_weight": 0.2,
            "feature_weight": 0.8,
        },
        "facilities": {
            "codes": [
                "HTL",
                "RSRT",
                "MALL",
                "MKT",
                "SCH",
                "UNIV",
                "HSP",
                "ZOO",
                "STDM",
                "PRK",
                "RECG",
                "RECR",
                "SPA",
                "ATHF",
                "ASYL",
            ],
            "min_population": 0,
            "base_score": 0.2,
            "pop_weight": 0.5,
            "feature_weight": 0.5,
        },
        "infrastructure": {
            "codes": [
                "BDG",
                "DAM",
                "LOCK",
                "LTHSE",
                "BRKW",
                "PIER",
                "QUAY",
                "PRMN",
                "OILR",
                "PS",
                "PSH",
                "PSN",
                "CTRM",
                "CTRF",
            ],
            "min_population": 0,
            "base_score": 0.15,
            "pop_weight": 0.3,
            "feature_weight": 0.7,
        },
        "government": {
            "codes": [
                "ADMF",
                "GOVL",
                "CTHSE",
                "DIP",
                "BANK",
                "PO",
                "PP",
                "CSTM",
                "SCHC",
                "MILB",
                "INSM",
            ],
            "min_population": 0,
            "base_score": 0.25,
            "pop_weight": 0.4,
            "feature_weight": 0.6,
        },
    }

    # Process each category with improved scoring
    for category, config in feature_categories.items():
        logger.info(f"Processing {category} features...")

        feature_codes = "', '".join(config["codes"])

        calculation = f"""
        {config["base_score"]} +
                    (
                        CASE
                            WHEN a.population > 10000000 THEN 0.4
                            WHEN a.population > 1000000 THEN 0.35
                            WHEN a.population > 100000 THEN 0.3
                            WHEN a.population > 10000 THEN 0.25
                            WHEN a.population > 1000 THEN 0.2
                            WHEN a.population > 100 THEN 0.15
                            WHEN a.population > 0 THEN 0.1
                            ELSE 0.05
                        END * {config["pop_weight"]}
                        +
                        CASE
                            WHEN a.feature_code IN ('PPLC', 'CSTL', 'MNMT') THEN 0.4
                            WHEN a.feature_code IN ('AIRP', 'TOWR', 'MUS', 'RUIN', 'PAL', 'PGDA') THEN 0.35
                            WHEN a.feature_code IN ('UNIV', 'PPLA', 'RSTN', 'MAR', 'HTL') THEN 0.3
                            WHEN a.feature_code IN ('MT', 'PK', 'VLC', 'ISL', 'BCH') THEN 0.25
                            WHEN a.feature_code IN ('CH', 'HSP', 'SCH', 'THTR', 'STDM') THEN 0.2
                            ELSE 0.1
                        END * {config["feature_weight"]}
                        +
                        CASE
                            WHEN LENGTH(a.alternatenames) > 1000 THEN 0.2
                            WHEN LENGTH(a.alternatenames) > 500 THEN 0.15
                            WHEN LENGTH(a.alternatenames) > 100 THEN 0.1
                            WHEN LENGTH(a.alternatenames) > 0 THEN 0.05
                            ELSE 0
                        END * 0.2
                    )"""

        insert_query = f"""
        INSERT INTO places_search
        SELECT
            a.geonameId,
            a.name,
            a.asciiname,
            a.admin0_code,
            a.admin1_code,
            a.admin2_code,
            a.admin3_code,
            a.admin4_code,
            a.feature_class,
            a.feature_code,
            f.name AS feature_name,
            a.latitude,
            a.longitude,
            a.population,
            a.elevation,
            a.alternatenames,
            c.Country AS country_name,
            -- More balanced importance scoring
            {config["base_score"]} +
            (
                -- Population component
                CASE
                    WHEN a.population > 10000000 THEN 0.4
                    WHEN a.population > 1000000 THEN 0.35
                    WHEN a.population > 100000 THEN 0.3
                    WHEN a.population > 10000 THEN 0.25
                    WHEN a.population > 1000 THEN 0.2
                    WHEN a.population > 100 THEN 0.15
                    WHEN a.population > 0 THEN 0.1
                    ELSE 0.05
                END * {config["pop_weight"]}
                +
                -- Feature type component
                CASE
                    -- Capital cities and major landmarks
                    WHEN a.feature_code IN ('PPLC', 'CSTL', 'MNMT') THEN 0.4
                    -- Major tourist destinations
                    WHEN a.feature_code IN ('AIRP', 'TOWR', 'MUS', 'RUIN', 'PAL', 'PGDA') THEN 0.35
                    -- Important facilities
                    WHEN a.feature_code IN ('UNIV', 'PPLA', 'RSTN', 'MAR', 'HTL') THEN 0.3
                    -- Notable natural features
                    WHEN a.feature_code IN ('MT', 'PK', 'VLC', 'ISL', 'BCH') THEN 0.25
                    -- General infrastructure
                    WHEN a.feature_code IN ('CH', 'HSP', 'SCH', 'THTR', 'STDM') THEN 0.2
                    -- Other features
                    ELSE 0.1
                END * {config["feature_weight"]}
                +
                -- Name recognition bonus (if it has many alternate names)
                CASE
                    WHEN LENGTH(a.alternatenames) > 1000 THEN 0.2
                    WHEN LENGTH(a.alternatenames) > 500 THEN 0.15
                    WHEN LENGTH(a.alternatenames) > 100 THEN 0.1
                    WHEN LENGTH(a.alternatenames) > 0 THEN 0.05
                    ELSE 0
                END * 0.2
            ) AS importance_score,
            -- Calculate tier directly during insert
            CASE
                WHEN (
                    {calculation}

                ) >= 0.8 THEN 1  -- Top tier
                WHEN ({calculation}) >= 0.6 THEN 2  -- High importance
                WHEN ({calculation}) >= 0.4 THEN 3  -- Medium importance
                WHEN ({calculation}) >= 0.2 THEN 4  -- Low importance
                ELSE 5  -- Minimal importance
            END AS importance_tier
        FROM
            allCountries a
        LEFT JOIN
            featureCodes f ON a.feature_class || '.' || a.feature_code = f.code
        LEFT JOIN
            countryInfo c ON a.admin0_code = c.ISO
        WHERE
            a.feature_code IN ('{feature_codes}')
            AND a.population >= {config["min_population"]}
            AND a.admin0_code IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM admin_search
                WHERE admin_search.geonameId = a.geonameId
            )
        """
        # Execute the query to insert data and get the number of rows added
        count = con.execute(insert_query).fetchone()[0]
        logger.debug(f"Added {count} {category} features")

    # Add remaining features
    logger.info("Adding remaining features with low importance...")

    processed_codes = []
    for config in feature_categories.values():
        processed_codes.extend(config["codes"])

    insert_remaining_query = f"""
    INSERT INTO places_search
    SELECT
        a.geonameId,
        a.name,
        a.asciiname,
        a.admin0_code,
        a.admin1_code,
        a.admin2_code,
        a.admin3_code,
        a.admin4_code,
        a.feature_class,
        a.feature_code,
        f.name AS feature_name,
        a.latitude,
        a.longitude,
        a.population,
        a.elevation,
        a.alternatenames,
        c.Country AS country_name,
        -- Base score for remaining features
        0.1 +
        CASE
            WHEN a.population > 0 THEN LOG10(a.population) / 20
            ELSE 0
        END +
        CASE
            WHEN LENGTH(a.alternatenames) > 0 THEN 0.05
            ELSE 0
        END AS importance_score,
        -- Calculate tier
        CASE
            WHEN (0.1 + CASE WHEN a.population > 0 THEN LOG10(a.population) / 20 ELSE 0 END) >= 0.2 THEN 4
            ELSE 5
        END AS importance_tier
    FROM
        allCountries a
    LEFT JOIN
        featureCodes f ON a.feature_class || '.' || a.feature_code = f.code
    LEFT JOIN
        countryInfo c ON a.admin0_code = c.ISO
    WHERE
        a.feature_code NOT IN ('{"', '".join(processed_codes)}')
        AND NOT (a.feature_code LIKE 'ADM%' OR a.feature_code LIKE 'PCL%')
        AND a.admin0_code IS NOT NULL
        AND a.feature_class IN ('P', 'S', 'T', 'H', 'L', 'V', 'R')
        AND a.name IS NOT NULL AND a.name != ''
    """

    count = con.execute(insert_remaining_query).fetchone()[0]
    logger.debug(f"Added {count} remaining features with low importance")

    # Create FTS index
    logger.debug("Creating FTS index for places_search table...")
    con.execute("""
    PRAGMA create_fts_index(
        places_search,
        geonameId,
        name, asciiname, alternatenames,
        stemmer = 'none',
        stopwords = 'none',
        ignore = '(\\.|[^a-z0-9])+',
        overwrite = 1
    )
    """)

    # Update statistics
    con.execute("VACUUM ANALYZE;")

    # Show tier distribution
    tier_dist = con.execute("""
        SELECT importance_tier, COUNT(*) as count
        FROM places_search
        GROUP BY importance_tier
        ORDER BY importance_tier
    """).pl()

    logger.info("Importance tier distribution:")
    for row in tier_dist.iter_rows(named=True):
        logger.info(f"  Tier {row['importance_tier']}: {row['count']:,} features")

    logger.debug("Writing places_search table to parquet...")
    con.table("places_search").pl().write_parquet(
        "./data/processed/geonames/places_search.parquet",
        partition_by=["importance_tier"],
    )
    logger.debug("Saved places_search table to parquet file")


build_places_search_table(con, overwrite=False)


# %%
con.close()

con = duckdb.connect(database=duck_db_path.as_posix(), read_only=True)


# %%
# The idea here is now we want to have a more flexible search function.
# As we have done above we have have created one big admin_search table, where the core idea of that is to allow us to search over multiple admin levels at once.
# This will enable us to have two types of searches:
# 1. Search for a specific admin level (e.g. admin1, admin2, etc.) and return results for that level.
#   - This is useful when we want to find specific entities at a certain level.
#   - It will take in a list of exactly length 5 that contains str or None for each admin level.
#   - We can search for the exact level we want and return results for that level.
# 2. Search for a term that is more flexible where you may not know what the exact level is.
#   - This is useful when we want to find entities that match a term but may not know the exact level.
#   - It will take in a list of potentially variable sizes (up to 5) that contains str only. The idea is that its essentially a window function and can sort of map it to the structured input of before.
#      - Lets say we have a flexible the input of [A, B, C] where we are unsure of the level for each of the inputs. What we get is esentially a window function we are able to search over.
#      - That input could be mapped to the structured input of [A, B, C, None, None] or [None, A, B, C, None] or [None, None, A, B, C], [A, None, B, None, C] etc. (and so on).
#      - But we know that for 'A' there are three possible levels for it, so we can search over all of these levels (e.g. admin0, admin1, admin2) and return the results for that level. 'B' could be admin1, admin2, admin3 and so on. The idea is that we can use the number of terms that we have / that aren't None to determine the levels we want to search over.
#   - This means that we need to be able to search over multiple levels at once and return results for all of them.
#   - Once we have the results we can try and filter the next level based on the previous results.
#   - Due to the nature of the flexible search it may not filter down nicely as the structured search, but we can try to filter it down as accurately as possible.


def get_latest_adjusted_score_level(columns: list[str]) -> int | None:
    adjusted_score_columns = [
        col for col in columns if col.startswith("adjusted_score_")
    ]
    if not adjusted_score_columns:
        return None
    # Extract the level from the column name and find the maximum level
    levels = [int(col.rsplit("_", maxsplit=1)[-1]) for col in adjusted_score_columns]
    max_level = max(levels)
    return max_level


def search_score_admin(
    df: pl.LazyFrame,
    level: int,
    text_weight: float = 0.35,
    pop_weight: float = 0.35,
    feature_weight: float = 0.15,
    parent_weight: float = 0.15,
    search_term: str | None = None,
) -> pl.LazyFrame:
    """
    A scoring function for geographic entities that better prioritizes
    significant locations.

    Parameters:
    - df: DataFrame with search results
    - level: Admin level (0=country, 1=admin1, etc.)
    - text_weight: Weight for text matching score
    - pop_weight: Weight for population-based importance
    - feature_weight: Weight for feature type significance
    - parent_weight: Weight for parent entity scores
    - search_term: Original search term (for exact match detection)

    Returns:
    - DataFrame with adjusted scores
    """
    assert level in range(5), "Level must be between 0 and 4"
    score_col = f"adjusted_score_{level}"
    columns = df.collect_schema().names()

    # ===== 1. Text relevance score =====
    fts_column = f"fts_score_{level}"
    if "fts_score" in columns:
        df = df.rename({"fts_score": fts_column})

        df = df.with_columns(
            # Calculate z-score
            z_score=(
                (pl.col(fts_column) - pl.col(fts_column).mean())
                / pl.when(pl.col(fts_column).std() > 0)
                .then(pl.col(fts_column).std())
                .otherwise(1.0)
            ),
        ).with_columns(
            # Apply sigmoid transformation: 1/(1+e^(-z))
            text_score=(1 / (1 + pl.col.z_score.mul(-1.5).exp()))
        )
        if search_term:
            df = df.with_columns(
                text_score=pl.when(
                    pl.col.name.str.to_lowercase() == search_term.lower()
                )
                .then(1)
                .otherwise(pl.col.text_score)
                .clip(0, 1)
            )
    else:
        logger.warning(
            f"Column '{fts_column}' not found in DataFrame. Skipping Z-score normalization."
        )
        df = df.with_columns(text_score=pl.lit(0.5))

    # ===== 2. Population importance - stronger scaling =====
    pop_col = "population"
    if pop_col in columns:
        df = df.with_columns(
            # Sigmoid normalized population factor
            pop_score=pl.when(pl.col(pop_col) > 0)
            .then(
                (
                    # Stronger population scaling using logarithmic curve
                    1 - 1 / (1 + (pl.col(pop_col).log10() / 3))
                )
            )
            .otherwise(0.1)
        )
    else:
        logger.warning(
            f"Column '{pop_col}' not found in DataFrame. Skipping population factor."
        )
        df = df.with_columns(pop_score=pl.lit(0.3))

    # ===== 3. Feature type importance =====
    feature_col = "feature_code"
    if feature_col in columns:
        df = df.with_columns(
            # More nuanced feature type scoring based on importance
            feature_score=pl.when(pl.col(feature_col) == "PCLI")
            .then(1.0)  # Independent countries
            .when(pl.col(feature_col).str.starts_with("PCL"))
            .then(0.9)  # Other country-like entities
            .when(pl.col(feature_col) == "PPLC")
            .then(0.95)  # Capital cities
            .when(pl.col(feature_col).str.starts_with("PPL"))
            .then(0.8)  # Major populated places
            .when(pl.col(feature_col).str.starts_with("ADM1"))
            .then(0.85)  # First-level admin (provinces/states)
            .when(pl.col(feature_col).str.starts_with("ADM2"))
            .then(0.75)  # Second-level admin (counties)
            .when(pl.col(feature_col).str.starts_with("ADM3"))
            .then(0.65)  # Third-level admin (districts)
            .when(pl.col(feature_col).str.starts_with("ADM"))
            .then(0.55)  # Other admin units
            .otherwise(0.5)
        )
    else:
        logger.warning(
            f"Column '{feature_col}' not found in DataFrame. Skipping feature factor."
        )
        df = df.with_columns(feature_score=pl.lit(0.5))

    # ===== 4. Country/region prominence - prioritize major countries =====
    country_col = "admin0_code"
    if country_col in columns:
        # List of major countries to prioritize
        major_countries = [
            "US",
            "GB",
            "DE",
            "FR",
            "JP",
            "CN",
            "IN",
            "BR",
            "RU",
            "CA",
            "AU",
        ]
        df = df.with_columns(
            country_score=pl.when(pl.col(country_col).is_in(major_countries))
            .then(0.8)  # Major countries
            .otherwise(0.5)  # Other countries
        )
    else:
        df = df.with_columns(country_score=pl.lit(0.5))

    parent_score_cols_exist = any(
        col.startswith("parent_adjusted_score_") for col in df.collect_schema().names()
    )

    if parent_score_cols_exist:
        df = df.with_columns(
            # Calculate mean of all parent_adjusted_score_ columns for the row.
            # fill_null(0.0) handles cases where a row has no matching parent scores
            # or a specific linkage didn't yield scores.
            average_parent_score=pl.mean_horizontal(
                cs.starts_with("parent_adjusted_score_")
            ).fill_null(0.0)
        )
        # Normalize this average_parent_score across the entire DataFrame (current batch)
        # Add a small epsilon to avoid division by zero if all average_parent_scores are 0.
        # Max is calculated over a dummy literal column to get the max over the whole frame partition.
        df = (
            df.with_columns(
                parent_max_score_overall=pl.col.average_parent_score.max().over(
                    pl.lit(1)
                )  # Max of averages
            )
            .with_columns(
                parent_factor=pl.when(pl.col.parent_max_score_overall > 1e-9)
                .then(
                    pl.col.average_parent_score
                    / (pl.col.parent_max_score_overall + 1e-9)
                )
                .otherwise(
                    0.5
                )  # Default if all parent scores are zero or no parent scores
                .clip(0.0, 1.0)  # Ensure it's strictly within [0,1]
            )
            .drop("parent_max_score_overall")
        )
    else:
        # This case handles when results_with_potential_parents had no parent_adjusted_score_ columns
        # (e.g., no previous_results or no successful joins in the loop)
        logger.warning(
            "No parent_adjusted_score_ columns found. Skipping parent factor."
        )
        df = df.with_columns(parent_factor=pl.lit(0.5))

    #     if get_latest_adjusted_score_level(columns) is not None:
    #         df = df.with_columns(
    #             average_parent_score=pl.mean_horizontal(cs.starts_with("adjusted_score_"))
    #         ).with_columns(
    #             parent_factor=pl.when(pl.col.average_parent_score > 0)
    #             .then(pl.col.average_parent_score / pl.col.average_parent_score.max())
    #             .otherwise(0.5)
    #         )

    #     else:
    #         logger.warning("No parent score column found. Skipping parent factor.")
    #         df = df.with_columns(parent_factor=pl.lit(0.5))

    # ===== 6. Final score calculation =====
    # Base score calculation
    df = df.with_columns(
        (
            pl.col("text_score").mul(text_weight)
            + pl.col("pop_score").mul(pop_weight)
            + pl.col("feature_score").mul(feature_weight)
            + pl.col("parent_factor").mul(parent_weight)
        ).alias("base_score")
    )

    # Apply country prominence boost to the final score
    df = df.with_columns(
        (pl.col("base_score") * (0.7 + (0.3 * pl.col("country_score")))).alias(
            score_col
        )
    )

    # For debugging, keep all intermediate scores
    return df.sort(score_col, descending=True)


def build_path_conditions(df: pl.DataFrame, admin_cols: list[str]) -> str:
    """
    Build SQL conditions by scanning backwards to find the last non-null value.
    """
    if not admin_cols or df.is_empty():
        return ""

    # Extract relevant columns and filter out all-null rows
    paths_df = (
        df.select(admin_cols).filter(~pl.all_horizontal(pl.all().is_null())).unique()
    )

    path_conditions = []
    for row in paths_df.iter_rows(named=True):
        # Scan backward to find the last non-null column
        last_non_null_idx = -1
        for idx in range(len(admin_cols) - 1, -1, -1):
            if row[admin_cols[idx]] is not None:
                last_non_null_idx = idx
                break

        if last_non_null_idx == -1:
            continue  # Skip rows with all nulls

        # Build conditions up through the last non-null column
        conditions = []
        for idx in range(last_non_null_idx + 1):
            col = admin_cols[idx]
            val = row[col]
            if val is None:
                conditions.append(f"{col} IS NULL")
            else:
                conditions.append(f"{col} = '{val}'")

        path_conditions.append(f"({' AND '.join(conditions)})")

    return " OR ".join(path_conditions)


def search_admin(
    term: str,
    levels: list[int] | int,
    con: DuckDBPyConnection,
    previous_results: pl.DataFrame | None = None,
    limit: int = 100,
    all_cols: bool = False,
) -> pl.DataFrame:
    """
    Search for admin entities across one or multiple admin levels with path-aware filtering.

    Parameters:
    - term: The search term to look for
    - levels: A list of admin levels to search over (0-4) or a single level
    - con: The DuckDB connection object
    - previous_results: Previous search results to filter against
    - limit: The maximum number of results to return

    Returns:
    - A DataFrame with the search results
    """
    # Normalize levels to a list
    if isinstance(levels, int):
        levels = [levels]
    elif not isinstance(levels, list):
        raise ValueError("Levels must be an integer or a list of integers")

    # Validate levels
    if not all(0 <= level <= 4 for level in levels):
        raise ValueError("All levels must be between 0 and 4")

    # Build level constraint
    level_conditions = " OR ".join([f"admin_level = {level}" for level in levels])
    where_clauses = [f"({level_conditions})"]

    # Special handling for country (admin_level = 0) exact matches
    has_country_level = 0 in levels
    country_exact_matches = None

    select_cols_list: list[str] = (
        [
            "geonameId",
            "name",
            "asciiname",
            "admin0_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "feature_class",
            "feature_code",
            "population",
            "latitude",
            "longitude",
        ]
        if not all_cols
        else ["*"]
    )

    if has_country_level and len(term) <= 3:
        # If the term is short (<= 3 characters), we can assume it's a country code
        # First try exact matches for country codes
        exact_match_query = f"""
        SELECT {", ".join(select_cols_list)},
        -- High fixed score for exact matches
        CASE
            WHEN LOWER(ISO) = LOWER($term) THEN 10.0
            WHEN LOWER(ISO3) = LOWER($term) THEN 8.0
            WHEN LOWER(fips) = LOWER($term) THEN 4.0
        END AS fts_score
        FROM admin_search
        WHERE admin_level = 0 AND (
            LOWER(ISO) = LOWER($term) OR
            LOWER(ISO3) = LOWER($term) OR
            LOWER(fips) = LOWER($term)
        )
        """

        country_exact_matches = con.execute(exact_match_query, {"term": term}).pl()

        # If we found exact matches, exclude these from the FTS search
        if not country_exact_matches.is_empty():
            country_ids = country_exact_matches["geonameId"].to_list()
            where_clauses.append(
                f"(admin_level != 0 OR geonameId NOT IN ({','.join(map(str, country_ids))}))"
            )

    admin_cols = []
    # Build path filtering from previous results
    if previous_results is not None and not previous_results.is_empty():
        # Determine which admin code columns to use based on the previous results
        for i in range(5):
            col = f"admin{i}_code"
            if (
                col in previous_results.columns
                and previous_results[col].drop_nulls().shape[0] > 0
            ):
                admin_cols.append(col)

        # Build path conditions using the admin code columns
        if admin_cols:
            path_conditions = build_path_conditions(previous_results, admin_cols)
            if path_conditions:
                where_clauses.append(f"({path_conditions})")

    # Build the WHERE clause
    where_clause = " AND ".join(where_clauses)

    # Build and execute the FTS search
    fts_query = f"""

    WITH filtered_results AS (
        SELECT {",".join(select_cols_list)}, fts_main_admin_search.match_bm25(geonameId, $term) AS fts_score
        FROM admin_search
        WHERE {where_clause}
    )
    -- Using a CTE to ensure we always filter before the FTS score is calculated. Because of the `WHERE fts_score IS NOT NULL` clause, the FTS score will be calculated for all rows, but we only want to keep those that match the search term, hence the subquery first in order to stop the filter push down.
    SELECT * FROM filtered_results
    WHERE fts_score IS NOT NULL
    ORDER BY fts_score DESC
    LIMIT $limit
    """
    logger.debug(f"Executing FTS query: {fts_query}")
    fts_results = con.execute(fts_query, {"term": term, "limit": limit * 2}).pl()

    # Combine exact matches with FTS results if we had exact matches
    if country_exact_matches is not None and not country_exact_matches.is_empty():
        # Ensure both have the same columns
        if not fts_results.is_empty():
            # Dont need this any more.
            # Make sure both have the same columns in the same order
            # all_columns = list(
            #     set(country_exact_matches.columns).union(set(fts_results.columns))
            # )

            # Add any missing columns with None values
            # for col in all_columns:
            #     if col not in country_exact_matches.columns:
            #         country_exact_matches = country_exact_matches.with_columns(
            #             pl.lit(None).alias(col)
            #         )
            #     if col not in fts_results.columns:
            #         fts_results = fts_results.with_columns(pl.lit(None).alias(col))

            # Combine and sort by score
            results = pl.concat(
                [country_exact_matches.lazy(), fts_results.lazy()],
                how="vertical_relaxed",
            )
            results = results.sort("fts_score", descending=True)
        else:
            # If no FTS results, just use exact matches
            results = country_exact_matches.lazy()
    else:
        # Just use FTS results
        results = fts_results.lazy()

    # Trying to get the adjusted scores from the previous results working with the flexible search. The issue is that we need to be able to join the previous results with the current results based on the admin codes. (Tracking the path back is much harder than when doing hierarchical search, as there are potentially multiple paths to the same entity. Unsure how to do this yet. )
    logger.info(admin_cols)
    if previous_results is not None and not previous_results.is_empty():
        previous_scores_df = previous_results.lazy()
        # Rename score columns from previous_results to mark them as parent scores
        parent_score_renames = {
            col: f"parent_{col}"
            for col in previous_results.collect_schema().names()
            if col.startswith("adjusted_score_")
        }
        previous_scores_df = previous_scores_df.rename(parent_score_renames)

        lfs = []
        # Iterate through increasingly specific join key sets
        for i in range(1, len(admin_cols) + 1):
            tmp_cols = admin_cols[
                :i
            ]  # e.g., ["admin0_code"], then ["admin0_code", "admin1_code"]
            # Select join keys and all parent score columns from previous_scores_df
            selected_previous = previous_scores_df.select(
                cs.by_name(tmp_cols), cs.starts_with("parent_adjusted_score_")
            )
            # Join current FTS results with these selected parent scores
            joined_lf = results.join(selected_previous, on=tmp_cols, how="left")
            lfs.append(joined_lf)
        if lfs:
            # Combine all joined versions. Each geonameId might appear multiple times
            # if it could be linked via different paths (different tmp_cols).
            results_with_potential_parents = pl.concat(lfs, how="vertical")
        else:
            # Should not happen if admin_cols is populated, but as a fallback
            results_with_potential_parents = results
            # Ensure results_with_potential_parents has a consistent schema for parent scores,
            # even if they are all nulls here.
            # This might be needed if search_score_admin expects parent_adjusted_score_ columns.
            # However, the modified search_score_admin below handles their absence.

    else:  # No previous_results
        results_with_potential_parents = results

    # Original way that works for hierarchical search but not flexible search. Want to try and get this working for flexible search as well.
    # For now we will just ignore any previous score when doing the flexible search as it complicates things too much.
    # A simple way to work out if we are doing a flexible search is to check the length of the admin_cols and the length of the levels.
    # if (
    #     previous_results is not None and not previous_results.is_empty()
    #     # and 1 == len(levels)
    # ):
    #     # Join with previous results to get adjusted scores
    #     results = results.join(
    #         previous_results.lazy().select(
    #             cs.by_name(admin_cols), cs.starts_with("adjusted_score_")
    #         ),
    #         on=admin_cols,
    #         how="left",
    #     )

    return (
        results_with_potential_parents.pipe(
            search_score_admin, min(levels), search_term=term
        )
        .sort(f"adjusted_score_{min(levels)}", descending=True)
        # Now, for each geonameId, pick the one that got the highest score
        # This effectively selects the "best" parent linkage.
        .unique("geonameId", keep="first", maintain_order=True)
        .head(limit)
        .select(
            # Your existing select logic
            (
                cs.by_name(select_cols_list)
                if select_cols_list != ["*"]
                else cs.all(),  # Ensure 'select' is a list of actual column names
                cs.starts_with(
                    "parent_adjusted_score_"
                ),  # Optionally keep parent scores for debugging
                cs.starts_with("adjusted_score_"),
            )
            if not all_cols
            else cs.all()
        )
        .collect()
    )


# %%
df = con.table("admin_search").pl()


# %%
a = search_admin("The united states of america", [0, 1], con, None, limit=20)


# %%
search_admin("california", [1, 2], con, a, limit=20)


# %%
a = (
    df.filter(pl.col("admin_level") == 2)
    .filter(pl.col.admin0_code.is_in(["US", "UK"]))
    .select(cs.exclude("admin_level"))
    .select(cs.starts_with("admin"))
    .unique()
)
a[[s.name for s in a if not (s.null_count() == a.height)]]


# %%
def search_score_place(
    df: pl.LazyFrame,
    text_weight: float = 0.35,
    importance_weight: float = 0.30,
    feature_weight: float = 0.15,
    distance_weight: float = 0.1,
    parent_admin_score_weight: float = 0.1,
    search_term: str | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> pl.LazyFrame:
    """
    Score places based on multiple factors.

    Parameters:
    - df: DataFrame with search results
    - text_weight: Weight for text matching score
    - importance_weight: Weight for pre-calculated importance
    - feature_weight: Weight for feature type relevance
    - distance_weight: Weight for geographic proximity
    - parent_admin_score_weight: Weight for scores from parent admin entities
    - search_term: Original search term for exact matching
    - center_lat/lon: Center point for distance calculation
    """
    score_col = "place_score"
    columns = df.collect_schema().names()

    # Text relevance score (FTS)
    fts_column = "fts_score"
    if fts_column in columns:
        # Normalize FTS score
        df = df.with_columns(
            z_score=(
                (pl.col(fts_column) - pl.col(fts_column).mean())
                / pl.when(pl.col(fts_column).std() > 0)
                .then(pl.col(fts_column).std())
                .otherwise(1.0)
            ),
        ).with_columns(text_score=(1 / (1 + pl.col.z_score.mul(-1.5).exp())))

        # Exact match bonus
        if search_term:
            df = df.with_columns(
                text_score=pl.when(
                    pl.col.name.str.to_lowercase() == search_term.lower()
                )
                .then(1)
                .when(pl.col.name.str.to_lowercase().str.contains(search_term.lower()))
                .then(pl.col.text_score + 0.25)
                .otherwise(pl.col.text_score)
                .clip(0, 1)
            )
    else:
        df = df.with_columns(text_score=pl.lit(0.5))

    # 2. Importance score (already normalized between 0-1)
    if "importance_score" in columns:
        df = df.with_columns(importance_norm=pl.col("importance_score").clip(0, 1))
    else:
        df = df.with_columns(importance_norm=pl.lit(0.5))

    # Feature type scoring for places
    if "feature_code" in columns:
        df = df.with_columns(
            # Capital/admin centers
            feature_score=pl.when(
                pl.col("feature_code").is_in(
                    ["PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4"]
                )
            )
            .then(1.0)
            # Major landmarks
            .when(pl.col("feature_code").is_in(["CSTL", "MNMT", "RUIN", "TOWR"]))
            .then(0.95)
            # Cultural venues
            .when(pl.col("feature_code").is_in(["MUS", "THTR", "AMTH", "LIBR", "OPRA"]))
            .then(0.9)
            # Populated places
            .when(pl.col("feature_code").is_in(["PPL", "PPLF", "PPLS", "PPLX"]))
            .then(0.85)
            # Transportation hubs
            .when(pl.col("feature_code").is_in(["AIRP", "RSTN", "PRT", "MAR"]))
            .then(0.8)
            # Educational/medical/institutions
            .when(pl.col("feature_code").is_in(["UNIV", "SCH", "HSP", "HTL", "RSRT"]))
            .then(0.75)
            # Commercial
            .when(pl.col("feature_code").is_in(["MALL", "MKT"]))
            .then(0.7)
            # Religious sites
            .when(pl.col("feature_code").is_in(["CH", "MSQE", "TMPL", "SHRN"]))
            .then(0.65)
            # Natural features
            .when(
                pl.col("feature_code").is_in(
                    ["MT", "PK", "VLC", "ISL", "BCH", "LK", "BAY"]
                )
            )
            .then(0.6)
            .otherwise(0.3)
        )
    else:
        df = df.with_columns(feature_score=pl.lit(0.5))

        # 4. Distance score (if center point provided)
    if (
        center_lat is not None
        and center_lon is not None
        and "latitude" in columns
        and "longitude" in columns
    ):
        # Haversine distance calculation
        df = (
            df.with_columns(
                x=pl.struct(latitude="latitude", longitude="longitude"),
                y=pl.struct(
                    latitude=center_lat,
                    longitude=center_lon,
                    schema={
                        "latitude": pl.Float32,
                        "longitude": pl.Float32,
                    },
                ),
            )
            .with_columns(distance_km=pld.col("x").dist.haversine("y", unit="km"))
            .drop("x", "y")
            .with_columns(
                # Convert distance to score (closer = higher score)
                # Using exponential decay: score = e^(-distance/50)
                distance_score=(-pl.col("distance_km") / 50).exp()
            )
        )
    else:
        df = df.with_columns(distance_score=pl.lit(0.5))

    # 5. Parent Admin Score Factor
    parent_score_cols_exist = any(
        col.startswith("parent_adjusted_score_") for col in df.collect_schema().names()
    )
    if parent_score_cols_exist:
        df = df.with_columns(
            # Calculate mean of all parent_adjusted_score_ columns for the row.
            # Assumes parent scores are already normalized (0-1).
            parent_admin_factor=pl.mean_horizontal(
                cs.starts_with("parent_adjusted_score_")
            )
            .fill_null(0.5)
            .clip(0.0, 1.0)
        )
    else:
        df = df.with_columns(parent_admin_factor=pl.lit(0.5))

    df = df.with_columns(
        (
            pl.col("text_score").mul(text_weight)
            + pl.col("importance_norm").mul(importance_weight)
            + pl.col("feature_score").mul(feature_weight)
            + pl.col("distance_score").mul(distance_weight)
            + pl.col("parent_admin_factor").mul(
                parent_admin_score_weight
            )  # Added parent admin score
        ).alias(score_col)
    )
    # 6. Apply tier boost (prioritize higher importance tiers)
    if "importance_tier" in columns:
        df = df.with_columns(
            pl.col(score_col)
            * pl.when(pl.col("importance_tier") == 1)
            .then(1.2)
            .when(pl.col("importance_tier") == 2)
            .then(1.1)
            .when(pl.col("importance_tier") == 3)
            .then(1.0)
            .when(pl.col("importance_tier") == 4)
            .then(0.9)
            .otherwise(0.8)
            .alias(score_col)
        )

    return df.sort(score_col, descending=True)


def search_place(
    term: str,
    con: DuckDBPyConnection,
    previous_results: pl.DataFrame | None = None,
    limit: int = 100,
    min_importance_tier: int = 5,
    center_lat: float | None = None,
    center_lon: float | None = None,
    all_cols: bool = False,
) -> pl.DataFrame:
    """
    Search for places within the places_search table.

    Parameters:
    - term: The search term for the place
    - con: Database connection
    - previous_results: Previous admin search results to filter by
    - limit: Maximum number of results
    - min_importance_tier: Minimum importance tier to include
    - progressive_search: Whether to start with high-importance places first

    Returns:
    - DataFrame with search results
    """
    logger.debug(f"Searching for places with term: {term}")

    select_cols_place: list[str] = (
        [
            "geonameId",
            "name",
            "asciiname",
            "admin0_code",
            "admin1_code",
            "admin2_code",
            "admin3_code",
            "admin4_code",
            "feature_class",
            "feature_code",
            "population",
            "latitude",
            "longitude",
            "importance_score",
            "importance_tier",
        ]
        if not all_cols
        else ["*"]
    )

    where_clauses = []
    join_on_admin_cols = []
    # Build path filtering from previous results
    if previous_results is not None and not previous_results.is_empty():
        # Determine which admin code columns to use based on the previous results
        for i in range(5):
            col = f"admin{i}_code"
            if (
                col in previous_results.columns
                and previous_results[col].drop_nulls().shape[0] > 0
            ):
                join_on_admin_cols.append(col)

        # Build path conditions using the admin code columns
        if join_on_admin_cols:
            path_conditions = build_path_conditions(
                previous_results, join_on_admin_cols
            )
            if path_conditions:
                where_clauses.append(f"({path_conditions})")
    # Add importance tier condition
    where_clauses.append(f"importance_tier <= {min_importance_tier}")

    # Build the WHERE clause
    where_clause = " AND ".join(where_clauses)

    # Extract center point from previous results if not provided
    if center_lat is None and center_lon is None and previous_results is not None:
        if (
            "latitude" in previous_results.columns
            and "longitude" in previous_results.columns
            and not previous_results.select(["latitude", "longitude"]).is_empty()
        ):
            center_data = previous_results.select(
                [
                    pl.mean("latitude").alias("center_lat"),
                    pl.mean("longitude").alias("center_lon"),
                ]
            ).row(0, named=True)
            if (
                center_data["center_lat"] is not None
                and center_data["center_lon"] is not None
            ):
                center_lat, center_lon = (
                    center_data["center_lat"],
                    center_data["center_lon"],
                )
                logger.debug(
                    f"Using center point from previous admin results: ({center_lat}, {center_lon})"
                )

    query = f"""
    WITH filtered_results AS (
        SELECT {",".join(select_cols_place)},
            fts_main_places_search.match_bm25(geonameId, $term) AS fts_score
        FROM places_search
        WHERE {where_clause}
    )
    SELECT * FROM filtered_results
    WHERE fts_score IS NOT NULL
    ORDER BY fts_score DESC,
        importance_score DESC
    LIMIT $limit
    """
    logger.debug(f"Executing FTS query: {query}")
    results_df = con.execute(
        query,
        {
            "term": term,
            "limit": limit * 3,
        },
    ).pl()
    logger.debug(f"Found {results_df.shape[0]} results")
    # Return empty frame if no results
    if results_df.is_empty():
        return results_df

    # Join with parent admin scores if available
    if (
        previous_results is not None
        and not previous_results.is_empty()
        and join_on_admin_cols
    ):
        previous_scores_df = previous_results.lazy()

        parent_score_renames = {
            col: f"parent_{col}"
            for col in previous_results.collect_schema().names()
            if col.startswith("adjusted_score_")
        }

        # Select only join keys and score columns to be renamed
        cols_to_select_from_previous = join_on_admin_cols + list(
            parent_score_renames.keys()
        )
        previous_scores_df = previous_scores_df.select(
            cols_to_select_from_previous
        ).rename(parent_score_renames)

        # Ensure unique admin paths from previous results before join
        previous_scores_df = previous_scores_df.unique(
            subset=join_on_admin_cols, keep="first", maintain_order=False
        )

        logger.debug(
            f"Joining place results with parent admin scores on: {join_on_admin_cols}"
        )
        results_df = (
            results_df.lazy()
            .join(previous_scores_df, on=join_on_admin_cols, how="left")
            .collect()
        )
        logger.debug(f"Shape after joining with parent scores: {results_df.shape}")
    # Score and sort results
    final_results = (
        results_df.lazy()
        .pipe(
            search_score_place,
            search_term=term,
            center_lat=center_lat,
            center_lon=center_lon,
        )
        .sort("place_score", descending=True)
        .head(limit)
    )

    # Define final columns to select
    if not all_cols:
        # Start with the basic place select columns
        final_select_expressions = [cs.by_name(select_cols_place)]
        # Add the main place_score
        final_select_expressions.append(cs.by_name("place_score"))
        # Optionally add parent admin scores for debugging/inspection
        final_select_expressions.append(cs.starts_with("parent_adjusted_score_"))
        # Add any intermediate scoring factors if desired (e.g., text_score, distance_score etc.)
        # final_select_expressions.append(cs.by_name(["text_score", "importance_norm", "feature_score", "distance_score", "parent_admin_factor"]))

        final_results = final_results.select(final_select_expressions)
    else:
        # If all_cols is True, select everything that has been computed
        final_results = final_results.select(cs.all())

    return final_results.collect()


# %%
class AdminHierarchy(NamedTuple):
    admin0: str | None = None
    admin1: str | None = None
    admin2: str | None = None
    admin3: str | None = None
    admin4: str | None = None
    place: str | None = None

    @classmethod
    def from_list(cls, search_terms: list[str | None]) -> Self:
        if len(search_terms) not in [5, 6]:
            raise ValueError("Search terms must be a list of length 5 or 6")
        terms = search_terms + [None] if len(search_terms) == 5 else search_terms
        return cls(*terms)

    def get_admin_values(self) -> list[str | None]:
        return [self.admin0, self.admin1, self.admin2, self.admin3, self.admin4]

    def find_last_non_null_admin_index(self) -> int:
        admin_values = self.get_admin_values()
        return max(
            (i for i, term in enumerate(admin_values) if term is not None), default=-1
        )

    # Removed move_last_admin_to_place as its logic is now integrated into hierarchical_search


class SearchResult(TypedDict, total=False):
    admin0: pl.DataFrame
    admin1: pl.DataFrame
    admin2: pl.DataFrame
    admin3: pl.DataFrame
    admin4: pl.DataFrame
    place: pl.DataFrame


def search_admin_hierarchy(
    search_terms: AdminHierarchy,
    con: DuckDBPyConnection,
    limit: int,
    all_cols: bool,
) -> tuple[SearchResult, pl.DataFrame | None]:
    """
    Search through the admin hierarchy levels.

    Returns:
        Tuple of (results dictionary, last successful results DataFrame)
    """
    results: SearchResult = {}
    last_results: pl.DataFrame | None = None

    for admin_level, term in enumerate(search_terms.get_admin_values()):
        if term is None:
            continue

        logger.debug(f"Searching for term '{term}' at admin level {admin_level}")

        search_results = search_admin(
            term, admin_level, con, last_results, limit, all_cols
        )

        if not search_results.is_empty():
            results[f"admin{admin_level}"] = search_results
            last_results = search_results
        else:
            logger.debug(
                f"No results found for term '{term}' at admin level {admin_level}"
            )

    return results, last_results


def place_as_admin(
    place_term: str,
    admin_level: int,
    con: DuckDBPyConnection,
    last_results: pl.DataFrame | None,
    limit: int,
    all_cols: bool,
) -> pl.DataFrame | None:
    """
    Try searching a place term as an admin level.

    Returns:
        DataFrame of results if successful, None otherwise
    """
    logger.debug(f"Trying place term '{place_term}' as admin level {admin_level}")

    results = search_admin(place_term, admin_level, con, last_results, limit, all_cols)

    if not results.is_empty():
        return results
    else:
        logger.debug(
            f"No results found for place term '{place_term}' at admin level {admin_level}"
        )
        return None


def search_place_with_context(
    place_term: str,
    con: DuckDBPyConnection,
    last_results: pl.DataFrame | None,
    limit: int,
    all_cols: bool = False,
) -> pl.DataFrame | None:
    """
    Search for a place with optional context from previous results.

    Returns:
        DataFrame of results if successful, None otherwise
    """
    logger.debug(f"Searching for place: '{place_term}'")

    place_results = search_place(
        place_term,
        con,
        previous_results=last_results,
        limit=limit,
        all_cols=all_cols,
    )

    if not place_results.is_empty():
        return place_results
    else:
        logger.debug(f"No place results found for '{place_term}'")
        return None


def find_next_null_admin_level(
    search_terms: AdminHierarchy | list[str | None],
) -> int | None:
    """
    Find the position of the first null admin level after the last non-null admin level.

    Args:
        search_terms: Either an AdminHierarchy or a list of search terms where the last element
                     is the place term (optional in list form)

    Returns:
        The index of the first null after the last non-null admin level,
        0 if all admin levels are null, or
        None if there is no null position available

    Examples:
        [None, None, A, None, B, Place] -> None (no null after B)
        [None, None, None, None, None, Place] -> 0 (all admin levels null)
        [A, None, None, None, None, Place] -> 1 (next null after A)
        [A, B, C, None, None, Place] -> 3 (next null after C)
    """
    # Handle AdminHierarchy object
    if isinstance(search_terms, AdminHierarchy):
        admin_terms = [
            search_terms.admin0,
            search_terms.admin1,
            search_terms.admin2,
            search_terms.admin3,
            search_terms.admin4,
        ]
    else:
        # For list input, consider all but the last element if length is 6
        admin_terms = search_terms[:-1] if len(search_terms) == 6 else search_terms
    # If all admin terms are None, return 0
    if all(term is None for term in admin_terms):
        return 0
    # Find the index of the last non-null admin term
    last_non_null_idx = max(
        (i for i, term in enumerate(admin_terms) if term is not None), default=-1
    )
    # Find the first null after the last non-null
    for i in range(last_non_null_idx + 1, len(admin_terms)):
        if admin_terms[i] is None:
            return i
    # If there's no null after the last non-null, return None
    return None


def hierarchical_search(
    search_terms_input: AdminHierarchy,
    con: DuckDBPyConnection,
    limit: int = 20,
    all_cols: bool = False,
    use_last_admin_as_implicit_place_if_none_given: bool = True,
    try_place_candidate_as_admin_in_gap: bool = True,
) -> SearchResult:
    """
    Perform hierarchical geographic search across admin levels.

    Args:
        search_terms_input: AdminHierarchy containing search terms.
        con: Database connection.
        limit: Maximum results to return per level.
        all_cols: Return all columns.
        use_last_admin_as_implicit_place_if_none_given: If True and no explicit place term is given,
            the last non-null admin term will be used as the candidate for place search and admin fallback.
        try_place_candidate_as_admin_in_gap: If True, the place candidate term (explicit or implicit)
            will be searched as an admin entity in the first available null admin slot
            after the explicitly specified admin terms.
    Returns:
        Dictionary mapping level names to search results.
    """

    results: SearchResult = {}
    last_results: pl.DataFrame | None = None
    # --- Step 1: Perform searches for explicitly provided admin terms ---
    admin_terms_from_input = list(search_terms_input.get_admin_values())

    # Create a temporary AdminHierarchy for search_admin_hierarchy, ensuring its .place is None
    # so it only processes the adminX fields from the input.
    temp_hierarchy_for_admin_search = AdminHierarchy(
        admin0=admin_terms_from_input[0],
        admin1=admin_terms_from_input[1],
        admin2=admin_terms_from_input[2],
        admin3=admin_terms_from_input[3],
        admin4=admin_terms_from_input[4],
        place=None,  # Explicitly None for this stage
    )

    # search_admin_hierarchy processes admin0-admin4 from temp_hierarchy_for_admin_search
    results, last_results_context = search_admin_hierarchy(
        temp_hierarchy_for_admin_search, con, limit, all_cols
    )

    # --- Step 2: Determine the term to be used for place-related searches (place_candidate_term) ---
    place_candidate_term = (
        search_terms_input.place
    )  # The explicitly provided place term

    if place_candidate_term is None and use_last_admin_as_implicit_place_if_none_given:
        # Find the last non-null term from the *input* admin levels
        last_admin_idx = search_terms_input.find_last_non_null_admin_index()

        if last_admin_idx != -1:
            place_candidate_term = admin_terms_from_input[last_admin_idx]
            logger.debug(
                f"No explicit place term. Using last admin term '{place_candidate_term}' "
                f"(from input level admin{last_admin_idx}) as the place candidate."
            )
            # Note: The search for this term as an admin entity (at admin{last_admin_idx})
            # has already been performed in Step 1.

    # --- Step 3: If a place_candidate_term exists, optionally try it as an admin entity in a "gap" ---
    if place_candidate_term is not None and try_place_candidate_as_admin_in_gap:
        # A "gap" is a null admin level in the *original input `search_terms_input`* that occurs
        # after the last non-null admin term specified in that input.
        # `find_next_null_admin_level` correctly identifies this.
        gap_admin_level = find_next_null_admin_level(search_terms_input)

        if gap_admin_level is not None and gap_admin_level < 5:
            # We try `place_candidate_term` at `gap_admin_level`.
            # The context for this search is `last_results_context` from Step 1 (or updated if Step 1 had results).
            logger.debug(
                f"Attempting to search place candidate '{place_candidate_term}' "
                f"as admin level {gap_admin_level} (fallback in gap)."
            )
            fallback_admin_results = place_as_admin(
                place_candidate_term,
                gap_admin_level,
                con,
                last_results_context,
                limit,
                all_cols,
            )
            if (
                fallback_admin_results is not None
                and not fallback_admin_results.is_empty()
            ):
                # This result is for the `gap_admin_level`.
                results[f"admin{gap_admin_level}"] = fallback_admin_results
                # This fallback search now becomes the latest context for the final place search.
                last_results_context = fallback_admin_results

    # --- Step 4: Perform the final search for place_candidate_term as a place ---
    if place_candidate_term is not None:
        logger.debug(f"Searching for '{place_candidate_term}' as a place entity.")
        final_place_results = search_place_with_context(
            place_candidate_term, con, last_results_context, limit, all_cols
        )
        if final_place_results is not None and not final_place_results.is_empty():
            results["place"] = final_place_results
            # Optionally update context if further steps needed it:
            # last_results_context = final_place_results

    return results


def flexible_search(
    search_terms_raw: list[str],
    con: DuckDBPyConnection,
    limit: int = 20,
    all_cols: bool = False,
    try_place_candidate_as_admin_fallback_on_fail: bool = True,
) -> list[pl.DataFrame]:
    """
    Perform flexible geographic search across admin levels.

    Args:
        search_terms_raw: List of search terms.
        con: Database connection.
        limit: Maximum results to return per level.
        all_cols: Return all columns.
        try_place_candidate_as_admin_fallback_on_fail: If True and the place candidate term
            search (as a place) fails, try searching it as an admin entity in subsequent levels.
    Returns:
        List of DataFrames with search results.
    """

    # --- Input Cleaning & Term Definition ---
    cleaned_terms = [
        term for term in search_terms_raw if term is not None and term.strip()
    ]
    if not cleaned_terms:
        # Return empty list or raise error based on desired behavior for no valid terms
        logger.warning("No valid search terms provided after cleaning.")
        return []
    if len(cleaned_terms) > 6:
        raise ValueError("Search terms must be a list of length <= 6 after cleaning.")

    admin_terms_for_flex_search: list[str]
    place_candidate_term: str
    # is_place_term_exclusive: True if the place_candidate_term was *only* for place search (input had 6 terms).
    is_place_term_exclusive: bool

    if len(cleaned_terms) == 6:
        admin_terms_for_flex_search = cleaned_terms[:-1]
        place_candidate_term = cleaned_terms[-1]
        is_place_term_exclusive = True
    else:  # 1 to 5 terms
        admin_terms_for_flex_search = list(
            cleaned_terms
        )  # All terms are part of admin sequence
        place_candidate_term = cleaned_terms[
            -1
        ]  # The last of these is also the place candidate
        is_place_term_exclusive = False

    logger.debug(f"Flexible search: Admin terms: {admin_terms_for_flex_search}")
    logger.debug(
        f"Flexible search: Place candidate: '{place_candidate_term}', Exclusive: {is_place_term_exclusive}"
    )

    all_found_results_list: list[pl.DataFrame] = []
    last_successful_admin_context: pl.DataFrame | None = None

    # --- Step 1: Iterative Flexible Admin Search ---
    num_actual_admin_terms = len(admin_terms_for_flex_search)
    # empty_admin_slots determines how many levels a term can "slide" over.
    # If 1 admin term, it can be level 0-4 (empty_admin_slots = 4).
    # If 5 admin terms, each term maps to one level (empty_admin_slots = 0).
    empty_admin_slots = 5 - num_actual_admin_terms

    for i, term_to_search in enumerate(admin_terms_for_flex_search):
        start_level = i
        # The term at index `i` can occupy levels from `i` up to `i + empty_admin_slots`.
        # Max admin level is 4.
        end_level = min(4, i + empty_admin_slots)

        current_search_levels = list(range(start_level, end_level + 1))

        if not current_search_levels:  # Should ideally not happen with correct logic
            logger.warning(
                f"Term '{term_to_search}': No valid admin levels to search (calculated range: {start_level}-{end_level}). Skipping."
            )
            continue

        logger.debug(
            f"Flex-searching admin term '{term_to_search}' for levels {current_search_levels}."
        )

        term_admin_results = search_admin(
            term_to_search,
            current_search_levels,
            con,
            last_successful_admin_context,
            limit,
            all_cols,
        )

        if not term_admin_results.is_empty():
            all_found_results_list.append(term_admin_results)
            last_successful_admin_context = term_admin_results
        else:
            logger.debug(
                f"No admin results for term '{term_to_search}' in levels {current_search_levels}."
            )
            # Context (last_successful_admin_context) remains from the previous successful search.

    # --- Step 2: Search for Place Candidate as a Place ---
    logger.debug(f"Searching for '{place_candidate_term}' as a place entity.")
    place_search_results = search_place_with_context(
        place_candidate_term, con, last_successful_admin_context, limit, all_cols
    )

    place_found_successfully = (
        place_search_results is not None and not place_search_results.is_empty()
    )

    if place_found_successfully:
        all_found_results_list.append(place_search_results)  # type: ignore[arg-type]
    else:
        logger.debug(f"No place results for '{place_candidate_term}'.")

        # --- Step 3: Fallback - Try Place Candidate as Admin if Place Search Failed ---
        if try_place_candidate_as_admin_fallback_on_fail:
            # Determine levels for this fallback. These should be levels *after* those
            # notionally covered by `admin_terms_for_flex_search`.
            # `num_actual_admin_terms` is the count of terms in `admin_terms_for_flex_search`.
            # So, the next available admin slot starts at index `num_actual_admin_terms`.
            fallback_start_level = num_actual_admin_terms

            if (
                fallback_start_level < 5
            ):  # Max admin level is 4. If fallback_start_level is 5, no slots.
                admin_fallback_levels = list(
                    range(fallback_start_level, 5)
                )  # e.g., if 3 admin terms, try levels 3, 4.

                if admin_fallback_levels:
                    logger.debug(
                        f"Fallback: Trying place candidate '{place_candidate_term}' as admin "
                        f"at levels {admin_fallback_levels}."
                    )
                    # Context for this fallback is still `last_successful_admin_context` from Step 1.
                    admin_fallback_data = place_as_admin(
                        place_candidate_term,
                        admin_fallback_levels,
                        con,
                        last_successful_admin_context,
                        limit,
                        all_cols,
                    )
                    if (
                        admin_fallback_data is not None
                        and not admin_fallback_data.is_empty()
                    ):
                        all_found_results_list.append(admin_fallback_data)

    return all_found_results_list


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
        print(i)
        code = row.get(f"admin{i}_code")
        codes.append(code)
        if code is not None:
            df = con.execute(
                f"SELECT geonameId, name FROM admin{i} {get_where_clause(codes)} LIMIT 1"
            ).pl()
            if not df.is_empty():
                hierarchy[f"admin{i}"] = df.to_dicts()[0]
    return hierarchy


# Hierarchical search with place
st = ["US", "CA", "Los Angeles County", "Beverly Hills", None, None]
search_terms = AdminHierarchy.from_list(st)
results = hierarchical_search(search_terms, con)
results.get("admin3")


# %%
# Define a configuration structure for the smart flexible search
class SmartFlexibleSearchConfig(TypedDict, total=False):
    limit: int  # Max results per search stage
    all_cols: bool  # Whether to return all columns from underlying tables

    # Max number of terms from input to treat as a sequence of admin entities.
    # If len(input_terms) > max_sequential_admin_terms, the term at
    # index `max_sequential_admin_terms` becomes an exclusive place candidate.
    # If len(input_terms) <= max_sequential_admin_terms, the *last* term of the input
    # serves as both the end of the admin sequence AND the place candidate.
    max_sequential_admin_terms: int

    # If True, the place_candidate_term will be proactively searched as an admin entity
    # in levels *after* the conceptual slots filled by `admin_terms_for_main_sequence`,
    # but *before* it's searched as a place. This applies if the place_candidate_term
    # was not already the last term of the admin_terms_for_main_sequence.
    attempt_place_candidate_as_admin_before_place_search: bool


MAX_ADMIN_LEVELS_COUNT = 5  # Represents 5 levels: 0, 1, 2, 3, 4


def smart_flexible_search(
    search_terms_raw: list[str],
    con: DuckDBPyConnection,
    config: SmartFlexibleSearchConfig | None = None,
) -> list[pl.DataFrame]:
    """
    Perform a smart, flexible geographic search across admin levels and for places.
    The user provides an ordered list of terms, and the function attempts to match
    them sequentially and flexibly.
    """

    cfg: SmartFlexibleSearchConfig = {
        "limit": 20,
        "all_cols": False,
        "max_sequential_admin_terms": 5,
        "attempt_place_candidate_as_admin_before_place_search": True,
    }
    if config:
        cfg.update(config)  # type: ignore

    # --- 1. Input Cleaning & Term Definition ---
    cleaned_terms = [term for term in search_terms_raw if term and term.strip()]
    if not cleaned_terms:
        logger.warning("No valid search terms provided after cleaning.")
        return []

    admin_terms_for_main_sequence: list[str]
    place_candidate_term: str | None = None

    is_place_candidate_also_last_admin_term_in_sequence: bool = False

    num_cleaned_terms = len(cleaned_terms)
    # Determine how many input terms form the primary administrative sequence
    num_admin_terms_in_sequence = min(
        num_cleaned_terms, cfg["max_sequential_admin_terms"]
    )
    admin_terms_for_main_sequence = cleaned_terms[:num_admin_terms_in_sequence]

    if num_cleaned_terms > num_admin_terms_in_sequence:
        place_candidate_term = cleaned_terms[num_admin_terms_in_sequence]
        if num_cleaned_terms > num_admin_terms_in_sequence + 1:
            extra_terms = " ".join(cleaned_terms[num_admin_terms_in_sequence + 1 :])
            place_candidate_term = f"{place_candidate_term} {extra_terms}"
            logger.info(
                f"Concatenated extra input terms into place candidate. New place candidate: '{place_candidate_term}'"
            )
    elif admin_terms_for_main_sequence:
        place_candidate_term = admin_terms_for_main_sequence[-1]
        is_place_candidate_also_last_admin_term_in_sequence = True

    logger.debug(
        f"Smart Flexible Search: Admin terms for main sequence: {admin_terms_for_main_sequence}"
    )
    if place_candidate_term:
        logger.debug(
            f"Smart Flexible Search: Place candidate term: '{place_candidate_term}' (Is also last admin in sequence: {is_place_candidate_also_last_admin_term_in_sequence})"
        )
    else:
        logger.debug(
            "Smart Flexible Search: No distinct place candidate term identified."
        )

    all_found_results_list: list[pl.DataFrame] = []
    last_successful_context: pl.DataFrame | None = None

    # --- 2. Iterative Flexible Admin Search (Main Sequence) ---
    if admin_terms_for_main_sequence:
        num_terms_in_main_seq = len(admin_terms_for_main_sequence)
        empty_admin_slots = MAX_ADMIN_LEVELS_COUNT - num_terms_in_main_seq
        min_level_from_last_success: int | None = None

        for i, term_to_search in enumerate(admin_terms_for_main_sequence):
            # Determine the search window for the current admin term
            effective_start_level = i  # Current term's natural starting slot
            if min_level_from_last_success is not None:
                # Next term must be at least one level deeper than where the previous term was found
                # and cannot start shallower than its natural slot 'i'.
                effective_start_level = max(i, min_level_from_last_success + 1)

            # The end level for the current term's search window is based on its slot and available empty_admin_slots
            current_search_window_end_level = min(
                MAX_ADMIN_LEVELS_COUNT - 1, i + empty_admin_slots
            )

            if effective_start_level > current_search_window_end_level:
                current_search_levels = []  # No valid levels for this term
            else:
                current_search_levels = list(
                    range(effective_start_level, current_search_window_end_level + 1)
                )

            if not current_search_levels:
                logger.warning(
                    f"Term '{term_to_search}': No valid admin levels to search (effective range: {effective_start_level}-{current_search_window_end_level}). Skipping."
                )
                continue

            logger.debug(
                f"Searching main admin sequence term '{term_to_search}' for admin levels {current_search_levels} with current context."
            )

            term_admin_results = search_admin(
                term=term_to_search,
                levels=current_search_levels,
                con=con,
                previous_results=last_successful_context,
                limit=cfg["limit"],
                all_cols=cfg["all_cols"],
            )

            if not term_admin_results.is_empty():
                logger.info(
                    f"Found {len(term_admin_results)} results for main admin term '{term_to_search}' in levels {current_search_levels}."
                )
                all_found_results_list.append(term_admin_results)
                last_successful_context = term_admin_results
                if "admin_level" in term_admin_results.columns:
                    min_level_from_last_success = term_admin_results[
                        "admin_level"
                    ].min()
                    # min() is used to ensure the next term is strictly deeper than the shallowest level found.
                else:
                    min_level_from_last_success = None
            else:
                logger.debug(
                    f"No results for main admin term '{term_to_search}' in levels {current_search_levels}."
                )
                # min_level_from_last_success retains its value from the *actual* last successful search.
    # --- 3. Proactive Admin Search for Place Candidate (if applicable) ---
    should_run_proactive_admin_search = (
        place_candidate_term is not None
        and cfg["attempt_place_candidate_as_admin_before_place_search"]
        and not is_place_candidate_also_last_admin_term_in_sequence
    )

    if should_run_proactive_admin_search:
        # Determine admin levels for this proactive search: levels after the main admin sequence.
        additional_admin_start_level = len(admin_terms_for_main_sequence)

        if additional_admin_start_level < MAX_ADMIN_LEVELS_COUNT:
            additional_admin_search_levels = list(
                range(additional_admin_start_level, MAX_ADMIN_LEVELS_COUNT)
            )

            if additional_admin_search_levels:
                logger.debug(
                    f"Proactively searching place candidate '{place_candidate_term}' as ADMIN "
                    f"at levels {additional_admin_search_levels} using current context."
                )

                proactive_admin_results = search_admin(
                    term=place_candidate_term,  # Safe: checked place_candidate_term is not None above
                    levels=additional_admin_search_levels,
                    con=con,
                    previous_results=last_successful_context,
                    limit=cfg["limit"],
                    all_cols=cfg["all_cols"],
                )
                if not proactive_admin_results.is_empty():
                    logger.info(
                        f"Found {len(proactive_admin_results)} results for place candidate '{place_candidate_term}' as proactive ADMIN in levels {additional_admin_search_levels}."
                    )
                    all_found_results_list.append(proactive_admin_results)
                    last_successful_context = proactive_admin_results
                else:
                    logger.debug(
                        f"No results for place candidate '{place_candidate_term}' as proactive ADMIN in levels {additional_admin_search_levels}."
                    )
        else:
            logger.debug(
                f"Skipping proactive admin search for '{place_candidate_term}': no subsequent admin levels available (start_level: {additional_admin_start_level})."
            )
    elif (
        place_candidate_term
        and cfg["attempt_place_candidate_as_admin_before_place_search"]
        and is_place_candidate_also_last_admin_term_in_sequence
    ):
        logger.debug(
            f"Skipping proactive admin search for '{place_candidate_term}': it was already processed as the last admin term in the main sequence."
        )

    # --- 4. Final Place Search (if a place_candidate_term exists) ---
    if place_candidate_term:
        logger.debug(
            f"Searching for place candidate '{place_candidate_term}' as PLACE entity using final context."
        )

        final_place_results_df = search_place(
            term=place_candidate_term,
            con=con,
            previous_results=last_successful_context,
            limit=cfg["limit"],
            all_cols=cfg["all_cols"],
        )

        if not final_place_results_df.is_empty():
            logger.info(
                f"Found {len(final_place_results_df)} results for place candidate '{place_candidate_term}' as PLACE."
            )
            all_found_results_list.append(final_place_results_df)
        else:
            logger.debug(
                f"No results for place candidate '{place_candidate_term}' as PLACE."
            )

    logger.info(
        f"Smart flexible search finished. Returning {len(all_found_results_list)} DataFrame(s)."
    )
    return all_found_results_list


# %%
cfg_example = {
    "limit": 5,
    "max_sequential_admin_terms": 5,
    "try_place_candidate_as_admin_fallback_on_place_search_fail": True,
}
results_list = smart_flexible_search(
    ["UK", "London", "Camden", "British Museum"], con, config=cfg_example
)
for i, df_result in enumerate(results_list):
    print(f"\n--- Results from Stage {i + 1} ---")
    print(df_result)

results_short = smart_flexible_search(["FL", "Lakeland"], con, config={"limit": 3})
for i, df_result in enumerate(results_short):
    print(f"\n--- Results from Stage {i + 1} (Short Input) ---")
    print(df_result)


# %%
con.execute(
    """WITH filtered_results AS (
        SELECT geonameId,name,asciiname,admin0_code,admin1_code,admin2_code,admin3_code,admin4_code,feature_class,feature_code,population,latitude,longitude,importance_score,importance_tier,
            fts_main_places_search.match_bm25(geonameId, $term) AS fts_score
        FROM places_search
        WHERE ((admin0_code = 'US' AND admin1_code = 'FL' AND admin2_code = '105' AND admin3_code = '7170309')) AND importance_tier <= 5
    )
    SELECT * FROM filtered_results
    WHERE fts_score IS NOT NULL
    ORDER BY fts_score DESC,
        importance_score DESC
    LIMIT $limit""",
    {"term": "Lakeland", "limit": 10},
).pl()


# %%
results_list[3]


# %%
results_short[1]


# %%
results_short[2]


# %%
results = flexible_search(["US", "Los Angeles County", "Beverly Hills"], con)
results


# %%
results[3]


# %%
a = (
    results[2]
    .select(cs.exclude("admin_level"))
    .select(cs.starts_with("admin"))
    .unique()
)
join_cols = a[[s.name for s in a if not (s.null_count() == a.height)]].columns
print(join_cols)
b = (
    con.table("admin_search")
    .pl()
    .lazy()
    .join(results[2].lazy().select(join_cols), on=join_cols)
    .filter(
        # These terms are esentially the same thing
        pl.col("admin_level").is_in([3, 4])
    )
)


# %%
# Flexible search with potential place
flex_terms = ["United Kingdom", "London", "Westminster", "Parlement"]
flex_results = flexible_search(flex_terms, con)

flex_results[2]


# %%
st = ["FR", "Provence-Alpes-Côte d'Azur", None, None, "Le Lavandou", None]
search_terms = AdminHierarchy.from_list(st)
# Search through the admin hierarchy
results = hierarchical_search(
    search_terms,
    con=con,
    all_cols=True,
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
results["admin4"]


# %%
results["place"]


# %%
s = flexible_search(st, con=con, limit=10)

d = s[1]

admin_cols = sorted(
    [c for c in d.columns if c.startswith("admin") and c.endswith("_code")]
)
admin_cols


# %%
r = search_admin("England", [0, 1], con)
# First find the administrative region
admin_results = search_admin("Dover", [3, 4], con, r)
admin_results


# %%
# Then search for places within that region
place_results = search_place(
    "Dover Ferry Terminal",
    con,
    previous_results=admin_results,
    limit=50,
)

place_results


# %%
r = search_admin("England", [0, 1], con)
a = search_admin("Islington", [1, 2, 3], con, r)
b = search_place(
    "Caledonian Road",
    con,
    previous_results=a,
    limit=50,
)
b


# %%
backfill_hierarchy(
    {
        "admin0_code": "GB",
        "admin1_code": "ENG",
        "admin2_code": "GLA",
        "admin3_code": "G3",
        "geonameId": 13269818,
    },
    con,
)


# %%
results = hierarchical_search(
    search_terms=AdminHierarchy.from_list([None, "FL", None, "Lakeland", None, None]),
    con=con,
    try_place_as_admin=False,
)
row = results["admin3"].row(0, named=True)


pprint(row)

pprint(backfill_hierarchy(row, con))


# %%
# Search through the admin hierarchy
results = hierarchical_search(
    search_terms=AdminHierarchy.from_list(
        [
            "FR",
            "Provence-Alpes-Côte d'Azur",
            "Var",
            "Arrondissement de Toulon",
            "Le Lavandou",
            None,
        ]
    ),
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


# %%
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


# %%
my_coordinates1 = np.array([51.549902, -0.121696], dtype=np.float32)
my_coordinates2 = np.array([37.77493, -122.41942], dtype=np.float32)

vidx = VectorIndex("latlon", data, metric="haversine")


# %%
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


# %%
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


# %%
search_with_distances(index, my_coordinates2, df.lazy())


# %%
output: Matches = index.search(vectors=my_coordinates1, count=10, log=True)
logger.debug(f"{output.computed_distances=}")
logger.debug(f"{output.visited_members=}")
df.filter(pl.col("geonameId").is_in(output.keys))


# %%
# con.execute(sql_file("create_view_*_NODES.sql", table="admin0"))

# con.execute(sql_file("create_view_*_FTS.sql", table="admin0"))

# # if (path := Path("./data/processed/latlon.index")).exists():
# #     logger.debug("Loading index...")
# #     index = Index.restore(path, view=True) or raise ValueError("Failed to load index")
# # else:
# #     logger.debug("Creating index...")
# #     coordinates = df.select(["latitude", "longitude"]).to_numpy(order="c")
# #     labels = df["geonameid"].to_numpy()
# #     index: Index = Index(ndim=2, metric="haversine", dtype="f32")
# #     index.add(keys=labels, vectors=coordinates, log=True)
# #     index.save(path)


# class VectorIndex:
#     default_index_path = Path("./data/indexes/vector")

#     def __init__(
#         self,
#         index_name: str,
#         data: pl.DataFrame | None = None,
#         id_column: str = "geonameId",
#         main_column: str = "vectors",
#         metric: str = "L2",
#         embedder: SentenceTransformer | None = None,
#     ):
#         self._index_path = self.default_index_path / f"{index_name}.index"
#         self._id_column = id_column
#         self._main_column = main_column
#         self._metric = metric
#         index = self.get_or_build_index(data, metric)
#         if isinstance(index, Err):
#             logger.debug(
#                 f"Index does not exist at '{self.index_path}', build index with 'build_index' method."
#             )
#             self._index = None  # type: ignore
#         else:
#             self._index: Index = index.ok_value

#     @property
#     def index(self) -> Index:
#         return self._index

#     @property
#     def id_column(self) -> str:
#         return self._id_column

#     @property
#     def main_column(self) -> str:
#         return self._main_column

#     @property
#     def index_path(self) -> Path:
#         return self._index_path

#     @property
#     def ndims(self) -> int:
#         return self._ndims

#     @property
#     def metric(self) -> str:
#         return self._metric

#     def _build_index(
#         self,
#         df: pl.DataFrame,
#         metric: str = "L2",  # TODO: Metric like
#     ) -> Result[Index, str]:
#         """Data passed should be an Id and a vector."""
#         logger.debug("Creating index...")
#         vectors = df[self.main_column].to_numpy()
#         labels = df[self.id_column].to_numpy()
#         ndims = vectors.shape[1]  # Find n dims
#         index: Index = Index(ndim=ndims, metric=metric, dtype="f32")
#         index.add(keys=labels, vectors=vectors, log=True)
#         index.save(self.index_path)
#         return Ok(index)

#     def get_index(self) -> Result[Index, str]:
#         if (path := self.index_path).exists():
#             logger.debug(f"Opening index at '{self.index_path}'")
#             index = Index.restore(path, view=True)
#             if index is not None:
#                 return Ok(index)
#         return Err(f"Index does not exist at '{self.index_path}'")

#     def get_or_build_index(
#         self,
#         df: pl.DataFrame | None = None,
#         metric: str = "L2",  # TODO: as above
#     ) -> Result[Index, str]:
#         self.index_path.parent.mkdir(parents=True, exist_ok=True)

#         if not self.index_path.exists():
#             if df is None:
#                 return Err(
#                     "Index does not exist. DataFrame is required to create index"
#                 )
#             match self._build_index(df, metric):
#                 case Ok(index):
#                     ...
#                 case Err(e):
#                     return Err(e)
#         else:
#             match self.get_index():
#                 case Ok(index):
#                     ...
#                 case Err(e):
#                     return Err(e)

#         self._ndims = index.ndim
#         logger.debug("Opening index")
#         return Ok(index)

#     def search(
#         self,
#         query: NDArray[np.float32],
#         limit: int = 10,
#         include: list[int] | None = None,
#         exclude: list[int] | None = None,
#     ) -> Result[pl.DataFrame, str]:
#         return self.vector_search(query, limit, include, exclude)

#     def vector_search(
#         self,
#         query: NDArray[np.float32],
#         limit: int = 10,
#         include: list[int] | None = None,
#         exclude: list[int] | None = None,
#         exact: bool = False,
#     ) -> Result[pl.DataFrame, str]:
#         output = self.index.search(vectors=query, count=limit, log=True, exact=exact)

#         logger.debug(f"Visited members: {output.visited_members}")
#         logger.debug(f"Computed distances: {output.computed_distances}")

#         # Extract keys (geonameids) and distances
#         keys = output.keys
#         distances = output.distances

#         # Create a DataFrame from the search results
#         results_df = pl.LazyFrame(
#             data={self.id_column: keys, "score": distances},
#             schema={self.id_column: pl.UInt32, "score": pl.Float32},
#         )
#         if self.metric == "haversine":
#             results_df = results_df.with_columns(pl.col("score") * 6371.0)

#         results_df = results_df.sort(
#             "score"
#         )  # TODO: ascending descending depending on metric.

#         return Ok(results_df.collect())


# class FTSIndex:
#     default_index_path = Path("./data/indexes/fts")

#     def __init__(
#         self,
#         index_name: str,
#         data: pl.DataFrame | None = None,
#         id_column: str = "geonameId",
#         main_column: str = "name",
#     ):
#         self._index_path = self.default_index_path / index_name
#         self._column_types = {}
#         self._id_column = id_column
#         self._main_column = main_column
#         index = self.get_or_build_index(data)
#         if isinstance(index, Err):
#             logger.debug(
#                 f"Index does not exist at '{self.index_path}', build index with 'build_index' method."
#             )
#             self._index = None  # type: ignore
#         else:
#             self._index: tantivy.Index = index.ok_value

#     @property
#     def index(self) -> tantivy.Index:
#         self._index.reload()
#         return self._index

#     @property
#     def column_types(self) -> dict[str, str]:
#         return self._column_types

#     @property
#     def id_column(self) -> str:
#         return self._id_column

#     @property
#     def main_column(self) -> str:
#         return self._main_column

#     @property
#     def index_path(self) -> Path:
#         return self._index_path

#     @property
#     def columns_not_id(self) -> list[str]:
#         return [col for col in self.column_types if col != self.id_column]

#     def _build_index(
#         self,
#         df: pl.DataFrame,
#         split_field: dict[str, list[str] | str] | None = None,
#     ) -> Result[tantivy.Index, str]:
#         """Only pass in data which you wish to build the ftx index with. split_field is a dictionary of fields to split by a delimiter. eg {",": ["field1", "field2"]} will split field1 and field2 by comma."""
#         # TODO: this programmatically into tantivy schema
#         schema_builder = tantivy.SchemaBuilder()

#         if self.id_column not in df.columns:
#             return Err(f"'{self.id_column}' column not found in DataFrame")

#         col_types = {}
#         for col in df.columns:
#             if col == self.id_column:
#                 schema_builder.add_integer_field(
#                     self.id_column, stored=True, indexed=True, fast=True
#                 )
#             # TODO: ADD support for other types
#             else:
#                 schema_builder.add_text_field(col)
#             col_types[col] = df[col].dtype._string_repr()

#         self._column_types = col_types

#         schema = schema_builder.build()
#         logger.debug(f"Creating index with columns:\n{json.dumps(col_types, indent=2)}")

#         index = tantivy.Index(schema, path=self.index_path.as_posix(), reuse=False)
#         writer = index.writer()
#         for row in df.rows(named=True):
#             if split_field:
#                 for splitter, fields in split_field.items():
#                     if isinstance(fields, str):
#                         fields = [fields]
#                     for field in fields:
#                         logger.debug(f"Splitting {field} by {splitter}...")
#                         row[field] = row[field].split(splitter)
#             writer.add_document(tantivy.Document(**row))
#         writer.commit()
#         writer.wait_merging_threads()
#         return Ok(index)

#     def get_index(self) -> Result[tantivy.Index, str]:
#         if tantivy.Index.exists(self.index_path.as_posix()):
#             logger.debug(f"Opening index at '{self.index_path}'")
#             return Ok(tantivy.Index.open(self.index_path.as_posix()))
#         return Err(f"Index does not exist at '{self.index_path}'")

#     def get_or_build_index(
#         self, df: pl.DataFrame | None = None
#     ) -> Result[tantivy.Index, str]:
#         if not self.index_path.exists() and df is None:
#             return Err("Index does not exist. DataFrame is required to create index")

#         self.index_path.mkdir(parents=True, exist_ok=True)

#         if not tantivy.Index.exists(self.index_path.as_posix()):
#             if df is None:
#                 return Err("DataFrame is required to create index")
#             match self._build_index(df):
#                 case Ok(index):
#                     ...
#                 case Err(e):
#                     return Err(e)
#         else:
#             match self.get_index():
#                 case Ok(index):
#                     ...
#                 case Err(e):
#                     return Err(e)
#         schema = json.loads((self.index_path / "meta.json").read_text())["schema"]
#         sc = {}
#         for v in schema:
#             type_ = v["type"]
#             if type_ == "text":
#                 type_ = pl.Utf8
#             elif type_ == "i64":
#                 type_ = pl.UInt32
#             sc[v["name"]] = type_

#         self._column_types = sc
#         logger.debug("Schema Loaded")
#         logger.debug("Opening country index")
#         return Ok(index)

#     def convert_fts_results(
#         self, hits: tantivy.SearchResult, searcher: tantivy.Searcher
#     ) -> pl.DataFrame:
#         logger.debug(f"FTS hits from search: {hits.count}")  # type: ignore

#         scores, gids = zip(
#             *[
#                 (score, searcher.doc(doc).get_first(self.id_column))
#                 for score, doc in hits.hits
#             ]
#         )

#         return (
#             pl.LazyFrame(
#                 {"geonameId": list(gids), "score": list(scores)},
#                 schema={"geonameId": pl.UInt32, "score": pl.Float32},
#             )
#             .sort("score", descending=True, maintain_order=True)
#             .collect()
#         )

#     def search(
#         self,
#         query: str,
#         limit: int = 10,
#         include: list[int] | None = None,
#         exclude: list[int] | None = None,
#     ) -> Result[pl.DataFrame, str]:
#         return self.fts_search(
#             query,
#             limit=limit,
#             include=include,
#             exclude=exclude,
#         )

#     def fts_search(
#         self,
#         query: str,
#         limit: int = 10,
#         include: list[int] | None = None,
#         exclude: list[int] | None = None,
#         main_term_query_boost: float = 3.0,
#         fuzzy_term_query_boost: float = 2.0,
#         max_fuzzy_distance: int = 2,
#         phrase: bool = True,
#     ) -> Result[pl.DataFrame, str]:
#         # Create for list of queries (batch search)
#         if phrase:
#             query = f"'{query}'"
#         else:
#             query = query.strip("\"'")
#         query = query.strip()
#         index = self.index

#         searcher = index.searcher()

#         bool_query_list: list[tuple[tantivy.Occur, tantivy.Query]] = []

#         # Calculate fuzzy distance based on query length
#         fuzzy_distance = min(max(0, len(query) - 2), max_fuzzy_distance)

#         if self.main_column in self.columns_not_id:
#             main_term_query = tantivy.Query.term_query(
#                 index.schema, self.main_column, query
#             )
#             bool_query_list.append(
#                 (
#                     tantivy.Occur.Should,
#                     tantivy.Query.boost_query(main_term_query, main_term_query_boost),
#                 )
#             )

#             if fuzzy_distance > 0:
#                 main_fuzzy_query = tantivy.Query.fuzzy_term_query(
#                     index.schema, self.main_column, query, distance=fuzzy_distance
#                 )

#                 bool_query_list.append(
#                     (
#                         tantivy.Occur.Should,
#                         tantivy.Query.boost_query(
#                             main_fuzzy_query, fuzzy_term_query_boost
#                         ),
#                     )
#                 )

#             rest_of_query = index.parse_query(
#                 query, list(set(self.columns_not_id) - {self.main_column})
#             )
#             bool_query_list.append((tantivy.Occur.Should, rest_of_query))

#         if include:
#             bool_query_list.append(
#                 (
#                     tantivy.Occur.Must,
#                     tantivy.Query.term_set_query(index.schema, self.id_column, include),
#                 )
#             )
#         if exclude:
#             bool_query_list.append(
#                 (
#                     tantivy.Occur.MustNot,
#                     tantivy.Query.term_set_query(index.schema, self.id_column, exclude),
#                 )
#             )
#         if bool_query_list:
#             final_query = tantivy.Query.boolean_query(bool_query_list)

#         else:
#             final_query: tantivy.Query = index.parse_query(
#                 query, default_field_names=self.columns_not_id
#             )

#         logger.debug(final_query)

#         hits: tantivy.SearchResult = searcher.search(final_query, limit=limit)

#         if hits.count == 0:  # type: ignore
#             if phrase:
#                 logger.debug("No results found, retrying without phrase search...")
#                 return self.fts_search(
#                     query,
#                     limit,
#                     include,
#                     exclude,
#                     main_term_query_boost,
#                     fuzzy_term_query_boost,
#                     max_fuzzy_distance,
#                     phrase=False,
#                 )
#             return Err("No results found")

#         return Ok(self.convert_fts_results(hits, searcher))


# class HybridIndex:
#     def __init__(self, fts_idx: FTSIndex, vidx: VectorIndex):
#         self._fts_idx = fts_idx
#         self._vidx = vidx

#     @property
#     def vector_index(self) -> VectorIndex:
#         return self._vidx

#     @property
#     def fts_index(self) -> FTSIndex:
#         return self._fts_idx

#     def search(
#         self,
#         query: str,
#         limit: int = 10,
#         include: list[int] | None = None,
#         exclude: list[int] | None = None,
#         main_term_query_boost: float = 3.0,
#         fuzzy_term_query_boost: float = 2.0,
#         max_fuzzy_distance: int = 2,
#         phrase: bool = True,
#     ) -> Result[pl.DataFrame, str]:
#         v_search = self.vector_index.vector_search


# country_index = FTSIndex("admin0", con.table("admin0_FTS").pl())
# country_index.fts_search("An Danmhairg").unwrap().join(
#     con.table("admin0").pl(), "geonameId", "left"
# )
