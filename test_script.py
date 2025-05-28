from heisenberg import Heisenberg
import random
import polars as pl
import polars.selectors as cs

pdf = pl.read_parquet("hberg_data/processed/place_search.parquet")
adf = pl.read_parquet("hberg_data/processed/admin_search.parquet")

hberg = Heisenberg(True)

re = hberg.search(
    [
        "US",
        "Fl",
        "Lakeland",
    ]
)

print(re[0].head(5))

re = hberg.resolve_location(["US", "Fl", "Lakeland"])

print(re)


def random_name(row: dict) -> tuple[int, str | None]:
    geonameId = row["geonameId"]
    # First random chance this entry is null 33.3%?
    if random.random() < (1 / 4):
        return (geonameId, None)
    # If not null then we pick either name, ascii_name or admin_code. If its admin_level 0 we could also select from [ISO, ISO3, official_name, fips, or alternate_name]
    admin_level: int | None = row.get("admin_level")

    if admin_level is not None and admin_level == 0:
        # Pick a random name from the list of names
        names = [
            "name",
            "asciiname",
            "alternatenames",
            "ISO",
            "ISO3",
            "official_name",
            "fips",
        ]
        weights = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    else:
        # Pick a random name from the list of names
        names = ["name", "asciiname", "alternatenames"]
        weights = [0.3, 0.3, 0.4]

    choice = random.choices(names, weights=weights, k=1)[0]

    if choice != "alternatenames":
        return (geonameId, row[choice])
    # Pick a random name from the list of names
    alt_names: list[str] = row["alternatenames"]
    if alt_names is not None and len(alt_names) > 0:
        # Pick a random name from the list of names
        return (geonameId, random.choice(alt_names))
    else:
        return (geonameId, None)


def backfill_filter_level(
    df: pl.LazyFrame | pl.DataFrame, vals: dict[str, str]
) -> tuple[int, str | None]:
    df = df.lazy()

    filter = {k: v for k, v in vals.items() if v is not None}

    # Extract the admin*_code (0-4) form the key in val dict and convert to int
    max_level = max([int(k.split("_")[0][-1]) for k, v in vals.items()])

    admin_level = pl.col("admin_level") <= max_level

    nulls = cs.matches("admin[0-4]_code").exclude(filter.keys()).is_null()

    df = (
        df.filter(nulls, admin_level, **filter)
        .sort(["admin_level", "feature_class"])
        .collect()
    )
    if df.is_empty():
        return (None, None)
    row = df.row(0, named=True)

    return random_name(row)


def backfill(
    df: pl.LazyFrame | pl.DataFrame, vals: dict[str, str]
) -> list[tuple[int, str | None]] | None:
    vals = dict(sorted(vals.items()))

    prev_vals = {}
    output: list[tuple[int, str | None]] = []

    for key, val in vals.items():
        prev_vals |= {key: val}
        output.append(backfill_filter_level(df, prev_vals))

    # If the last value is null then we need to drop it and keep going till the last value is not null
    for i in range(len(output) - 1, -1, -1):
        if output[i][1] is None:
            output.pop(i)
        else:
            break

    if len(output) < 2:
        return None

    return output


rows: list[list[tuple[int, str | None]]] = []

sample = 100

for row in (
    pl.concat(
        [
            pdf.filter(
                pl.col.admin0_code.is_in(
                    [
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
                )
            ).sample(sample / 2),
            pdf.sample(sample / 2),
        ]
    )
    .unique()
    .iter_rows(named=True)
):
    current = random_name(row)

    row = dict(
        filter(lambda i: i[0] in [f"admin{i}_code" for i in range(5)], row.items())
    )

    if (out := backfill(adf, row)) is None:
        continue
    if current[1] is not None:
        out.append(current)
    rows.append(out)


def unzip(
    lst: list[list[tuple[int, str | None]]],
) -> tuple[list[list[int]], list[list[str]]]:
    gids = list(map(lambda i: list(map(lambda j: j[0], i)), lst))
    names = list(
        map(
            lambda i: list(filter(lambda j: j is not None, map(lambda j: j[1], i))), lst
        )
    )

    return gids, names


gids, names = unzip(rows)

out = hberg.resolve_location(names)
