CREATE OR REPLACE TABLE equivalent AS
SELECT A.geonameId AS geonameId_a,
    P.geonameId AS geonameId_p
FROM (
        SELECT *
        FROM allCountries
        WHERE feature_class = 'A'
    ) AS A
    INNER JOIN (
        SELECT *
        FROM allCountries
        WHERE feature_class = 'P'
    ) AS P ON COALESCE(A.name, 'N/A') = COALESCE(P.name, 'N/A')
    AND COALESCE(A.asciiname, 'N/A') = COALESCE(P.asciiname, 'N/A')
    AND COALESCE(A.ISO, 'N/A') = COALESCE(P.ISO, 'N/A')
    AND COALESCE(A.admin1_code, 'N/A') = COALESCE(P.admin1_code, 'N/A')
    AND COALESCE(A.admin2_code, 'N/A') = COALESCE(P.admin2_code, 'N/A')
    AND COALESCE(A.admin3_code, 'N/A') = COALESCE(P.admin3_code, 'N/A')
    AND COALESCE(A.admin4_code, 'N/A') = COALESCE(P.admin4_code, 'N/A')
    AND COALESCE(A.timezone, 'N/A') = COALESCE(P.timezone, 'N/A')
ORDER BY geonameId_a,
    geonameId_p;
/*
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
 join_nulls=True,
 )
 .rename({"geonameId": "geonameId_a"})
 .select("geonameId_a", "geonameId_p")
 .sort("geonameId_a", "geonameId_p")
 )
 ab = q.collect()
 */