-- Active: 1745401560330@@127.0.0.1@3306
SELECT geonameId,
    name,
    score
FROM (
        SELECT *,
            fts_main_country.match_bm25(
                geonameId,
                "United States of American of America"
            ) AS score
        FROM country
    ) sq
WHERE score IS NOT NULL
ORDER BY score DESC;