CREATE TABLE
    shapes AS
SELECT
    *
FROM
    ST_Read ('./data/raw/geonames/shapes_simplified_low.json');