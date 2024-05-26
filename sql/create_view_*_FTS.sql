CREATE OR REPLACE VIEW {table}_FTS AS
SELECT
    geonameId, name, Country, ISO, ISO3, alternatenames
FROM
    {table}