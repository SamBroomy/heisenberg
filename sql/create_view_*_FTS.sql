CREATE OR REPLACE VIEW {table}_FTS AS
SELECT
    geonameId, name, official_name, ISO, ISO3, alternatenames
FROM
    {table}