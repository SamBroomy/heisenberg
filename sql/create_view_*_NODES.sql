CREATE
OR REPLACE VIEW {table}_NODES AS
SELECT
    geonameId,
    name,
    feature_class,
    feature_code
FROM
    {table};