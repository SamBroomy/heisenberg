CREATE TABLE
    IF NOT EXISTS unique_ids AS
SELECT
    geonameId,
    name,
    feature_class,
    feature_code
FROM
    allCountries
WHERE
    geonameId IN (
        SELECT
            parentId
        FROM
            hierarchy
        UNION
        SELECT
            childId
        FROM
            hierarchy
    )