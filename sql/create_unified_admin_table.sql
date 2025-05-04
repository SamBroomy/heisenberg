CREATE OR REPLACE TABLE admin_search AS
WITH empty_table AS (
    -- Empty placeholder to establish schema
    SELECT
        NULL::INTEGER AS geonameId,
        NULL::VARCHAR AS name,
        NULL::VARCHAR AS asciiname,
        0 AS admin_level,
        NULL::VARCHAR AS admin0_code,
        NULL::VARCHAR AS admin1_code,
        NULL::VARCHAR AS admin2_code,
        NULL::VARCHAR AS admin3_code,
        NULL::VARCHAR AS admin4_code,
        NULL::VARCHAR AS feature_class,
        NULL::VARCHAR AS feature_code,
        NULL::VARCHAR AS ISO,
        NULL::VARCHAR AS ISO3,
        NULL::INTEGER AS ISO_Numeric,
        NULL::VARCHAR AS official_name,
        NULL::VARCHAR AS fips,
        NULL::DOUBLE AS latitude,
        NULL::DOUBLE AS longitude,
        NULL::BIGINT AS population,
        NULL::DOUBLE AS area,
        NULL::TEXT AS alternatenames,
        NULL::VARCHAR AS country_name
    WHERE 1 = 0
)
SELECT * FROM empty_table;

-- Create essential indexes upfront
CREATE INDEX idx_admin_search_geoname ON admin_search (geonameId);

CREATE INDEX idx_admin_search_level ON admin_search (admin_level);

CREATE INDEX idx_admin_search_feature ON admin_search (feature_class, feature_code);

CREATE INDEX idx_admin_search_name ON admin_search (name);

CREATE INDEX idx_admin_search_admin_codes ON admin_search (
    admin0_code,
    admin1_code,
    admin2_code,
    admin3_code,
    admin4_code
);

-- Views for each admin level
CREATE
OR REPLACE VIEW admin0 AS
SELECT *
FROM admin_search
WHERE
    admin_level = 0;

CREATE
OR REPLACE VIEW admin1 AS
SELECT *
FROM admin_search
WHERE
    admin_level = 1;

CREATE
OR REPLACE VIEW admin2 AS
SELECT *
FROM admin_search
WHERE
    admin_level = 2;

CREATE
OR REPLACE VIEW admin3 AS
SELECT *
FROM admin_search
WHERE
    admin_level = 3;

CREATE
OR REPLACE VIEW admin4 AS
SELECT *
FROM admin_search
WHERE
    admin_level = 4;