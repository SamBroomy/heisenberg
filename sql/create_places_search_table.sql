CREATE OR REPLACE TABLE places_search AS
WITH empty_table AS (
    -- Empty placeholder to establish schema
    SELECT
        NULL::INTEGER AS geonameId,
        NULL::VARCHAR AS name,
        NULL::VARCHAR AS asciiname,
        NULL::VARCHAR AS admin0_code,
        NULL::VARCHAR AS admin1_code,
        NULL::VARCHAR AS admin2_code,
        NULL::VARCHAR AS admin3_code,
        NULL::VARCHAR AS admin4_code,
        NULL::VARCHAR AS feature_class,
        NULL::VARCHAR AS feature_code,
        NULL::VARCHAR AS feature_name,
        NULL::DOUBLE AS latitude,
        NULL::DOUBLE AS longitude,
        NULL::BIGINT AS population,
        NULL::INTEGER AS elevation,
        NULL::TEXT AS alternatenames,
        NULL::VARCHAR AS country_name,
        NULL::DOUBLE AS importance_score,
        -- Generated column for importance tier
        NULL::INTEGER AS importance_tier
    WHERE 1 = 0
)
SELECT * FROM empty_table;

-- Drop the table if it exists and recreate with proper schema
DROP TABLE IF EXISTS places_search;

CREATE TABLE places_search (
    geonameId INTEGER,
    name VARCHAR,
    asciiname VARCHAR,
    admin0_code VARCHAR,
    admin1_code VARCHAR,
    admin2_code VARCHAR,
    admin3_code VARCHAR,
    admin4_code VARCHAR,
    feature_class VARCHAR,
    feature_code VARCHAR,
    feature_name VARCHAR,
    latitude DOUBLE,
    longitude DOUBLE,
    population BIGINT,
    elevation INTEGER,
    alternatenames TEXT,
    country_name VARCHAR,
    importance_score DOUBLE,
    importance_tier INTEGER
);

-- Create indexes for efficient searching
CREATE INDEX idx_places_search_geoname ON places_search (geonameId);

CREATE INDEX idx_places_search_feature ON places_search (feature_class, feature_code);

CREATE INDEX idx_places_search_name ON places_search (name);

CREATE INDEX idx_places_search_admin_codes ON places_search (
    admin0_code,
    admin1_code,
    admin2_code,
    admin3_code,
    admin4_code
);

CREATE INDEX idx_places_search_coords ON places_search (latitude, longitude);

CREATE INDEX idx_places_search_importance ON places_search (importance_score DESC);

CREATE INDEX idx_places_search_tier ON places_search (importance_tier);

CREATE INDEX idx_places_search_admin_tier ON places_search (
    admin0_code,
    admin1_code,
    admin2_code,
    admin3_code,
    admin4_code,
    importance_tier
);