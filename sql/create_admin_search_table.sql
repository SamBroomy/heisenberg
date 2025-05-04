
CREATE OR REPLACE TABLE admin_search AS
SELECT
    geonameId,
    name,
    asciiname,
    admin0_code,
    NULL::VARCHAR AS admin1_code,
    NULL::VARCHAR AS admin2_code,
    NULL::VARCHAR AS admin3_code,
    NULL::VARCHAR AS admin4_code,
    feature_class,
    feature_code,
    ISO,
    ISO3,
    ISO_Numeric,
    official_name,
    fips,
    population,
    area,
    alternatenames,
    NULL::INTEGER AS parent_id,
    NULL::VARCHAR AS country_name,
    NULL::VARCHAR AS parent_name,
    'admin0' AS source_table
FROM admin0

UNION ALL

-- Admin1


SELECT
    geonameId,
    name,
    asciiname,
    admin0_code,
    admin1_code,
    NULL::VARCHAR AS admin2_code,
    NULL::VARCHAR AS admin3_code,
    NULL::VARCHAR AS admin4_code,
    feature_class,
    feature_code,
    NULL::VARCHAR AS ISO,
    NULL::VARCHAR AS ISO3,
    NULL::INTEGER AS ISO_Numeric,
    NULL::VARCHAR AS official_name,
    NULL::VARCHAR AS fips,
    population,
    NULL::FLOAT AS area,
    alternatenames,
    parent_id,
    country_name,
    parent_name,
    'admin1' AS source_table
FROM admin1

UNION ALL

-- Admin2
SELECT
    geonameId,
    name,
    asciiname,
    admin0_code,
    admin1_code,
    admin2_code,
    NULL::VARCHAR AS admin3_code,
    NULL::VARCHAR AS admin4_code,
    feature_class,
    feature_code,
    NULL::VARCHAR AS ISO,
    NULL::VARCHAR AS ISO3,
    NULL::INTEGER AS ISO_Numeric,
    NULL::VARCHAR AS official_name,
    NULL::VARCHAR AS fips,
    population,
    NULL::FLOAT AS area,
    alternatenames,
    parent_id,
    country_name,
    parent_name,
    'admin2' AS source_table
FROM admin2
UNION ALL
-- Admin3
SELECT
    geonameId,
    name,
    asciiname,
    admin0_code,
    admin1_code,
    admin2_code,
    admin3_code,
    NULL::VARCHAR AS admin4_code,
    feature_class,
    feature_code,
    NULL::VARCHAR AS ISO,
    NULL::VARCHAR AS ISO3,
    NULL::INTEGER AS ISO_Numeric,
    NULL::VARCHAR AS official_name,
    NULL::VARCHAR AS fips,
    population,
    NULL::FLOAT AS area,
    alternatenames,
    parent_id,
    country_name,
    parent_name,
    'admin3' AS source_table
FROM admin3
UNION ALL
-- Admin4
SELECT
    geonameId,
    name,
    asciiname,
    admin0_code,
    admin1_code,
    admin2_code,
    admin3_code,
    admin4_code,
    feature_class,
    feature_code,
    NULL::VARCHAR AS ISO,
    NULL::VARCHAR AS ISO3,
    NULL::INTEGER AS ISO_Numeric,
    NULL::VARCHAR AS official_name,
    NULL::VARCHAR AS fips,
    population,
    NULL::FLOAT AS area,
    alternatenames,
    parent_id,
    country_name,
    parent_name,
    'admin4' AS source_table
FROM admin4;

CREATE INDEX idx_admin_search_geoname ON admin_search (geonameId);

CREATE INDEX source_table_admin_search ON admin_search (source_table);

CREATE INDEX idx_admin_search_parent ON admin_search (parent_id) CREATE INDEX idx_admin_search_admins_code ON admin_search (
    admin0_code,
    admin1_code,
    admin2_code,
    admin3_code,
    admin4_code
)