CREATE
OR REPLACE TABLE admin_search (
    geonameId UINTEGER NOT NULL,
    name VARCHAR(175) NOT NULL,
    asciiname VARCHAR(175),
    admin_level UTINYINT NOT NULL,
    admin0_code CHAR(2),
    admin1_code VARCHAR(20),
    admin2_code VARCHAR(75),
    admin3_code VARCHAR(75),
    admin4_code VARCHAR(75),
    feature_class CHAR(1),
    feature_code VARCHAR(5),
    ISO CHAR(2),
    ISO3 CHAR(3),
    ISO_Numeric USMALLINT,
    official_name VARCHAR(50),
    fips CHAR(2),
    latitude FLOAT,
    longitude FLOAT,
    population INTEGER,
    area FLOAT,
    alternatenames TEXT,
    country_name VARCHAR(50),
    PRIMARY KEY (geonameId),
);

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