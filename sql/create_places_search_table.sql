CREATE
OR REPLACE TABLE places_search (
    geonameId UINTEGER NOT NULL,
    name VARCHAR(175) NOT NULL,
    asciiname VARCHAR(175),
    admin0_code CHAR(2),
    admin1_code VARCHAR(20),
    admin2_code VARCHAR(75),
    admin3_code VARCHAR(75),
    admin4_code VARCHAR(75),
    feature_class CHAR(1),
    feature_code VARCHAR(5),
    feature_name VARCHAR(50),
    latitude FLOAT,
    longitude FLOAT,
    population INTEGER,
    elevation SMALLINT,
    alternatenames TEXT,
    country_name VARCHAR(50),
    importance_score FLOAT NOT NULL,
    importance_tier UTINYINT NOT NULL,
    PRIMARY KEY (geonameId),
);

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