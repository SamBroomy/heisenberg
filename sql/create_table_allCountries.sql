CREATE TABLE allCountries (
    geonameId INTEGER PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    asciiname VARCHAR(200),
    alternatenames TEXT,
    latitude DOUBLE NOT NULL CHECK (
        latitude BETWEEN -90 AND 90
    ),
    longitude DOUBLE NOT NULL CHECK (
        longitude BETWEEN -180 AND 180
    ),
    feature_class CHAR(1) CHECK (
        feature_class IN ('A', 'H', 'L', 'P', 'R', 'S', 'T', 'U', 'V')
    ),
    feature_code VARCHAR(10),
    admin0_code CHAR(2),
    cc2 VARCHAR(200),
    admin1_code VARCHAR(20),
    admin2_code VARCHAR(80),
    admin3_code VARCHAR(20),
    admin4_code VARCHAR(20),
    population BIGINT,
    elevation INTEGER,
    dem INTEGER,
    timezone VARCHAR(40),
    modification_date DATE
);
CREATE INDEX idx_allCountries_coords ON allCountries (latitude, longitude);
CREATE INDEX idx_allCountries_name ON allCountries (name);
CREATE INDEX idx_allCountries_feature ON allCountries (feature_class, feature_code);
CREATE INDEX idx_allCountries_admin ON allCountries (
    admin0_code,
    admin1_code,
    admin2_code,
    admin3_code,
    admin4_code
);