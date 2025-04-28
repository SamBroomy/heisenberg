CREATE TABLE admin2Codes (
    code VARCHAR(80) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    asciiname VARCHAR(200),
    geonameId INTEGER,
    FOREIGN KEY (geonameId) REFERENCES allCountries(geonameId)
);