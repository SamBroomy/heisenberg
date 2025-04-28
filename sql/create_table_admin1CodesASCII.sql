CREATE TABLE admin1CodesASCII (
    code VARCHAR(20) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    name_ascii VARCHAR(200),
    geonameId INTEGER,
    FOREIGN KEY (geonameId) REFERENCES allCountries(geonameId)
);