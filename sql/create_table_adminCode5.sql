CREATE TABLE adminCode5 (
    geonameId INTEGER PRIMARY KEY,
    adm5code VARCHAR(20) NOT NULL,
    FOREIGN KEY (geonameId) REFERENCES allCountries(geonameId)
);