CREATE TABLE alternateNamesV2 (
    alternateNameId INTEGER PRIMARY KEY,
    geonameId INTEGER NOT NULL,
    isolanguage VARCHAR(10),
    alternate_name VARCHAR(400),
    isPreferredName BOOLEAN DEFAULT FALSE,
    isShortName BOOLEAN DEFAULT FALSE,
    isColloquial BOOLEAN DEFAULT FALSE,
    isHistoric BOOLEAN DEFAULT FALSE,
    "from" VARCHAR(50),
    "to" VARCHAR(50),
    FOREIGN KEY (geonameId) REFERENCES allCountries(geonameId)
);
CREATE INDEX idx_alternateNamesV2_geonameId ON alternateNamesV2 (geonameId);
CREATE INDEX idx_alternateNamesV2_lang ON alternateNamesV2 (isolanguage);