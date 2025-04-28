CREATE TABLE countryInfo (
    ISO CHAR(2) PRIMARY KEY,
    ISO3 CHAR(3),
    ISO_Numeric INTEGER,
    fips VARCHAR(10),
    Country VARCHAR(200) NOT NULL,
    Capital VARCHAR(200),
    Area DOUBLE,
    Population INTEGER,
    Continent CHAR(2) CHECK (
        Continent IN ('AF', 'AS', 'EU', 'NA', 'OC', 'SA', 'AN')
    ),
    tld VARCHAR(10),
    CurrencyCode VARCHAR(10),
    CurrencyName VARCHAR(50),
    Phone VARCHAR(20),
    Postal_Code_Format VARCHAR(100),
    Postal_Code_Regex VARCHAR(200),
    Languages VARCHAR(200),
    geonameId INTEGER,
    neighbours VARCHAR(100),
    EquivalentFipsCode VARCHAR(10)
);