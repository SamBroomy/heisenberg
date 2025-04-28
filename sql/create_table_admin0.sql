-- CREATE OR REPLACE TABLE admin0 AS
-- SELECT allCountries.geonameId,
--     allCountries.name,
--     allCountries.asciiname,
--     allCountries.alternatenames,
--     allCountries.latitude,
--     allCountries.longitude,
--     allCountries.feature_class,
--     allCountries.feature_code,
--     allCountries.cc2,
--     allCountries.elevation,
--     allCountries.dem,
--     allCountries.timezone,
--     allCountries.admin0_code,
--     countryInfo.ISO,
--     countryInfo.ISO3,
--     countryInfo.ISO_Numeric,
--     countryInfo.fips,
--     countryInfo.Country,
--     countryInfo.Capital,
--     countryInfo.Area,
--     countryInfo.Population,
--     countryInfo.Continent,
--     countryInfo.tld,
--     countryInfo.CurrencyCode,
--     countryInfo.CurrencyName,
--     countryInfo.Phone,
--     countryInfo.Postal_Code_Format,
--     countryInfo.Postal_Code_Regex,
--     countryInfo.Languages,
--     countryInfo.neighbours,
--     FROM countryInfo
--     INNER JOIN allCountries ON countryInfo.geonameId = allCountries.geonameId
-- ORDER BY allCountries.geonameId;
-- CREATE INDEX country_geonameId ON admin0 (geonameId);
--------------------------------------------------
CREATE OR REPLACE TABLE admin0 AS
SELECT c.geonameId,
    c.name,
    c.asciiname,
    c.admin0_code,
    c.cc2,
    ci.ISO,
    ci.ISO3,
    ci.ISO_Numeric,
    ci.Country AS official_name,
    ci.fips,
    ci.Population,
    ci.Area,
    c.alternatenames,
    c.feature_class,
    c.feature_code
FROM allCountries c
    INNER JOIN countryInfo ci ON c.geonameId = ci.geonameId
WHERE c.feature_code != 'PCLH'
ORDER BY c.geonameId;
-- Create efficient indexes
CREATE INDEX idx_admin0_geoname ON admin0 (geonameId);
CREATE INDEX idx_admin0_codes ON admin0 (admin0_code, ISO, ISO3);