CREATE OR REPLACE VIEW cities_view AS
SELECT g.geonameId,
    g.name,
    g.asciiname,
    g.latitude,
    g.longitude,
    g.admin0_code AS country_code,
    c.Country AS country_name,
    g.admin1_code,
    a1.name AS admin1_name,
    g.admin2_code,
    a2.name AS admin2_name,
    g.population,
    g.timezone
FROM allCountries g
    LEFT JOIN countryInfo c ON g.admin0_code = c.ISO
    LEFT JOIN admin1CodesASCII a1 ON g.admin0_code || '.' || g.admin1_code = a1.code
    LEFT JOIN admin2Codes a2 ON g.admin0_code || '.' || g.admin1_code || '.' || g.admin2_code = a2.code
WHERE g.feature_class = 'P'
    AND g.population > 5000;