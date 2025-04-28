CREATE
OR REPLACE VIEW locations_full AS
SELECT
    g.geonameId,
    g.name,
    g.asciiname,
    g.feature_class,
    g.feature_code,
    f.name AS feature_name,
    f.description AS feature_description,
    g.admin0_code,
    c.Country AS country_name,
    g.admin1_code,
    a1.name AS admin1_name,
    g.admin2_code,
    a2.name AS admin2_name,
    g.admin3_code,
    g.admin4_code,
    g.latitude,
    g.longitude,
    g.population,
    g.timezone,
    tz.GMT_offset_1_Jan_2024 AS timezone_offset
FROM
    allCountries g
    LEFT JOIN countryInfo c ON g.admin0_code = c.ISO
    LEFT JOIN admin1CodesASCII a1 ON g.admin0_code || '.' || g.admin1_code = a1.code
    LEFT JOIN admin2Codes a2 ON g.admin0_code || '.' || g.admin1_code || '.' || g.admin2_code = a2.code
    LEFT JOIN featureCodes f ON g.feature_class || '.' || g.feature_code = f.code
    LEFT JOIN timeZones tz ON g.timezone = tz.TimeZoneId;