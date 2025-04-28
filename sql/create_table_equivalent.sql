CREATE OR REPLACE TABLE equivalent AS
SELECT A.geonameId AS geonameId_a,
    P.geonameId AS geonameId_p
FROM (
        SELECT *
        FROM allCountries
        WHERE feature_class = 'A'
    ) AS A
    INNER JOIN (
        SELECT *
        FROM allCountries
        WHERE feature_class = 'P'
    ) AS P ON COALESCE(A.name, 'N/A') = COALESCE(P.name, 'N/A')
    AND COALESCE(A.asciiname, 'N/A') = COALESCE(P.asciiname, 'N/A')
    AND COALESCE(A.admin0_code, 'N/A') = COALESCE(P.admin0_code, 'N/A')
    AND COALESCE(A.admin1_code, 'N/A') = COALESCE(P.admin1_code, 'N/A')
    AND COALESCE(A.admin2_code, 'N/A') = COALESCE(P.admin2_code, 'N/A')
    AND COALESCE(A.admin3_code, 'N/A') = COALESCE(P.admin3_code, 'N/A')
    AND COALESCE(A.admin4_code, 'N/A') = COALESCE(P.admin4_code, 'N/A')
    AND COALESCE(A.timezone, 'N/A') = COALESCE(P.timezone, 'N/A')
ORDER BY geonameId_a,
    geonameId_p;
CREATE INDEX idx_equivalent_a ON equivalent (geonameId_a);
CREATE INDEX idx_equivalent_p ON equivalent (geonameId_p);