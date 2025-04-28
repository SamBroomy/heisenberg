CREATE TABLE timeZones (
    CountryCode CHAR(2),
    TimeZoneId VARCHAR(40) PRIMARY KEY,
    GMT_offset_1_Jan_2024 DOUBLE,
    DST_offset_1_Jul_2024 DOUBLE,
    rawOffset DOUBLE
);