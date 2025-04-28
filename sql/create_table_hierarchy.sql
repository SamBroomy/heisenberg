CREATE TABLE hierarchy (
    parentId INTEGER,
    childId INTEGER,
    type VARCHAR(20),
    PRIMARY KEY (parentId, childId),
    FOREIGN KEY (parentId) REFERENCES allCountries(geonameId),
    FOREIGN KEY (childId) REFERENCES allCountries(geonameId)
);
CREATE INDEX idx_hierarchy_parent ON hierarchy (parentId);
CREATE INDEX idx_hierarchy_child ON hierarchy (childId);