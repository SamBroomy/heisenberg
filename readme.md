# Task

To turn unstructured data and incomplete structured data into

# Inputs

- `place`: This is the name of the place to search for.
- `country`: The top level domain (TLD) of the country to search in. For example, "us" for the United States, "uk" for the United Kingdom, etc. This would be the country code.
- `state`: The state (the first administrative division) of the country to search in. For example, "CA" for California, "NY" for New York, etc.
- `county`: The county (the second administrative division) of the country to search in. For example, "Los Angeles County" for Los Angeles, "Cook County" for Chicago, etc.
- `city`: The city (the third administrative division) of the country to search in. For example, "Los Angeles" for Los Angeles, "Chicago" for Chicago, etc.
- `postal_code`: The postal code of the place to search for. For example, "90001" for Los Angeles, "60601" for Chicago, etc.

# Is there a better word than state county and city as these terms are not used in all countries,  and can have diffrent administrative divisions in different countries?

A better word than "state" could be "region" or "province," as these terms are more universally applicable across different countries. Similarly, instead of "county," you could use "district" or "locality," and instead of "city," you could use "town" or "municipality." These terms are more inclusive and can accommodate the various administrative divisions used in different countries.

# Inputs

I think the main idea needs to be it tries to aggressively filter the data based on previous searches, and once it has filtered

# Postal outputs

Parser labels

The address parser can technically use any string labels that are defined in the training data, but these are the ones currently defined, based on the fields defined in OpenCage's address-formatting library, as well as a few added by libpostal to handle specific patterns:

- house: venue name e.g. "Brooklyn Academy of Music", and building names e.g. "Empire State Building"category: for category queries like "restaurants", etc.
- near: phrases like "in", "near", etc. used after a category phrase to help with parsing queries like "restaurants in Brooklyn"
 house_number: usually refers to the external (street-facing) building number. In some countries this may be a compount, hyphenated number which also includes an apartment number, or a block number (a la Japan), but libpostal will just call it the house_number for simplicity.
- road: street name(s)
- unit: an apartment, unit, office, lot, or other secondary unit designator
- level: expressions indicating a floor number e.g. "3rd Floor", "Ground Floor", etc.
- staircase: numbered/lettered staircase
- entrance: numbered/lettered entrance
- po_box: post office box: typically found in non-physical (mail-only) addresses
- postcode: postal codes used for mail sorting
- suburb: usually an unofficial neighborhood name like "Harlem", "South Bronx", or "Crown Heights"
- city_district: these are usually boroughs or districts within a city that serve some official purpose e.g. "Brooklyn" or "Hackney" or "Bratislava IV"
- city: any human settlement including cities, towns, villages, hamlets, localities, etc.
- island: named islands e.g. "Maui"
- state_district: usually a second-level administrative division or county.
- state: a first-level administrative division. Scotland, Northern Ireland, Wales, and England in the UK are mapped to "state" as well (convention used in OSM, GeoPlanet, etc.)
- country_region: informal subdivision of a country without any political status
- country: sovereign nations and their dependent territories, anything with an ISO-3166 code.
- world_region: currently only used for appending “West Indies” after the country name, a pattern frequently used in the English-speaking Caribbean e.g. “Jamaica, West Indies”

# Plan

Ok so it seems like what I really want to do is to have at least is to try and use as much of the leading data (country, state, and potentially district) to try and narrow down the search as much as possible.

From there I then think its a case of trying to search against the name of the place.

The difficulty lies in the fact that the each entry could correspond to any type of entity. Most of the entities are just places, and would be the final result of the search. But some entities (the ones i need to narrow down the search), are the admin districts.

I need to construct the data in such a way, that I am able to query at the different levels of the hierarchy and filter accordingly based on the results, to then be able to search for the place name in the smaller search space.

Based on the geonames data, I can construct the top level hierarchy (country level) by using the following sql query:

But this is maybe where the main issues lie. What I need to do is to search for the place for that particular country.

TODO:

- [ ] Ordering of the where clause to be more efficient and correctly structured
- [ ] Ensure the search tables have distinct geoname ids
- [ ] Get the adjusted score for flexible search to follow the correct path.
