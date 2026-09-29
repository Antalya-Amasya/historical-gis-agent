# Pleiades v2 fixture — provenance and attribution

This notice describes the tracked derived file
`backend/tests/fixtures/pleiades_index_v2_baseline.sqlite3`.

It is **not** a legal determination and **does not** clear the fixture for
redistribution.

## Fixture identity

| | |
| --- | --- |
| File | `pleiades_index_v2_baseline.sqlite3` |
| Size | 40,968,192 bytes (~41 MB) |
| SHA-256 | `401be897990dd1f7bb1f94af5b1f789b48a4c28bfc4986ffe609015c2ff97190` |

The file is a **derived SQLite gazetteer index**, not the upstream zip.

## Upstream record

Identified release: **Pleiades Datasets 4.1** (Zenodo version `v4.1`).

- Record: https://zenodo.org/records/15540082
- DOI: [10.5281/zenodo.15540082](https://doi.org/10.5281/zenodo.15540082)
- Zenodo publication / release date: **2025-05-28**

V1-D6 and this notice used that record. A fetch of the same page lists
editors including Elliott, Tom; Talbert, Richard; Bagnall, Roger; Becker,
Jeffrey; Bond, Sarah; Gillies, Sean; Holman, Lindsay; Horne, Ryan; Moss,
Gabe; Rabinowitz, Adam; and others (the page reports 13 authors/editors
total). Content contributors are described on the record as listed in
upstream `data/rdf/authors.ttl`; that file is not stored in this
repository.

Zenodo recommended citation badge target: `https://doi.org/10.5281/zenodo.15540082`.

### UPSTREAM LICENSE IDENTIFICATION

Verified on **2026-09-29** from the official API
https://zenodo.org/api/records/15540082:

- dataset title: **Pleiades Datasets 4.1**
- publication date: **2025-05-28**
- `metadata.license.id` = **`cc-by-3.0`**
- record description links to the CC BY 3.0 deed:
  https://creativecommons.org/licenses/by/3.0/

The HTML landing page still shows the short phrase “Creative Commons
Attribution license (cc-by)”; the **versioned identifier** is the API
`license.id` value, not that unversioned phrase.

## Embedded fixture metadata

Read-only `metadata` table in this SQLite file (not inferred from Zenodo):

| Key | Value |
| --- | --- |
| dataset_version | 4.1 |
| dataset_release_date | 2025-05-28 |
| official_source | https://zenodo.org/records/15540082 |
| sha256 (embedded source ZIP hash) | `94c5c337d27a07f1a5fa231b6513e40e6d3cf7d5ad7b8c010f8e8bfd9159cd85` |
| license | **CC BY 3.0** |
| index_schema_version | 2 |
| importer_version | 2 |
| place_count | 41480 |
| location_count | 44663 |
| name_count | 75671 |
| attestation_count | 85203 |
| build_timestamp | 2026-09-01T03:58:17.180876+00:00 |

Table row counts match those metadata values.

### DERIVED SQLITE REDISTRIBUTION REVIEW

The derived index stores `license=CC BY 3.0`. That string was written into
the file at index-build time. It agrees with the upstream API identifier
`cc-by-3.0`. Agreement on the **license version** is **not** an independent
publication clearance for this derived SQLite file.

## Test purpose

Path used by tests:

`backend/tests/fixtures/pleiades_index_v2_baseline.sqlite3`

- `backend/tests/test_pleiades_index_g3.py` — copy the file and assert
  runtime **rejects schema v2** (`Unsupported Pleiades index schema: '2'`).
- `backend/tests/test_g6da_pleiades_v3_location_source_preservation.py` —
  same schema-v2 rejection, and **v2/v3 resolution parity** for selected
  names against a schema-v3 index built from the current importer.

The fixture is a frozen schema-v2 baseline for those checks. It is not the
production gazetteer path.

## Missing v2 generation contract

`scripts/build_pleiades_index.py` currently writes **schema version 3** and
**importer version 3**. It does **not** reproduce this schema-v2 baseline.

The exact command, script revision, and options used to build this v2 file
are **not present** in the current repository. This notice does not invent
them.

## Remaining uncertainty

- **DERIVED SQLITE REDISTRIBUTION REVIEW** remains open: whether Git may
  keep or publish this derived 41 MB index is **not** decided by this
  notice.
- Completeness of author/editor lists beyond the Zenodo record summary is
  not established from files in this repo. Contributor attribution has not
  been independently audited against upstream `authors.ttl`.
- The exact importer-v2 build command remains missing.
