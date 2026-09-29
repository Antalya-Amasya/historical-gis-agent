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

### UPSTREAM RECORD (license wording)

The fetched Zenodo summary states that content is distributed under a
**Creative Commons Attribution license (cc-by)**. That page text does
**not** state an explicit CC version number.

This notice does **not** treat that wording as an independent verification
of CC BY 3.0.

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

### EMBEDDED FIXTURE METADATA (license string)

The derived index stores `license=CC BY 3.0`. That string was written into
the file at index-build time. It is **not** the same as the unversioned
cc-by wording on the fetched Zenodo summary. The two are recorded
separately and are not reconciled here.

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

- License version on the upstream landing page vs `CC BY 3.0` in SQLite
  metadata is unresolved.
- Whether Git may redistribute this derived 41 MB index (as distinct from
  citing the Zenodo dataset) is **not** decided by this notice.
- Completeness of author/editor lists beyond the Zenodo record summary is
  not established from files in this repo.
