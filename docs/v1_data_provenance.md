# V1 data provenance and dependencies

Scope: what a clone of `agent/cursor` at `ce9a60b674378aeded33f688bc0fe5ff6182e386` contains, and which runtime assets stay outside Git. Evidence is the tracked tree plus local files inspected for V1-RC9 and this note. This audit did not download data, rebuild Chroma, or re-fetch upstream records.

Labels:

- **VERIFIED** — stated in a tracked file, or checked against a local file in that audit.
- **OBSERVED LOCALLY** — present on the canonical Windows machine, not in Git.
- **UNVERIFIED** — a recorded claim that this audit did not re-check against the upstream source.
- **UNKNOWN** — not recorded in the repository.

## Runtime asset inventory

| Asset | Role | Path or setting | Tracked | Availability |
| --- | --- | --- | --- | --- |
| Backend packages | API and agent | `backend/requirements.txt` | Yes | **VERIFIED** in Git |
| Frontend packages | Map UI | `frontend/pnpm-lock.yaml` | Yes | **VERIFIED** in Git. `node_modules` is ignored |
| Provider keys | Live LLM | `.env.example` has empty key fields. Real `.env` is gitignored | Example only | **VERIFIED** ignore rule. Live keys are **OBSERVED LOCALLY** and are not documented here |
| Chroma collection `roman_republic_primary_sources_v2` | Retrieval | `C:\D\python\202608231533\data\chroma_server_roman_republic_v2` via the launcher | No | **OBSERVED LOCALLY**: 7,230 records (V1-RC1) |
| Embedding model | Query and ingest vectors | `rag_embedding_model=intfloat/multilingual-e5-small` | No | Weights are a local cache. Launcher sets `HF_HUB_OFFLINE=1` |
| Pleiades index | Place resolution | `data/pleiades_v4_1/pleiades_v4_1.sqlite3` or `PLEIADES_GAZETTEER_PATH` | No | **OBSERVED LOCALLY** in this worktree |
| Natural Earth 1:10m | Optional sea surface | `MARITIME_SURFACE_DATA_ROOT` | No | **OBSERVED LOCALLY** under `data/gis/natural_earth_10m`. Manifest audit `AVAILABLE` |
| Itiner-e roads | Optional Roman-road graph | `ROMAN_ROAD_GEOJSON_PATH` | Metadata only | GeoJSON **OBSERVED LOCALLY** and gitignored. `data/raw/itiner_e/metadata.json` is tracked |
| Caesar corpus JSON | One tracked source extract | `data/historical_sources/raw/caesar/corpus.json` | Yes | Not the full collection |

The launcher in `scripts/start_historical_gis.ps1` still uses `C:\D\python\202608231533` as a compatibility fallback for Python, Chroma, and `.env` when no overrides are set. Independent developers can point at another complete runtime with `HISTORICAL_GIS_RUNTIME_ROOT` and related variables (see README). That fallback is a local development machine, not a portable default, and it does not make a fresh clone able to rebuild the 7,230-record collection.

## Corpus and Chroma

**VERIFIED:** `backend/app/rag/ingestion/corpus_cli.py` can dry-run a registry or resume ingestion for explicit `--document-id` values into `roman_republic_primary_sources_v2`. `--all-documents` raises `all_documents_not_enabled_for_smoke`. Legacy `--ingest` is disabled. Embeddings in `production_lifecycle.py` are prefixed with `passage: ` and written with chunk metadata. The default model name is `intfloat/multilingual-e5-small`.

**VERIFIED:** `docs/data_sources.md` says raw PDFs are local and excluded from Git. It lists a Polybius PDF (Project Gutenberg attribution in front matter; public domain indicated) and a Livy PDF (original URL not supplied; license not independently verified).

**OBSERVED LOCALLY:** the running collection count is 7,230. The ingestion registry `data/historical_sources/corpus.json` and the incoming source set are not tracked. `git ls-files` shows only `data/historical_sources/raw/caesar/corpus.json` under historical sources.

**MEASURED in V1-D4 (2026-09-29):** Project Gutenberg eBook 10657 no-images EPUB was downloaded from the catalog href `/ebooks/10657.epub.noimages` (resolved `https://www.gutenberg.org/cache/epub/10657/pg10657.epub`). OPF identifier `http://www.gutenberg.org/10657`. SHA-256 `3a038af60bdba9bd66e89127628af65c64c625fd70b2c64c41b430473bd837f2` (422,797 bytes). This is **not** byte-identical to the previously observed local hash `4840578daed0627bffcaad09feedd6322e22f41f3e7465040854f7f88ef97e77`. Isolated parse/chunk with `generic-epub-navigation-v2` / `section-window-v1` (4000/600) produced 35 sections and 306 chunks twice with identical IDs. Isolated PersistentClient collection `v1_d4_caesar_10657_smoke` ingested 306 records twice with matching IDs/text/metadata. Production `roman_republic_primary_sources_v2` remained 7,230. Catalog copyright line: Public domain in the USA. See `scripts/v1_d4_isolated_caesar_rebuild.py`. This does not rebuild the 16-document collection.

**MEASURED in V1-D5 (2026-09-29):** The same isolated rebuild script accepted a second Gutenberg edition via CLI flags. Catalog `https://www.gutenberg.org/ebooks/6386` advertised `/ebooks/6386.epub.noimages` (resolved `https://www.gutenberg.org/cache/epub/6386/pg6386.epub`). OPF identifier `http://www.gutenberg.org/6386`. Title *The Lives of the Twelve Caesars, Volume 01: Julius Caesar*; creator Suetonius. Catalog copyright: Public domain in the USA. SHA-256 `9f8e7289f068bfb7ff369ddf3e7ae36d8b8b03dfbcf953e0e96a7b4455bbc38d` (136,964 bytes), not byte-identical to the previously observed local hash `a29c48055d1c0bebcaa41e4f5576fcbb9c0cafc9d32e0d66673dfa7ebbab528f`. Two parses produced 16 sections and 74 chunks with identical IDs/text/provenance. Two isolated PersistentClient collections `v1_d5_suetonius_6386_smoke` each stored 74 records with matching IDs/text/required provenance. Production `roman_republic_primary_sources_v2` remained 7,230. This still does not rebuild the 16-document collection.

## Pleiades

**VERIFIED in** `scripts/build_pleiades_index.py`:

- dataset version `4.1`, release date `2025-05-28`
- `OFFICIAL_SOURCE` `https://zenodo.org/records/15540082`
- `LICENSE` `CC BY 3.0`
- `EXPECTED_SHA256` `94c5c337d27a07f1a5fa231b6513e40e6d3cf7d5ad7b8c010f8e8bfd9159cd85`
- `EXPECTED_PLACE_COUNT` 41480
- index schema version `3`

The script builds a read-only sqlite index. Those strings were not re-checked against Zenodo in this task (**UNVERIFIED** against the upstream record).

**OBSERVED LOCALLY:** `data/pleiades_v4_1/pleiades_v4_1.sqlite3` exists and is gitignored. Tests look for `data/pleiades_v4_1/pleiades.datasets-v4.1.zip` and skip if it is absent. The zip is not tracked.

The place registry uses representative coordinates and states that they are not exact historical event locations.

**UNKNOWN:** whether the derived sqlite, as distinct from the upstream dataset, has any extra redistribution terms beyond the license string in the build script.

## Natural Earth 1:10m

**VERIFIED in** `data/gis/natural_earth_10m/surface-manifest.json` (local file; the layer GeoJSON and this manifest are gitignored):

- `schema_version` 1, `dataset_family` `Natural Earth`, `dataset_version` `5.1.1`
- files `ne_10m_land.geojson`, `ne_10m_ocean.geojson`, `ne_10m_lakes.geojson`, `ne_10m_minor_islands.geojson`, each with a sha256

`NaturalEarthSurfaceClassifier.from_manifest` checks those hashes. A local load reported `availability=AVAILABLE` (**OBSERVED LOCALLY**).

**UNKNOWN:** acquisition URL and license. None are in the tracked tree. Do not infer them.

`MARITIME_SURFACE_DATA_ROOT` is optional. If it is unset, sea legs do not get a surface. The default launcher does not set it.

## Itiner-e

**VERIFIED in** tracked `data/raw/itiner_e/metadata.json`:

- source name: Itiner-e — The Digital Atlas of Ancient Roads
- source URL: `https://zenodo.org/api/records/17122148/files/itinere_roads.geojson/content`
- accessed `2026-08-28T01:05:50.432929+00:00`
- license `CC BY 4.0`
- citation: Brughmans, T., de Soto, P., Pažout, A. and Bjerregaard Vahlstrup (2024), Itiner-e: the digital atlas of ancient roads, https://itiner-e.org
- local filename `itinere_roads_zenodo_17122148.geojson`
- byte size 78054413
- sha256 `4f8f5cf99f387bab2c509b1bf50143e9a3117691e316f2f46a9bb1d3d4b1cc34`

The GeoJSON is gitignored (`data/raw/itiner_e/*.geojson`). It is **OBSERVED LOCALLY**. This audit did not recompute the sha256 (**UNVERIFIED** against the file bytes).

Roman-road mode is off unless `ROMAN_ROAD_ENABLED=true` and the GeoJSON path exists. The launcher does not enable it. Road geometry is infrastructure, not proof of a historical march.

## Embedding dependencies

**VERIFIED:** production retrieval uses `SentenceTransformerEmbeddingProvider` and `intfloat/multilingual-e5-small` unless settings change. The launcher sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. A clone with an empty cache cannot fetch the model while offline.

**VERIFIED:** `.env.example` can select `AGENT_LLM_PROVIDER=fake` with empty API keys. Live provider keys must stay out of Git.

**VERIFIED:** tests can use `DeterministicHashEmbedding`. That hasher does not reproduce the 7,230-record collection.

## Tracked derived artifacts

`backend/tests/fixtures/pleiades_index_v2_baseline.sqlite3` is tracked and about 41 MB. `test_g6da_pleiades_v3_location_source_preservation.py` and `test_pleiades_index_g3.py` open it as a v2 baseline. The file contains embedded source metadata (Pleiades 4.1 / Zenodo 15540082, schema and importer 2). The exact v2 build command is not in this repository; `build_pleiades_index.py` currently emits schema v3. Available attribution is recorded in `backend/tests/fixtures/PLEIADES_BASELINE_NOTICE.md`. That notice does **not** independently declare redistribution requirements resolved. Do not delete the sqlite as cleanup.

`.gitignore` ignores `.env`, most of `data/`, road and surface GeoJSON, `outputs/`, and logs. It does not ignore `docs/eval/`. Those evaluation notes are untracked and are not part of this document.

## Acquisition and redistribution uncertainties

| Asset | Acquisition recorded in Git | License recorded in Git | Redistribute the local artifact? |
| --- | --- | --- | --- |
| Itiner-e GeoJSON | Yes, metadata URL, size, sha256 | `CC BY 4.0` in metadata | File is not in Git. Whether to commit the GeoJSON is a separate choice; this note does not add it |
| Pleiades 4.1 zip / index | Zenodo URL and sha256 in the build script | Script constant `CC BY 3.0` (**UNVERIFIED** against Zenodo) | Derived sqlite redistribution **UNKNOWN** |
| Natural Earth layers | Not recorded | **UNKNOWN** | Do not commit the layers on the basis of this note |
| Chroma collection | No complete source list | **UNKNOWN** | Do not publish the 7,230-record store as a rebuildable release artifact |
| Embedding weights | Model name only | **UNKNOWN** | Use the upstream model cache; do not commit weights |

Public upstream data is not, by itself, permission to commit every derived file.

## Fresh-clone boundaries

A clone can install Python and frontend dependencies, use the fake provider, and run tests that do not need Chroma or the gitignored datasets.

A clone cannot, from Git alone:

- start the documented launcher;
- serve `roman_republic_primary_sources_v2`;
- resolve places against the Pleiades index;
- enable Roman roads or the maritime surface.

The count 7,230 is an observation of the local collection. It is not a reproducibility promise.

## Historical Source Edition Inventory

Observed on 2026-09-29 from the local registry `data/historical_sources/corpus.json` (not in Git) and from each EPUB’s `dc:identifier`. The registry lists 16 enabled `primary_source` documents. Every registry `source_url` is `https://www.gutenberg.org/`. Every registry `license` is `public_domain`. No source-file checksum is stored. EPUB identifiers below are **LOCAL_EPUB_IDENTIFIER** observations. They were not re-checked on Gutenberg or Archive.org. **VERIFIED_UPSTREAM_LICENSE: NOT PERFORMED.** **REDISTRIBUTION_PERMISSION** is not granted by this note.

| document_id | author / work | edition identifier | source | license |
| --- | --- | --- | --- | --- |
| `appian_roman_history_civil_wars` | Appian, Roman History / Civil Wars | Internet Archive item `appiansromanhist02appiuoft` | Registry URL is the Gutenberg homepage. EPUB id is Archive.org | Recorded `public_domain`. Upstream check not performed |
| `caesar_gallic_civil_wars` | Julius Caesar, Gallic War + Civil War | Gutenberg 10657 | Registry URL is the Gutenberg homepage. EPUB id is `http://www.gutenberg.org/10657` | Recorded `public_domain`. Upstream check not performed |
| `cassius_dio_roman_history_v1` | Cassius Dio, Roman History, volume 1 | Gutenberg 18047 | Same homepage versus EPUB id | Recorded `public_domain`. Upstream check not performed |
| `cassius_dio_roman_history_v2` | Cassius Dio, Roman History, volume 2 | Gutenberg 11607 | Same | Recorded `public_domain`. Upstream check not performed |
| `cassius_dio_roman_history_v3` | Cassius Dio, Roman History, volume 3 | Gutenberg 10162 | Same | Recorded `public_domain`. Upstream check not performed |
| `cassius_dio_roman_history_v4` | Cassius Dio, Roman History, volume 4 | Gutenberg 10883 | Same | Recorded `public_domain`. Upstream check not performed |
| `cassius_dio_roman_history_v5` | Cassius Dio, Roman History, volume 5 | Gutenberg 10890 | Same | Recorded `public_domain`. Upstream check not performed |
| `cassius_dio_roman_history_v6` | Cassius Dio, Roman History, volume 6 | Gutenberg 12061 | Same | Recorded `public_domain`. Upstream check not performed |
| `livy_history_of_rome_books_9_26` | Livy, History of Rome, books 9–26 | Gutenberg 10907 | Same | Recorded `public_domain`. Upstream check not performed |
| `livy_history_of_rome_books_27_36` | Livy, History of Rome, books 27–36 | Gutenberg 12582 | Same | Recorded `public_domain`. Upstream check not performed |
| `livy_history_of_rome_books_37_end` | Livy, History of Rome, books 37–end | Gutenberg 44318 | Same | Recorded `public_domain`. Upstream check not performed |
| `plutarch_parallel_lives` | Plutarch, Parallel Lives | Gutenberg 674 | Same | Recorded `public_domain`. Upstream check not performed |
| `polybius_histories_v1` | Polybius, Histories, volume 1 | Gutenberg 44125 | Same | Recorded `public_domain`. Upstream check not performed |
| `polybius_histories_v2` | Polybius, Histories, volume 2 | Gutenberg 44126 | Same | Recorded `public_domain`. Upstream check not performed |
| `sallust_catiline_jugurthine_war` | Sallust, Catiline + Jugurthine War | Gutenberg 7990 | Same | Recorded `public_domain`. Upstream check not performed |
| `suetonius_julius_caesar` | Suetonius, Julius Caesar | Gutenberg 6386 | Same | Recorded `public_domain`. Upstream check not performed |

Pairings were read from each file’s OPF, not assigned from a list. All 16 local files were present. No pairing was ambiguous.

Recording an identifier does not mean the same file bytes, the same chunk ids, or the same vectors.

| Level | What it identifies | Current evidence |
| --- | --- | --- |
| A. Historical work | Author and work title | Registry `author` and `work` |
| B. Digital edition | A particular EPUB | Local `dc:identifier` only. Registry URL is not an edition URL |
| C. Identical file bytes | The same EPUB bytes | No checksum in the registry. Not established |
| D. Identical sections and chunks | Parser `generic-epub-navigation-v2`, chunker `section-window-v1` | Deterministic only for the same bytes and the same code |
| E. Identical embeddings | `intfloat/multilingual-e5-small`, prefix `passage: `, L2-normalized | Model revision is not pinned |
| F. The existing Chroma collection | `roman_republic_primary_sources_v2`, observed count 7,230 | Not a rebuild contract |
