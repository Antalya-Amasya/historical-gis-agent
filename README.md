# Historical Military GIS Agent

Evidence-grounded **Historical GIS Agent** for Roman Republic-era movement questions: primary-source retrieval determines **what** movement happened; configured GIS layers simulate one **plausible how** for map presentation.

**Not claimed:** exact historical itinerary recovery, archaeological certainty, complete Mediterranean coverage, or a production-grade “historical truth engine.”

See the [V1 interview demo walkthrough](docs/v1_interview_demo.md) for official demo queries, a 3–5 minute script, and a pre-interview checklist.

## Project purpose

Historical GIS / evidence-grounded route reconstruction agent. The system answers natural-language questions about Roman Republic-era campaigns and movements, retrieves primary-source evidence from a local vector store, extracts movement claims when supported, resolves ancient places against a local geography authority, assembles routes with explicit full / partial / no-route outcomes, and presents results with provenance on an interactive map when geometry is available.

This is not a general historical Q&A oracle. Answers and routes are bounded by retrieved evidence and conservative admission rules.

## Historical facts vs simulation (contract)

| Historical facts (evidence-backed) | Simulation (algorithmic presentation) |
| --- | --- |
| Actor and movement claims admitted from retrieved passages | Representative Pleiades coordinates as **anchors only** |
| Endpoints and travel mode when extraction and authority allow | Itiner-e Roman-road geometry |
| Explicit `LAND` / `SEA` / `UNKNOWN` historical mode when supported | SRTM terrain geometry and slope plausibility |
| Evidence IDs and provenance in the UI | Simulated coastal access connectors |
| `route_result_status` (`FULL_ROUTE`, `PARTIAL`, `NO_ROUTE`, …) | Natural Earth–validated direct-water edges **or explicit failed gaps** |
| | Algorithm-selected simulation mode for presentation |

**Simulation geometry is not historical evidence.** Map lines illustrate one plausible reconstruction under stated constraints; they must not be read as attested paths.

## Architecture

```text
Natural-language query
  → Retrieval (Chroma + bounded hybrid reranking)
  → Historical evidence selection
  → Actor / movement / endpoint authority (extraction + admission)
  → Pleiades place resolution (representative anchors)
  → HistoricalRoute assembly (WHAT)
  → GIS reconstruction (HOW) — optional, separate from claims
  → Frontend map + evidence + route details
```

**GIS reconstruction (when enabled):**

| Mode | Behavior |
| --- | --- |
| **LAND** | Itiner-e Roman-road network preferred → **SRTM** terrain fallback when road access or a connected path is unavailable |
| **SEA** | Explicitly simulated coastal access where needed → Natural Earth ocean-only direct-water validation; otherwise **failed_gap** |
| **UNKNOWN** | Simulation may compare plausible alternatives for display without changing the historical movement claim |

Backend (`backend/`) owns retrieval, agent orchestration, route logic, and APIs. Frontend (`frontend/`) is the query workspace UI. Shared runtime data and Python environment may live outside this Git worktree; see [data provenance](docs/v1_data_provenance.md) and launcher overrides below.

## V1 capabilities

- Evidence-grounded historical movement reconstruction
- Resolved historical places with eligible geographic simulation anchors
- Explicit route result states: `FULL_ROUTE`, `PARTIAL`, `NO_ROUTE`, `ERROR`
- Provenance and fail-closed safety when evidence is insufficient
- Frontend map visualization when presentation geometry is attached
- The standard launcher activates the prepared local Roman-road, terrain, and maritime assets (see health check)

## GIS and map data assets

| Asset | Role in V1 | Participates in historical reasoning? |
| --- | --- | --- |
| **Pleiades** | Historical place identity and representative coordinates | Yes (place resolution only; coordinates are simulation anchors) |
| **Itiner-e** | Ancient-road routing **prior** for land legs | No (infrastructure candidate for GIS only) |
| **SRTM** (HGT mosaic) | Terrain slope / plausibility; land fallback routing | No |
| **Natural Earth 1:10m** | Land / ocean / lake topology for maritime validation | No |
| **OpenStreetMap** (frontend basemap tiles) | Map background for the UI | **No** — display only; not used in retrieval, admission, or route calculation |

Provenance and local paths: [docs/v1_data_provenance.md](docs/v1_data_provenance.md).

## Interview demos (V1)

Two primary live demos — full script in [docs/v1_interview_demo.md](docs/v1_interview_demo.md):

| Demo | Query | Historical | Simulation highlight |
| --- | --- | --- | --- |
| **A — Land** | `Reconstruct Quintus Fabius Pictor's return journey after consulting the oracle.` | Pictor; Delphi → Rome | Itiner-e (+ SRTM fallback) |
| **B — Sea** | `Trace Libo's route from Oricum to Brundisium.` | Libo; Oricum/Orikon → Brundisium; SEA | Coastal access + Natural Earth water edge when valid |

Internal regression id for Demo A: **G7-A02**.

## Quick start

### One-command launcher (recommended)

From this worktree:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_historical_gis.ps1
```

Or double-click `start_historical_gis.bat`.

Use `-NoBrowser` to skip opening the browser:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_historical_gis.ps1 -NoBrowser
```

Expected services after startup:

| Service | URL / port |
|---|---|
| Frontend | http://127.0.0.1:5173 |
| Backend | http://127.0.0.1:8000 |
| Chroma | http://127.0.0.1:8002 |

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The prepared standard runtime must report all four GIS fields as **`ACTIVE`**. The backend reports **`UNAVAILABLE`** when a provider is absent, and the launcher refuses readiness if a required asset is unavailable:

| Field | Meaning |
| --- | --- |
| `pleiades` | Pleiades sqlite index reachable |
| `srtm` | SRTM HGT directory configured and readable |
| `itiner_e` | Roman-road GeoJSON loaded (`ROMAN_ROAD_ENABLED=true`) |
| `natural_earth` | Maritime surface manifest validated (`MARITIME_SURFACE_DATA_ROOT`) |

Example shape (provider fields vary): `status`, `agent`, `pleiades`, `srtm`, `itiner_e`, `natural_earth`. No API secrets are returned.

### Manual startup

If you need separate terminals, run this preparation block from the repository root in each backend/Chroma PowerShell terminal. It reuses the launcher's configurable paths and GIS defaults:

```powershell
. .\scripts\start_historical_gis.ps1
Resolve-HistoricalGisLauncherConfig
Enable-DefaultGisAssets
Assert-OptionalGisDependencies
```

Then start in this order:

1. Chroma (shared persistence):

```powershell
& $chromaExe run `
  --path $chromaData `
  --host 127.0.0.1 --port 8002
```

2. Backend (from this worktree):

```powershell
$env:PYTHONPATH = $projectRoot
& $pythonExe -m uvicorn backend.app.main:app `
  --host 127.0.0.1 --port 8000 `
  --env-file $externalEnv `
  --app-dir $projectRoot
```

3. Frontend:

```powershell
Set-Location frontend
pnpm install --frozen-lockfile
pnpm run dev --port 5173 --strictPort
```

Open http://127.0.0.1:5173/.

## Runtime requirements

The original Windows machine uses the layout below. These are documented examples; another prepared runtime can use the launcher overrides:

| Resource | Path |
|---|---|
| Git worktree (code) | `C:\D\python\historical-gis-cursor` |
| Shared Python venv | `C:\D\python\202608231533\.venv` |
| External `.env` | `C:\D\python\202608231533\.env` |
| Chroma persistence | `C:\D\python\202608231533\data\chroma_server_roman_republic_v2` |
| Chroma collection | `roman_republic_primary_sources_v2` |
| Pleiades index | Worktree `data/pleiades_v4_1/pleiades_v4_1.sqlite3`, or `PLEIADES_GAZETTEER_PATH` |
| Terrain | `DEM_HGT_DIR` in the private runtime environment (original machine: `C:\data\srtm-hgt`) |
| Embedding model | `intfloat/multilingual-e5-small` (cached locally; launcher sets offline HF flags) |
| Frontend deps | `frontend/node_modules` via `pnpm install` |
| Runtime logs | `%LOCALAPPDATA%\HistoricalGISAgent\runtime\` |

Prerequisites: Python 3.11+, Node.js 20+, pnpm, packages in `backend/requirements.txt`.

Copy `.env.example` to a private runtime `.env` (the original machine uses `C:\D\python\202608231533\.env`) and configure provider keys there. The launcher never copies secrets into the worktree.

Launcher path overrides (explicit environment → existing original-machine layout → error). See [data provenance](docs/v1_data_provenance.md). A fresh clone still cannot rebuild the 7,230-record collection.

| Variable | Purpose |
|---|---|
| `HISTORICAL_GIS_RUNTIME_ROOT` | Shared runtime directory containing `.venv`, `.env`, and Chroma persistence |
| `HISTORICAL_GIS_PYTHON` | Python executable |
| `HISTORICAL_GIS_CHROMA_EXE` | Chroma executable |
| `HISTORICAL_GIS_ENV_FILE` | Private environment file (may live outside the repo) |
| `HISTORICAL_GIS_CHROMA_DATA` | Chroma persistence directory for `roman_republic_primary_sources_v2` |
| `HISTORICAL_GIS_LOG_ROOT` | Optional launcher log directory (default `%LOCALAPPDATA%\HistoricalGISAgent\runtime`) |
| `PLEIADES_GAZETTEER_PATH` | Override the default worktree Pleiades sqlite path |
| `ROMAN_ROAD_ENABLED` / `ROMAN_ROAD_GEOJSON_PATH` | Launcher defaults to enabled and the existing worktree Itiner-e GeoJSON; explicit overrides take precedence |
| `MARITIME_SURFACE_DATA_ROOT` | Override the default worktree `data/gis/natural_earth_10m` manifest-backed surface |
| `DEM_HGT_DIR` | Existing SRTM HGT directory, supplied through the runtime environment |

The launcher activates the existing worktree GIS assets and validates required health fields. Explicit path overrides never fall back to the original-machine runtime. `ROMAN_ROAD_ENABLED=0` explicitly disables roads. A fresh clone needs the external corpus, embedding cache, Pleiades index, SRTM tiles, road GeoJSON, and Natural Earth layers before it can run the interview demos. SRTM supplies terrain, never water authority; Roman roads supply routing priors, never proof of historical usage. OSM is a remote frontend basemap; no local basemap tiles are provided.

## Other queries

Use the two official interview queries above for the prepared demo. Other movement questions are supported when retrieval and admission rules allow; live results depend on retrieved evidence and provider availability.

Non-route example:

```text
What happened at the Battle of Cannae?
```

## Result states

Top-level API field: `route_result_status`

| Status | Meaning |
|---|---|
| `FULL_ROUTE` | Complete ordered historical route under the canonical route contract; does not certify complete GIS geometry or an exact itinerary |
| `PARTIAL` | Partial historical route structure or presentation; inspect diagnostics and any failed geometry gaps |
| `NO_ROUTE` | Evidence insufficiency or admission failure; fail-closed, not a transport error |
| `ERROR` | Provider / tool / processing failure distinct from historical insufficiency |
| `null` | Non-route answer request |

Frontend renders status banners, clears stale map state on new requests, shows map only for `FULL_ROUTE` / `PARTIAL` when presentation geometry exists, and keeps evidence accessible for partial / no-route outcomes.

## Known limitations

- The **exact historical itinerary** is not reconstructed; GIS shows one plausible path under constraints.
- **Representative Pleiades coordinates** may not coincide with exact event sites or harbors.
- Most **Itiner-e** geometry is ancient **infrastructure prior**, not proof a figure used those segments.
- **Coastal access** segments may be **simulated** for visualization when anchors are inland.
- No **vegetation**, **water-source**, **season**, **logistics**, or campaign supply modeling in V1.
- **Wind and current** are not modeled for sea legs.
- Republic-wide retrieval recall is incomplete under bounded budgets; some gold passages remain outside top-k.
- Live agent behavior depends on configured LLM provider availability (private `.env`).
- Broad G7 benchmark coverage is research scope, not a V1 interview requirement.

## Interview talking points

Short notes for the project owner (expanded checklist in [docs/v1_interview_demo.md](docs/v1_interview_demo.md)):

- Retrieval alone does not produce admissible routes — extraction, actor identity, and endpoint authority are separate stages.
- GIS never rewrites historical claims; it consumes an already admitted `HistoricalRoute`.
- Roman roads rank paths on a digital atlas of ancient roads, not on “what Caesar actually walked.”
- Representative coordinates are simulation anchors only; the system must not silently move settlements to the ocean to draw sea lines.
- **`UNKNOWN`** travel mode stays available when evidence does not force LAND/SEA.
- **`failed_gap`** presentation is preferable to fabricating geometry.

## Technical debt (non-blocking)

Maintained for engineers; not part of the live demo narrative:

- Internal **`FULL_ROUTE` / `PARTIAL_ROUTE`** taxonomy vs API `route_result_status` naming drift in older tests/docs.
- `test_route_provenance.py` has three stale synthetic expectations: generic army fixtures yield no canonical route, and the disabled legacy path no longer emits the expected rejection reason. Reconcile these fixtures separately; do not restore legacy authority to make them pass.
- Two `test_g4g3_terminal_grounding_closure.py` cases (`test_successful_submit_grounded_answer_unchanged`, `test_successful_submit_after_route_build_does_not_trigger_closure`) expect submitted prose to survive despite an absent canonical route. Current behavior returns grounded insufficiency; these are stale test contracts.
- When an admitted route exists and **`submit_grounded_answer`** is omitted, terminal closure now generates an evidence-linked route-state summary. `route_terminal_submission_missing` remains a diagnostic; it does not erase the route. Absent routes still return insufficiency.
- Multimodal LAND/SEA thresholds and maritime validation heuristics are **V1 policy**, not settled historiography.

## Safety / interpretation

- Reconstructed GIS geometry is a candidate presentation layer, not automatically attested historical path evidence.
- Evidence insufficiency may intentionally produce `NO_ROUTE`; that is a valid outcome, not a bug by itself.
- Regional centroids and representative anchor points are not treated as exact daily march coordinates.
- Roman-road and terrain segments show algorithmic reconstruction constraints; failed legs remain explicit gaps.
- Do not overclaim historical accuracy from map lines alone.

## Tests

```powershell
# Backend focused regressions
$env:PYTHONPATH = (Get-Location).Path
# Resolve launcher paths as in Manual startup before using $pythonExe.
& $pythonExe -m pytest `
  backend/tests/test_v1c_route_result_status.py `
  backend/tests/test_v1b3_same_movement_direct_subject_admission.py `
  backend/tests/test_g4b_route_components.py -q

# Frontend
Set-Location frontend
pnpm test
pnpm run build
```

## Repository layout

| Path | Role |
|---|---|
| `backend/` | FastAPI app, agent, retrieval, route pipeline |
| `frontend/` | Vite + React query workspace (`index.html` → `src/main.tsx`) |
| `geography_mcp/` | Local ancient-place authority service + tests |
| `scripts/start_historical_gis.ps1` | Production launcher |
| `start_historical_gis.bat` | Windows wrapper for launcher |
| `scripts/test_launcher_paths.ps1` | Launcher path override regression |
| `scripts/test_launcher_identity.ps1` | Launcher process identity regression |
| `docs/v1_interview_demo.md` | Interview demo script and checklist |
| `docs/v1_data_provenance.md` | Git vs local runtime assets |
| `docs/v1_rc1_runtime_smoke.md` | Bounded runtime smoke notes |
| `docs/eval/` | Evaluation reports (local / optional) |
| `outputs/` | Local evaluation/runtime artifacts (git-ignored) |

Primary-source corpus, Chroma indexes, embedding cache, and shared `.env` are local runtime assets under `C:\D\python\202608231533` and are deliberately not committed.
