# Historical Military GIS Agent

Evidence-grounded Historical GIS workspace for query-driven movement reconstruction, place resolution, deterministic route assembly, and map presentation.

## Project purpose

Historical GIS / evidence-grounded route reconstruction agent. The system answers natural-language questions about Roman Republic-era campaigns and movements, retrieves primary-source evidence from a local vector store, extracts movement claims when supported, resolves ancient places against a local geography authority, assembles routes with explicit full / partial / no-route outcomes, and presents results with provenance on an interactive map when geometry is available.

This is not a general historical Q&A oracle. Answers and routes are bounded by retrieved evidence and conservative admission rules.

## Architecture

```text
Natural-language query
  → Retrieval (Chroma + hybrid reranking)
  → Evidence selection
  → Event extraction
  → Geography authority resolution
  → Relation construction + admission
  → Route assembly
  → Presentation (map / evidence / limitations)
```

Backend (`backend/`) owns retrieval, agent orchestration, route logic, and APIs. Frontend (`frontend/`) is the query workspace UI. Shared runtime data and Python environment live outside this Git worktree at `C:\D\python\202608231533` on the canonical Windows development machine.

## V1 capabilities

- Evidence-grounded historical movement reconstruction
- Exact / eligible geographic anchors via local geography data
- Explicit route result states: `FULL_ROUTE`, `PARTIAL`, `NO_ROUTE`, `ERROR`
- Provenance and fail-closed safety when evidence is insufficient
- Frontend map visualization when presentation geometry is attached
- Optional Roman-road infrastructure candidates (disabled by default)

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

### Manual startup

If you need separate terminals, start in this order:

1. Chroma (shared persistence):

```powershell
C:\D\python\202608231533\.venv\Scripts\chroma.exe run `
  --path C:\D\python\202608231533\data\chroma_server_roman_republic_v2 `
  --host 127.0.0.1 --port 8002
```

2. Backend (from this worktree):

```powershell
$env:PYTHONPATH = "C:\D\python\historical-gis-cursor"
C:\D\python\202608231533\.venv\Scripts\python.exe -m uvicorn backend.app.main:app `
  --host 127.0.0.1 --port 8000 `
  --env-file C:\D\python\202608231533\.env `
  --app-dir C:\D\python\historical-gis-cursor
```

3. Frontend:

```powershell
Set-Location frontend
pnpm install --frozen-lockfile
pnpm run dev --port 5173 --strictPort
```

Open http://127.0.0.1:5173/.

## Runtime requirements

This project assumes the current canonical Windows layout:

| Resource | Path |
|---|---|
| Git worktree (code) | `C:\D\python\historical-gis-cursor` |
| Shared Python venv | `C:\D\python\202608231533\.venv` |
| External `.env` | `C:\D\python\202608231533\.env` |
| Chroma persistence | `C:\D\python\202608231533\data\chroma_server_roman_republic_v2` |
| Chroma collection | `roman_republic_primary_sources_v2` |
| Geography data | `C:\D\python\202608231533\data\geography\` |
| Embedding model | `intfloat/multilingual-e5-small` (cached locally; launcher sets offline HF flags) |
| Frontend deps | `frontend/node_modules` via `pnpm install` |
| Runtime logs | `%LOCALAPPDATA%\HistoricalGISAgent\runtime\` |

Prerequisites: Python 3.11+, Node.js 20+, pnpm, packages in `backend/requirements.txt`.

Copy `.env.example` to the shared runtime `.env` and configure provider keys there. The launcher reads `C:\D\python\202608231533\.env`; it does not copy secrets into the worktree.

Optional Roman-road capability before backend startup:

```powershell
$env:ROMAN_ROAD_ENABLED = "true"
$env:ROMAN_ROAD_GEOJSON_PATH = "C:\D\python\202608231533\data\raw\itiner_e\itinere_roads_zenodo_17122148.geojson"
```

Roman-road geometry is infrastructure evidence, not proof of historical movement.

## Example query

Reproducible route-control example (direct pipeline / unit tests; live agent retrieval may vary):

```text
Trace the route from Oricum to Brundisium.
```

Gold evidence passage (Caesar, Civil War XXIII): Libo sailed from Oricum to Brundisium. Named variant that aligns episode subject admission:

```text
Trace Libo's route from Oricum to Brundisium.
```

Many other natural-language movement questions are supported when retrieval and admission rules allow; the controls above are regression anchors, not the only supported queries.

Non-route example:

```text
What happened at the Battle of Cannae?
```

## Result states

Top-level API field: `route_result_status`

| Status | Meaning |
|---|---|
| `FULL_ROUTE` | Ordered historical route with sufficient evidence-backed anchors |
| `PARTIAL` | Some route structure or presentation exists, but the result is not a complete attested route |
| `NO_ROUTE` | Evidence insufficiency or admission failure; fail-closed, not a transport error |
| `ERROR` | Provider / tool / processing failure distinct from historical insufficiency |
| `null` | Non-route answer request |

Frontend renders status banners, clears stale map state on new requests, shows map only for `FULL_ROUTE` / `PARTIAL` when presentation geometry exists, and keeps evidence accessible for partial / no-route outcomes.

## Known limitations

- Republic-wide recall is incomplete; many movement queries miss gold evidence under bounded retrieval budgets.
- Some evidence is lost under hybrid retrieval caps even when present in the corpus.
- Ambiguous or regional geography often yields `PARTIAL` or `NO_ROUTE`.
- Event completeness and relation admission remain conservative (`NO_EPISODE_RELEVANT_LEGACY_CLAIMS`, partial components, unresolved places/times).
- Actor grounding can remain `UNKNOWN`.
- Broad G7 benchmark coverage is research/evaluation scope, not a V1 release requirement.
- Live agent path depends on LLM provider availability (configured via shared `.env`).

## Safety / interpretation

- Reconstructed GIS geometry is a candidate presentation layer, not automatically attested historical path evidence.
- Evidence insufficiency may intentionally produce `NO_ROUTE`; that is a valid outcome, not a bug by itself.
- Regional centroids and representative anchor points are not treated as exact daily march coordinates.
- Roman-road and terrain segments show algorithmic reconstruction constraints; failed legs remain explicit gaps.
- Do not overclaim historical accuracy from map lines alone.

## Tests

```powershell
# Backend focused regressions
$env:PYTHONPATH = "C:\D\python\historical-gis-cursor"
C:\D\python\202608231533\.venv\Scripts\python.exe -m pytest `
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
| `docs/eval/` | Evaluation reports (local / optional) |
| `outputs/` | Local evaluation/runtime artifacts (git-ignored) |

Primary-source corpus, Chroma indexes, embedding cache, and shared `.env` are local runtime assets under `C:\D\python\202608231533` and are deliberately not committed.
