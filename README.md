# Historical Military GIS Agent

An incremental research and demonstration project for explaining historical military movements with explicitly separated historical evidence, geographic tools, agent reasoning, and map visualization.

## Phase 1 status

Phase 1 keeps the no-key Mock Agent and adds the audited HistoricalPlace → Backend API → MapLibre GL JS marker chain. The FastAPI backend, independently runnable Geography MCP-compatible HTTP service, and React frontend share a small local Pleiades-backed demo repository. The map renders markers and source-rich popups, but does not draw or infer a historical route.

## Prerequisites

- Python 3.11 or later
- Node.js 20 or later

## Run (three terminals)

```powershell
Copy-Item .env.example .env
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
# Standard MCP (stdio; used by the Agent in a later phase)
python -m geography_mcp.mcp_server
```

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000
```

```powershell
cd frontend
pnpm install
pnpm run dev
```

Open `http://127.0.0.1:5173`. With the default environment it returns a deterministic Mock Agent response for Hannibal's Alpine crossing and renders the returned HistoricalPlace markers using MapLibre. The base map is the keyless MapLibre demo style; the demo does not use geocoding for ancient places.

## Test

```powershell
.\.venv\Scripts\Activate.ps1
pytest backend\tests geography_mcp\tests -q
```

## Current boundaries

- `POST /api/v1/agent/chat` is ordinary HTTP; SSE is intentionally deferred.
- The Geography MCP server is independently runnable over stdio. The parallel HTTP adapter (`uvicorn geography_mcp.server:app --port 8001`) exists only for local API and integration testing.
- The mock agent does not claim a historical route. It only returns a bounded event/place result and uncertainty notice.
- The two Phase 1 demo places are hand-curated local records, with coordinates copied from their Pleiades representative points: Carthago ([314921](https://pleiades.stoa.org/places/314921)) and Carthago Nova ([265849](https://pleiades.stoa.org/places/265849)). Each API record carries `source`, `source_id`, `source_url`, `confidence`, and `uncertain`; no LLM or geocoder supplies coordinates.
- ChromaDB, live Pleiades queries, real LLM providers, route construction, candidate routes, route scoring, and environmental analysis remain out of scope.
