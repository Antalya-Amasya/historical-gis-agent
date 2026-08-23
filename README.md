# Historical Military GIS Agent

An incremental research and demonstration project for explaining historical military movements with explicitly separated historical evidence, geographic tools, agent reasoning, and map visualization.

## Phase 0 status

Phase 0 defines stable API models and provides a no-key Mock Agent demonstration. It includes a FastAPI backend, an independently runnable Geography MCP-compatible HTTP service, a React frontend shell, RAG/session abstractions, and automated tests. Route inference, vector retrieval, and map rendering begin in later phases.

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

Open `http://127.0.0.1:5173`. With the default environment it returns a deterministic Mock Agent response for Hannibal's Alpine crossing and has no external API dependency.

## Test

```powershell
.\.venv\Scripts\Activate.ps1
pytest backend\tests geography_mcp\tests -q
```

## Current boundaries

- `POST /api/v1/agent/chat` is ordinary HTTP; SSE is intentionally deferred.
- The Geography MCP server is independently runnable over stdio. The parallel HTTP adapter (`uvicorn geography_mcp.server:app --port 8001`) exists only for local API and integration testing.
- The mock agent does not claim a historical route. It only returns a bounded event/place result and uncertainty notice.
- ChromaDB, live Pleiades queries, real LLM providers, route construction, and MapLibre map layers are out of scope for Phase 0.
