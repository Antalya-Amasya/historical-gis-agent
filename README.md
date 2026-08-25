# Historical Military GIS Agent

An evidence-constrained historical GIS demonstration. It separates source-backed historical claims, audited place coordinates, deterministic route reconstruction, and map presentation.

## Demo setup

Prerequisites: Python 3.11+, Node.js 20+, and the local dependencies listed below. No LLM key is required for the deterministic bounded-Agent demo: `.env.example` configures `AGENT_LLM_PROVIDER=fake`.

```powershell
Copy-Item .env.example .env
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

The Hannibal Agent demo requires the prepared local semantic index at `data/chroma_semantic` (`historical_primary_sources_semantic`). It is intentionally Git-ignored because it is generated local corpus data. If this directory is absent, the Caesar read-only presentation still works, but the Hannibal Agent request cannot retrieve its evidence; restore the prepared demo data before presenting Hannibal.

Start the backend and frontend in separate terminals:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000
```

```powershell
cd frontend
pnpm install
pnpm run dev
```

Open `http://127.0.0.1:5173/`. The root page is the sole presentation entry; legacy phase/smoke URLs redirect back to it.

## Fixed demo scenarios

- **Demo A — Hannibal crossing Alps**: enter `展示汉尼拔翻越阿尔卑斯进入意大利的路线`, then select **请求路线**. Confirm `historical_route` intent, `hannibal_italy_campaign`, source-backed markers, and a terrain-aware LineString.
- **Demo B — Caesar conquest of Gaul**: select **Caesar conquest of Gaul**, then select **加载路线**. This deliberately loads the reviewed read-only Caesar presentation, including Book I / VII and Alesia metadata.
- **Demo C — Alpine-pass uncertainty**: ask `告诉我汉尼拔准确经过哪个阿尔卑斯山口`. The system must not identify a uniquely established pass from its current Evidence. State the uncertainty and keep the route presentation evidence-grounded.

## Architecture

```text
LLM / bounded Agent
  → interprets a permitted request; the reviewed campaign ontology selects a structured entity, route type, and route context
RAG
  → returns primary-source Evidence only
Registry
  → holds manually reviewed campaign/event configuration, ontology metadata, and presentation metadata
Geography MCP
  → resolves audited geographic data; it is the coordinate boundary for HistoricalRoute extraction
A* + terrain/cost model
  → produces deterministic, algorithmic candidate connections between supplied anchors
GIS presentation
  → renders backend GeoJSON, markers, source references, and knowledge panels
```

The frontend never geocodes a waypoint, invents a location, builds a route, or generates historical text. It consumes backend presentation DTOs and only displays supplied metadata.

## Demo Script

**30-second opening:** “This is an evidence-grounded historical route reconstruction agent. The language model can understand a permitted request, but source Evidence, reviewed registries, Geography MCP, and deterministic terrain-aware search keep historical claims, coordinates, and route geometry auditable.”

1. Select the campaign and enter the fixed question.
2. Explain the structured Agent intent and Evidence boundary.
3. Load the backend presentation and point out source labels, Book/Chapter, and the evidence count on a marker popup.
4. Open a knowledge panel, then toggle the route line to show that the map only consumes backend GeoJSON.
5. Close with the route limitation: it is a terrain-aware schematic candidate, not a precise historical daily march.

## Demo Workflow

```text
User question
  → Agent intent detection
  → Historical campaign ontology (reviewed entity, type, period, and parent context)
  → Historical Evidence retrieval
  → Evidence validation
  → Campaign registry / reviewed anchors
  → Terrain-aware reconstruction
  → GeoJSON presentation
```

The Hannibal demo uses the bounded Agent response. The Caesar selector deliberately loads an already reviewed, read-only campaign presentation; it does not infer a campaign from a free-text query.

## Design Philosophy

- **LLM** understands permitted user intent; it does not create coordinates or route geometry.
- **RAG** supplies historical evidence.
- **Registry** holds manually reviewed facts and configuration.
- **MCP** supplies bounded geographic capabilities.
- **A\*** performs deterministic spatial search over supplied terrain cells.
- **GIS** visualizes backend-provided GeoJSON and metadata only.

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest backend\tests geography_mcp\tests -q

cd frontend
pnpm test
pnpm run build
```

## Limitations

- A displayed historical route is not an exact GPS trace or a claim about each day's march.
- Alpine pass locations and some ancient geographic locations may remain debated; representative coordinates are labeled as such.
- Coordinates can represent a region, river, or place rather than an exact crossing point.
- DEM data changes terrain cost only. It does not establish historical facts, validate a campaign sequence, or create evidence.
- External links in knowledge panels are display-only; the application does not fetch or validate them.
