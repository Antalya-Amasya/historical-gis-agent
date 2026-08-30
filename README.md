# Historical Military GIS Agent

An evidence-constrained, query-driven Historical GIS workspace. It keeps historical evidence, place resolution, deterministic route reconstruction, Roman-road infrastructure candidates, and map presentation as separate auditable layers.

## Local setup

Prerequisites: Python 3.11+, Node.js 20+, the packages in `backend/requirements.txt`, and the local data described below.

```powershell
Copy-Item .env.example .env
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

The primary-source corpus is local generated data and deliberately Git-ignored. Production retrieval defaults to Chroma on `127.0.0.1:8002`, collection `roman_republic_primary_sources_v2`, using `intfloat/multilingual-e5-small`.

## Run the query-driven workspace

On the canonical Windows development machine, double-click
`start_historical_gis.bat` to start the existing Chroma store, backend, and
frontend, then open the application in a browser. The launcher uses the shared
runtime and external `.env`; it does not copy secrets or rebuild the collection.

For manual startup, use the commands below.

Start Chroma with the Roman Republic v2 persistence directory:

```powershell
.\.venv\Scripts\chroma.exe run --path data\chroma_server_roman_republic_v2 --host 127.0.0.1 --port 8002
```

To enable the optional Roman-road capability before starting the backend:

```powershell
$env:ROMAN_ROAD_ENABLED = "true"
$env:ROMAN_ROAD_GEOJSON_PATH = "data/raw/itiner_e/itinere_roads_zenodo_17122148.geojson"
```

The backend loads the road graph once during application startup. If enabled data is absent, startup fails explicitly; it does not silently fall back to terrain routing.

```powershell
.\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --port 8000
```

In a separate terminal:

```powershell
Set-Location frontend
pnpm install
pnpm run dev
```

Open `http://127.0.0.1:5173/`.

The only production frontend entry is `index.html` → `src/main.tsx` → `QueryApp`. A user asks a natural-language question; the Agent answer is always primary. When evidence supports a `HistoricalRoute`, the backend may attach a GIS presentation. No presentation is a valid answer state, not a request failure.

## Production chain

```text
Natural-language query
  → HistoricalGisAgent
  → RAG primary-source Evidence
  → evidence-constrained MovementClaim / HistoricalRoute when supported
  → deterministic GIS reconstruction
  → optional Roman-road candidate orchestration
  → COMPLETE / PARTIAL / UNAVAILABLE presentation
  → map, knowledge, and provenance display
```

Roman-road geometry is infrastructure evidence, not proof of historical movement. Failed legs remain explicit gaps; the application does not join them with terrain, straight-line, or LLM-generated geometry.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests geography_mcp\tests -q

Set-Location frontend
pnpm test
pnpm run build
```

## Limitations

- Historical evidence is necessary but does not establish an exact daily march track.
- Ancient place coordinates can be representative points or regional geometries and retain uncertainty.
- Roman-road candidates do not prove use of a road at a particular historical time.
- Terrain changes algorithmic cost only; it does not create evidence or validate a movement claim.
- A query without a safely grounded route remains an answer-only result.
