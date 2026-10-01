# V1 interview demo package

Recorded for technical interviews on branch `agent/cursor`. This document is interviewer-facing; it does not change product behavior.

## Official demo cases

### Demo A — land (G7-A02 / Pictor)

**Query (paste exactly):**

```text
Reconstruct Quintus Fabius Pictor's return journey after consulting the oracle.
```

| Layer | Content |
| --- | --- |
| Historical | Actor: **Quintus Fabius Pictor**; movement: return from oracle consultation; endpoints: **Delphi → Rome** (evidence-backed) |
| Simulation | **Itiner-e** Roman-road reconstruction between representative Pleiades anchors; **SRTM** terrain fallback when road network cannot connect |
| Say aloud | Historical movement is evidence-backed; map endpoints are **representative simulation anchors**, not attested daily halts; the drawn road path is a **plausible reconstruction**, not the literal historical itinerary |

### Demo B — sea (Libo)

**Query (paste exactly):**

```text
Trace Libo's route from Oricum to Brundisium.
```

| Layer | Content |
| --- | --- |
| Historical | Actor: **Libo**; endpoints: **Oricum / Orikon → Brundisium**; mode: **SEA** when extraction supports it |
| Simulation | **Simulated coastal access** connectors where needed; **Natural Earth** land/sea validation; **direct-water edge** geometry only when endpoint surface rules pass |
| Say aloud | The sea movement is evidence-backed; **exact embarkation points are not** attested; coastal access segments are **generated for visualization**; water lines are algorithmic, not primary-source geometry |

Representative settlement coordinates may use explicitly labeled simulated coastal access. If no valid access pair or ocean-only water path can be constructed, the UI retains an explicit **`failed_gap`** instead of a water line.

## 3–5 minute walkthrough script

1. **User query** — Submit Demo A or B in the frontend chat. State that the user asks for a **historical movement**, not a turn-by-turn GPS trace.
2. **Evidence retrieval** — Agent calls bounded retrieval against the local Chroma collection (`roman_republic_primary_sources_v2`). Mention hybrid lexical + semantic ranking and coverage caps; no evidence ⇒ no route claim.
3. **Historical event extraction** — Passages become structured events: actor, movement relation, travel mode when supported, evidence IDs. Authority rules drop unsupported claims.
4. **Place resolution** — Named places resolve through **Pleiades** (representative coordinates). Coordinates anchor simulation; they are not proof of exact event sites.
5. **Route simulation** — After a **HistoricalRoute** is admitted, GIS runs separately: **LAND** — Itiner-e preferred, SRTM fallback; **SEA** — Natural Earth surface check, then direct-water or explicit gap; **UNKNOWN** mode may compare plausible alternatives without rewriting the historical claim.
6. **Frontend visualization** — Map shows basemap (OSM tiles for display only), historical anchor points, Roman-road / terrain / water segments, route details, and evidence panel. Labels distinguish historical vs simulated layers where rendered.
7. **Historical fact vs simulation** — Close with: **WHAT** moved (evidence) is separate from **HOW** it might have moved (GIS). Simulation geometry is **not** historical evidence.

## Pre-interview checklist

These demos require the prepared external runtime and GIS assets described in [data provenance](v1_data_provenance.md); a fresh Git clone alone does not supply them. The standard launcher activates all four GIS assets.

- [ ] Chroma listening on `127.0.0.1:8002` with production collection loaded (~7,230 records on canonical machine)
- [ ] Backend on `127.0.0.1:8000`; `GET /health` returns `status: ok`
- [ ] Health shows **`pleiades: ACTIVE`**, **`srtm: ACTIVE`**, **`itiner_e: ACTIVE`**, **`natural_earth: ACTIVE`**
- [ ] Frontend dev server on `127.0.0.1:5173`
- [ ] Demo A (Pictor / G7-A02 query) returns route state with map presentation when GIS enabled
- [ ] Demo B (Libo query) returns **`PARTIAL`** or better with presentation (water line or explicit failed gap)
- [ ] Map renders (lines and/or anchors); route details drawer opens
- [ ] Evidence panel lists retrieved passages with IDs
- [ ] LLM provider configured in private `.env` (not in Git)

Start stack:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_historical_gis.ps1
```

## Screenshot guidance

No demo screenshots are checked into this repository (**do not fabricate**). Capture manually before the interview:

1. **Demo A** — Full UI: Pictor query, Delphi–Rome anchors, Itiner-e (or terrain) line, route details showing historical vs simulation labels.
2. **Demo B** — Libo query with SEA leg: coastal access + direct-water LineString **or** explicit `failed_gap` with `MARITIME_GEOMETRY_UNAVAILABLE` and non-empty evidence.
3. **Evidence + route details** — Split or single frame showing evidence panel and endpoint/mode summary side by side.

Store captures outside Git or in a private interview folder.

## Owner talking points (short)

- **RAG alone is insufficient** — Retrieval returns passages; admission, actor identity, and movement mode need deterministic rules.
- **Historical authority ≠ GIS** — The pipeline builds a evidence-backed route object first; reconstruction is an optional second pass.
- **Roman roads are priors** — Itiner-e segments are ancient **infrastructure candidates**, not proof a general marched on them.
- **Representative coordinates** — Pleiades anchors are allowed only as **simulation endpoints**, not as silent rewrites of attested place language.
- **UNKNOWN preserved** — When mode is unclear, simulation may explore alternatives without upgrading claims to attested LAND/SEA.
- **Failed gaps beat fabricated lines** — Missing or invalid geometry stays visible rather than inventing embarkation points or ocean snaps.

## Related docs

- [README](../README.md) — architecture, assets, run instructions
- [Data provenance](v1_data_provenance.md) — what is in Git vs local runtime
- [V1 runtime smoke notes](v1_rc1_runtime_smoke.md) — bounded live validation history
