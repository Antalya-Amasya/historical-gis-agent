# V1-RC1 runtime smoke (accepted)

Recorded: 2026-09-28 16:20 (北京时间, UTC+8)  
Branch: `agent/cursor`  
Baseline commit: `a196bb7dccd53d5439ee86e53388e16bf88dfd36`  
Launcher: `scripts/start_historical_gis.ps1` (see README quick start)

This note captures the **accepted V1-RC1** runtime smoke outcome. It is observational documentation only; it does not change product behavior or release requirements.

## Measured results

| Check | Outcome |
|---|---|
| Launcher | Started the documented runtime (Chroma, backend, frontend) |
| Readiness | Chroma, backend, and frontend readiness checks passed |
| Chroma collection | `roman_republic_primary_sources_v2` contained **7,230** records |
| Live query | Libo route query (`Trace Libo's route from Oricum to Brundisium.`) returned **`PARTIAL`** |
| Response payload | Two historical waypoints and **one SEA** movement claim |
| Historical route presentation | **Empty** (no presentation geometry attached for this response) |
| Warning | `route_terminal_submission_missing` |
| Focused integration tests | **35** tests passed (same RC1 acceptance run) |

## Verified vs contract-tested vs not live-verified

### Verified (runtime / API)

- One-command launcher brings up Chroma (`127.0.0.1:8002`), backend (`127.0.0.1:8000`), and frontend (`127.0.0.1:5173`) per README.
- Health/readiness gates succeeded for those services.
- Live agent API call for the Libo query returned a structured response with `route_result_status` = `PARTIAL`, waypoint and claim counts as observed above, and the documented warning field.

### Contract-tested (automated; not re-proven on every manual smoke)

- LAND, SEA, and mixed-route assembly behavior.
- Failed-gap / partial-route handling and related route-result contracts.
- The **35** focused integration tests executed in the RC1 acceptance pass (pytest integration scope for V1 route / observation contracts).

### Not live-verified in this smoke

- End-to-end chat-to-GIS presentation for every claim type.
- Actual browser map rendering of maritime or land geometry for the Libo live request (presentation was empty despite SEA claim in payload).

## Recorded limitations (unchanged)

- **B04**: final-evidence miss remains a known retrieval/evaluation gap; this smoke does not claim B04 recovery.
- **Terminal submission**: the observed Libo request surfaced `route_terminal_submission_missing`; this document records the warning only and does **not** assert a root cause.
- **Maritime map**: no demonstrated live maritime map presentation in this smoke (empty historical route presentation for the live Libo response).

## Follow-up runtime verification (2026-09-28, RC3–RC6)

Recorded: 2026-09-28 17:26 (北京时间, UTC+8)  
Documentation commit baseline: `157dc3bba11dad19428a426d187b77e66e5eaa31`  
These slices extend RC1 observational notes only; they do not change release requirements.

### RC3 — terminal closure vs GIS presentation

- **`build_historical_route` succeeded** on the live Libo control path under review.
- **`submit_grounded_answer` was not called** (`route_terminal_submission_missing` surfaced).
- The **existing terminal-closure guardrail** retained route state and did not treat missing submission as a transport failure.
- **Missing terminal submission did not cause the missing GIS presentation** in the RC1/RC3 observation window (presentation absence and terminal closure are separate concerns).

### RC4 — optional GIS assets and launcher defaults

- **Roman-road GeoJSON** and **Natural Earth 1:10m** datasets are present in the worktree layout audited for RC4.
- Natural Earth **manifest audit returned `AVAILABLE`**.
- The **normal launcher path** (`scripts/start_historical_gis.ps1`) leaves **optional GIS configuration disabled** unless operators set `ROMAN_ROAD_ENABLED` / `MARITIME_SURFACE_DATA_ROOT` (see README).

### RC5 — configured maritime runtime smoke

- An **isolated backend** (separate port, process env only; shared `.env` not edited) **successfully loaded `RomanRoadRouteOrchestrator`** when GIS variables were enabled.
- The **live Libo query** reached the **maritime planner** (failure moved from `MARITIME_PLANNER_UNAVAILABLE` to geometry validation).
- GIS result: **`MARITIME_GEOMETRY_UNAVAILABLE`** with **`road_network` status `UNAVAILABLE`**.
- **`historical_route_presentation` was present** but contained an **undrawn `failed_gap`** (`geometry: null`, `failure_status: MARITIME_GEOMETRY_UNAVAILABLE`).
- **Normal runtime services** on the default ports were **left unchanged** after the smoke (isolated instance stopped).

### RC6 — coordinate classification audit

- The geography registry’s **representative coordinates** for **Orikon** and **Brundisium** both classify as **land interior** under the configured Natural Earth surface dataset.
- A **direct maritime planner probe** for that pair returned **`endpoint_not_ocean_interior`**.
- **No maritime candidate LineString** was constructed (consistent with RC5 `failed_gap` behavior).
- These coordinates were **registry-derived** for the audit; RC6 **did not independently re-extract** anchor coordinates from the live agent process.

## Verified runtime capabilities (RC1 + RC3–RC6)

- Launcher and default-stack readiness (RC1).
- Live Libo API can yield **`PARTIAL`** route state with historical waypoints and SEA-oriented GIS leg metadata when optional GIS is enabled (RC5).
- Optional GIS can be enabled without editing the shared `.env` (isolated process env; RC5).
- **`RomanRoadRouteOrchestrator`** loads when `ROMAN_ROAD_ENABLED` and dataset paths are valid (RC4–RC5).
- Maritime surface manifest can reach **`AVAILABLE`** and the planner can run to an explicit **`MARITIME_GEOMETRY_UNAVAILABLE` / `failed_gap`** outcome rather than failing closed as planner-unconfigured (RC5–RC6).
- Terminal-closure guardrails can preserve route output when **`submit_grounded_answer`** is omitted (RC3).

## Known maritime geometry limitation

**Architectural constraint (accepted):**

- **Settlement representative coordinates must not be silently moved to ocean coordinates** to force a drawable sea leg.
- The system must **not invent embarkation locations** or draw an **unvalidated maritime LineString** when endpoint surface validation fails.

**Observed mechanism (RC5–RC6):** Libo’s Orikon→Brundisium leg uses registry representative anchors that classify as **land interior**; direct-water planning rejects the pair (`endpoint_not_ocean_interior`), yielding **`MARITIME_GEOMETRY_UNAVAILABLE`** and an explicit **undrawn gap**. Historical evidence remains the authority for movement and waypoint identity; GIS reconstruction stays algorithmic and fail-closed on invalid water endpoints.

RC1 recorded limitations (**B04**, terminal warning without root-cause speculation) remain in force.

## Unverified capabilities

- **Browser rendering of a successful maritime route** has **not** been demonstrated in runtime smoke (RC1 empty presentation; RC5 only **`failed_gap`** with null geometry).
- End-to-end chat-to-GIS presentation for every claim type under default launcher settings (optional GIS off unless configured).

## Related docs

- Runtime layout and launcher: repository `README.md`
- Phase retrieval evaluation (separate scope): `docs/phase_2_5_evaluation.md`
