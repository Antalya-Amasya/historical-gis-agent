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

## Related docs

- Runtime layout and launcher: repository `README.md`
- Phase retrieval evaluation (separate scope): `docs/phase_2_5_evaluation.md`
