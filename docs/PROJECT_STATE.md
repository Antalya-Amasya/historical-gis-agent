# Project State

This is a context-recovery reference for development agents, not authority over
the implementation. **CURRENT WORKTREE + CURRENT TEST RESULTS OVERRIDE THIS
DOCUMENT IF THEY CONFLICT.** Verify actual Git and runtime state before acting.

## Product goal

Historical evidence determines **what movement happened**. GIS determines one
plausible simulated **how** for visualization. The project does not claim to
reconstruct the literal historical itinerary. Historical facts remain
evidence-backed; generated geometry is simulation.

## Historical authority rules

- Actor identity must be grounded in bounded evidence.
- A movement must be historically supported before GIS can simulate it.
- Origin and destination must be evidence-backed; GIS cannot invent historical
  waypoints or endpoints.
- Explicit episode/time constraints fail closed when the required authority is
  absent, conflicting, or unknown.
- `UNKNOWN` is not `FALSE` and is not positive support.
- GIS cannot create authority for historical actor, movement, episode,
  chronology, or waypoint claims.

## Simulation rules

- `representative_point != exact_site`: an eligible resolved settlement or port
  may anchor simulation without claiming an exact historical site.
- Roman roads are routing priors, not proof an actor used a particular road.
- SRTM terrain is a plausibility/cost input, not historical evidence.
- Simulated coastal access is not a historical embarkation point.
- Simulation-selected LAND, SEA, or mixed mode must not be promoted into the
  historical travel mode.
- Algorithm-generated road nodes, terrain vertices, and coastal access points
  are not historical waypoints.
- Simulation geometry is not evidence; unresolved geometry should remain an
  explicit gap rather than a fabricated line.

## Canonical GIS asset paths

| Asset | Canonical location | Role / status |
| --- | --- | --- |
| Pleiades SQLite | Repository `data/pleiades_v4_1/` (ignored runtime data) | Place identity and representative coordinates |
| Itiner-e | Repository `data/raw/itiner_e/` (ignored runtime data) | Ancient-road routing prior |
| Natural Earth | Repository `data/gis/natural_earth_10m/` (ignored runtime data) | Land/ocean/lake topology and maritime validation |
| SRTM HGT | `C:\data\srtm-hgt` | External terrain/elevation input; do not add to Git |
| Runtime / Chroma | Preferred root `C:\data\historical-gis-runtime` | External venv, private `.env`, and Chroma persistence; currently a Windows junction to existing runtime data |

The runtime target behind the junction may vary by machine. The launcher accepts
`HISTORICAL_GIS_RUNTIME_ROOT` and per-resource overrides. Its legacy fallback is
`C:\D\python\202608231533`; this is not the preferred root. A fresh clone needs
separately provisioned runtime/corpus, embedding cache, and ignored GIS datasets.
No multi-gigabyte runtime data belongs in Git.

## Active GIS capabilities

Accepted V1 status: Pleiades, SRTM, Itiner-e, and Natural Earth are active on
the prepared runtime. This is not a claim of universal route completion.

- **LAND:** prefer Itiner-e; use SRTM fallback when road routing cannot provide
  a connected path.
- **SEA:** Natural Earth validates direct-water simulation; simulated coastal
  access may be used where appropriate.
- **UNKNOWN historical mode:** GIS may evaluate plausible simulation
  alternatives, but must not rewrite historical mode.

## Official interview demos

See [`v1_interview_demo.md`](v1_interview_demo.md) for the full walkthrough.

- **LAND:** “Reconstruct Quintus Fabius Pictor's return journey after
  consulting the oracle.” Historical endpoints Delphi → Rome; simulate with
  Itiner-e (SRTM fallback).
- **SEA:** “Trace Libo's route from Oricum to Brundisium.” Historical movement
  is Libo, Oricum/Orikon → Brundisium, SEA when supported; simulate with coastal
  access and Natural Earth-validated water geometry where valid.

## V1 readiness and non-blocking debt

Last audited milestone status: **V1 GIS READY WITH DOCUMENTED LIMITATIONS**;
**interview package READY**; **publication readiness READY WITH NON-BLOCKING
TECH DEBT**. Readiness does not mean a push or publication occurred.

Known debt (non-blocking unless current behavior changes):

- Legacy route-provenance tests contain stale expectations.
- Old synthetic no-route submission tests expect prose to survive without a
  canonical route.
- `FULL_ROUTE` versus internal `PARTIAL_ROUTE` taxonomy remains inconsistent
  in some tests/docs.
- A fresh clone requires external runtime assets and ignored GIS datasets.
- Some V1 routing and maritime thresholds are heuristics, not settled
  historiography.

## Closed design questions

**Do not reopen without a new reproducible defect.**

- Exact historical itinerary reconstruction is not a V1 requirement.
- Representative settlement coordinates are permitted as simulation anchors.
- Roman roads do not need historical proof segment by segment.
- Vegetation, water-source, season, and logistics modeling are out of V1 scope.
- Local visual map tiles are not required.
- Natural-language answers do not require a full deterministic
  proposition-proof engine.
- GIS asset integration is complete; do not restart that work without a defect.

## Deferred from V1

Vegetation; water-source reconstruction; season/weather; logistics/supply;
wind/current; exact archaeological itinerary reconstruction; advanced
multimodal optimization; full proposition theorem/proof engine. These are
deferred, not current implementation recommendations.

## Agent startup checklist

Before every substantial task:

1. Run `git rev-parse HEAD`.
2. Run `git status --short` and inspect staged changes with `git diff --cached`.
3. Inspect the actual worktree; it overrides this document.
4. Verify runtime/service state when relevant.
5. Verify tests instead of trusting old reports.
6. Preserve unrelated untracked debug/eval files.
7. Do not push unless explicitly authorized.

Do not assume a milestone commit named in this file is still HEAD.

## Model-agnostic task discipline

- Do a read-only audit before broad fixes when the cause is unclear.
- Prefer bounded, generic mechanisms; do not hardcode benchmark names/IDs or
  tune exact passage IDs merely to pass a fixture.
- Keep historical authority separate from GIS simulation.
- Stop rather than widening scope when the architecture is insufficient; report
  the concrete blocker.
