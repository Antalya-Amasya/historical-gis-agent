# Engineering Execution Policy

## 1. Product Stage

This repository is currently in **V1 integration and closure**.

The default goal is to make the approved product behavior work reliably with the **smallest correct change**.

Do not redesign working architecture merely because a more theoretically complete design is possible.

Prefer a working, evidence-safe V1 over speculative completeness or research-grade rigor that is not required by the current task.

---

## 2. Minimal Engineering First

For bugs and integration failures, use this priority order:

1. Preserve information already present in the pipeline.
2. Remove or relax an unnecessary restriction.
3. Fix the producer of incorrect data.
4. Correctly wire an existing component.
5. Make a local implementation fix.
6. Only then consider a new abstraction or architectural mechanism.

Prefer fixing the source of a problem over adding another downstream filter, validator, fallback, or compatibility layer.

Do not introduce architecture for hypothetical future requirements.

Do not generalize beyond the demonstrated problem unless the existing architecture clearly requires it.

A successful investigation may result in **zero production-code changes**.

---

## 3. Complexity Budget

For an ordinary bug fix or integration task, default to:

- production files changed: <= 2
- new production classes: 0
- new validators: 0
- new fallback paths: 0
- new authority types: 0
- new dependencies: 0
- new parallel pipelines: 0

These are defaults, not absolute limits.

If the correct solution materially exceeds them, stop before implementing the larger design and explain why the existing architecture cannot solve the problem locally.

Do not create an abstraction merely to satisfy the complexity budget.

---

## 4. Diagnose the First Real Failure

When behavior crosses multiple pipeline stages, locate the **FIRST_ZERO / first incorrect transformation**.

Example:

`Evidence → HistoricalEvent → PlaceBinding → Geographic Resolution → Anchor → Ordering → Route → GIS`

Do not repair downstream symptoms before identifying the earliest incorrect stage.

Once the root cause is established, fix that cause and stop.

Do not continue broad forensic work after the demonstrated failure has been explained.

Do not modify multiple pipeline layers simultaneously unless the root cause genuinely spans them.

---

## 5. Risk-Proportional Validation

Apply rigor according to actual risk.

### Low risk

Examples:

- UI and CSS
- presentation
- mechanical refactors
- types
- configuration wiring
- approved small fixes

Implement directly and run focused validation.

### Medium risk

Examples:

- backend behavior
- orchestration
- interfaces
- retrieval plumbing
- ordinary agent behavior

Inspect the relevant path, implement the smallest approved fix, run focused tests, then relevant regression tests.

### High risk

Examples:

- historical fact authority
- evidence provenance
- geographic identity
- coordinates
- historical ordering
- movement direction
- destructive schema/data operations

Validate the relevant authority before changing behavior.

High risk does **not** mean automatically adding more gates or validators.

---

## 6. Historical GIS Authority

The primary firewall protects against **invented historical claims**, not against incomplete precision.

### Strictly block

- LLM-only historical facts
- LLM-only historical waypoints
- guessed coordinates
- unsupported historical chronology
- retrieval-rank chronology
- coordinate/distance-derived chronology
- known temporal contradictions
- GIS geometry presented as documentary historical fact

### Allow when evidence-backed

- historically relevant places with incomplete temporal precision
- contextual historical places supported by retrieved Evidence
- UNKNOWN / UNRESOLVED states when they do not create a false assertion
- broad geographic constraints such as regions, rivers, and mountain barriers
- propagation of already established evidence-backed information

**UNKNOWN is not FALSE.**

Incomplete evidence is not automatically hallucinated evidence.

Before rejecting an evidence-backed value, ask:

> Would allowing this value assert a historical fact that the Evidence does not support?

If **yes**, reject it.

If **no**, preserve the information and its uncertainty unless another concrete authority conflict exists.

---

## 7. Historical Facts vs GIS Reconstruction

Keep these layers distinct:

### Historical Fact / Constraint

Must originate from retrieved Evidence and auditable historical/geographic authority.

### GIS Reconstruction

May infer plausible geometry between accepted historical constraints using:

- ancient-road networks
- terrain / DEM
- traversability
- distance
- slope
- geographic barriers

GIS reconstruction is a **candidate reconstruction**, not documentary proof that an actor followed the exact computed geometry.

Ancient roads are infrastructure priors, not proof that a historical actor used a particular road.

The core rule is:

> Evidence constrains historical points and broad order; GIS may reconstruct plausible geometry between them.

Never derive a historical waypoint, coordinate, or historical movement fact directly from unconstrained LLM prose.

---

## 8. Preserve Evidence Before Adding Gates

Evidence-backed information should survive the pipeline unless a concrete rule invalidates it.

Do not discard valid Evidence merely because:

- exact time is unknown;
- a place is contextual rather than a strict movement endpoint;
- documentary evidence does not specify exact point-to-point geometry;
- another event lacks temporal grounding;
- the system cannot establish research-grade certainty.

A filter or gate must protect against a demonstrated false assertion or product invariant.

Do not add a gate solely because additional strictness seems theoretically safer.

When a gate causes valid evidence-backed information to disappear, prefer simplifying or correcting that gate rather than adding another recovery layer downstream.

---

## 9. Movement and Ordering

Historical movement direction requires evidence-backed authority.

Valid ordering may come from existing approved mechanisms such as:

- explicit movement direction;
- comparable evidence-grounded temporal information;
- trustworthy structural ordering already recognized by the system.

Do not manufacture ordering from:

- retrieval order;
- list order;
- coordinates;
- distance;
- GIS paths;
- LLM narrative;
- simple co-occurrence of two places.

A single-endpoint movement is valid information.

For example:

`? → Alesia`

must remain a valid destination constraint when the Evidence supports Alesia but does not state the origin.

Do not invent the missing endpoint merely to form a route.

---

## 10. No Defensive Layer Stacking

Avoid the pattern:

`producer bug → validator → fallback → second validator → compatibility layer`

When possible, fix the producer.

Do not add:

- speculative fallback chains;
- duplicate validators;
- redundant authority checks;
- silent exception swallowing;
- parallel implementations of existing logic;
- compatibility adapters without a demonstrated compatibility requirement.

Extend an existing authoritative abstraction when one already exists.

Unexpected failures should remain observable through existing diagnostics, explicit state, tests, or appropriate logging.

---

## 11. Scope Discipline

Solve the requested problem.

Do not:

- refactor unrelated code;
- clean up unrelated files;
- redesign neighboring systems;
- fix unrelated failing tests;
- add speculative future functionality;
- rewrite working modules for stylistic consistency;
- create planning or architecture documents unless requested or necessary.

Existing unrelated failures may be reported without being fixed.

When architecture has already been approved, implement within it unless there is a concrete contradiction, unsafe invariant, or demonstrated blocker.

---

## 12. Validation

Use the cheapest validation that gives sufficient confidence.

Preferred order:

1. targeted test for changed behavior;
2. directly related regression tests;
3. type/lint/build checks when relevant;
4. minimal runtime smoke test;
5. expensive live/provider acceptance only when necessary.

Do not repeatedly rerun unchanged expensive validation without new evidence.

Reuse valid checkpoints and previous test evidence.

For external LLM/provider acceptance, preflight deterministic infrastructure first.

If validation cannot be run, report why.

---

## 13. Stop Rules

Stop when the requested acceptance criteria pass.

Also stop when investigation proves:

- current production behavior is correct;
- the remaining limitation is caused by missing evidence rather than code;
- the proposed fix would require inventing historical authority;
- the task requires an architectural change outside the approved scope.

Do not modify code merely to demonstrate progress.

Report the limitation and the next real blocker instead.

---

## 14. Repository and Git Safety

Respect the current worktree and branch.

Do not:

- reset;
- restore;
- stash;
- checkout over existing user/agent work;
- rewrite unrelated dirty files;
- amend or squash accepted checkpoints;

unless explicitly instructed.

Preserve existing uncommitted work that belongs to another accepted or pending phase.

Do not commit temporary forensic scripts, runtime outputs, secrets, local environment files, or generated acceptance artifacts unless explicitly requested.

Do not create commits or branches unless the task explicitly requests them.

---

## 15. Secrets and Runtime

Never print, copy, commit, or expose API keys or secrets.

Reuse the approved local runtime, environment, indexes, databases, and persisted data.

Do not rebuild, re-ingest, migrate, reinstall, or replace working infrastructure merely to troubleshoot an application-level problem unless concrete evidence shows the infrastructure itself is defective.

---

## 16. Working Style

Keep plans narrow.

Before implementing, identify:

- the observed failure;
- the likely first incorrect stage;
- the smallest relevant files;
- the focused validation;
- important non-goals.

During implementation:

- reuse existing patterns;
- prefer small patches;
- preserve working behavior;
- avoid speculative improvements.

After implementation, report:

- root cause;
- files changed;
- behavior changed;
- tests run;
- remaining limitation;
- whether any new abstraction, validator, fallback, authority type, or dependency was introduced.

If none were introduced, say so.

The preferred outcome is:

`targeted context → root cause → smallest correct patch → proportional validation → stop`
