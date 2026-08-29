# Engineering Execution Policy

## Risk-Proportional Engineering

Apply rigor in proportion to actual risk. For low-risk UI, presentation, CSS,
types, mechanical refactors, approved small implementations, and test-fixture
migrations: implement directly and run focused validation. For medium-risk
backend features, orchestration, interfaces, and ordinary agent behavior:
implement the approved design, run focused tests, then relevant regression.
Investigate further only after concrete failures. For high-risk evidence
provenance, HistoricalEvent authority, place identity, coordinates, Event to
Anchor projection, ordering, movement relations, GIS authority, or destructive
data/schema work: validate conservatively and fail closed when authority cannot
be established.

## Execution and Forensics

When architecture is explicitly approved, implement it; reopen broad design
only for a concrete contradiction, unsafe invariant, or blocker. Start
forensics only for observed failures, invariant violations, unexplained runtime
behavior, or security/trust-boundary risk. Once the cause is established,
implement the approved fix rather than repeating the audit.

Avoid speculative fallback chains, compatibility adapters, duplicate validators,
silent exception swallowing, redundant logging, and parallel logic. Extend the
existing authoritative abstraction when one exists. Keep unexpected failures
observable through explicit state, diagnostics, tests, or appropriate logging.
Stop when requested acceptance criteria pass; report known limitations rather
than adding speculative safeguards or unrelated cleanup.

## Historical GIS Firewall

Narrative flexibility must never propagate into spatial authority. Keep strict:
Evidence identity, HistoricalEvent provenance, place identity, coordinate
provenance, Event-to-Anchor eligibility, anchor ordering, and historical
movement relations. GIS may infer candidate geometry only between sufficiently
established historical anchors. Roads, terrain, passes, shortest paths, modern
roads, and LLM narrative are never historical evidence merely because they are
plausible.

Historical QA prose is not GIS authority. The required chain is:

`Evidence → HistoricalEvent → strict geographic/event projection → HistoricalAnchor → ordering → HistoricalRoute → GIS candidate reconstruction`

Never derive a HistoricalAnchor or historical route fact directly from LLM
prose.

## Token and Iteration Efficiency

Use the shortest validation path capable of detecting realistic regressions.
Do not repeatedly reread unchanged architecture, rerun broad forensics or live
probes, or restate established constraints unless new evidence requires it.
Reuse valid checkpoint and test evidence.
