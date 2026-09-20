"""R2-FINAL-R1.2: soft-membership temporal eligibility and equal-date structural order."""
from __future__ import annotations

from backend.app.models import (
    EventPlaceRole,
    HistoricalEventTemporalGrounding,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
    relation_non_phase_eligible,
)
from backend.app.routes.route_observations import ObservationOrderingAuthority, project_observation_ordering
from backend.app.routes.soft_phase_membership import build_soft_phase_membership_index
from backend.tests.test_g6ds_route_assembly_acceptance import POSITIVE_FIRST, POSITIVE_SECOND, QUERY, _pair
from backend.tests.test_v1_1g9q1r2sd2_soft_phase_membership import _edge
from backend.tests.test_v1_1g9r2finalr11_query_time_authority import (
    _classify,
    _event,
    _obs,
    _rel,
    _evidence,
)


def _dated(year: int) -> HistoricalEventTemporalGrounding:
    return HistoricalEventTemporalGrounding(
        raw_expression=f"{abs(year)} BCE",
        normalized_start=str(year),
        normalized_end=str(year),
        precision=TemporalPrecision.YEAR,
        evidence_refs=["ev1"],
        status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
    )


def _range(start: int, end: int) -> HistoricalEventTemporalGrounding:
    return HistoricalEventTemporalGrounding(
        raw_expression=f"{abs(start)}-{abs(end)} BCE",
        normalized_start=str(start),
        normalized_end=str(end),
        precision=TemporalPrecision.YEAR_RANGE,
        evidence_refs=["ev1"],
        status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
    )


def _index(query: str, edges: list[tuple[str, str, int | None]]):
    observations, events, evidence, relations = [], [], [], []
    for left, right, year in edges:
        obs, evs, evi, rel = _edge(left, right)
        if year is not None:
            evs = [evs[0].model_copy(update={"temporal_grounding": _dated(year)})]
        observations.extend(obs)
        events.extend(evs)
        evidence.extend(evi)
        relations.append(rel)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    index = build_soft_phase_membership_index(relations, obs_by, events_by, evidence_by, (query,))
    return index, relations, obs_by, events_by, evidence_by


def test_wrong_time_witness_excluded_from_soft_membership():
    query = "Trace Commander Delta from Port A toward City D in 200 BCE."
    index, relations, obs_by, events_by, evidence_by = _index(query, [
        ("Port A", "Station B", -150),
        ("Station B", "Gate C", -200),
        ("Gate C", "City D", -150),
    ])
    scope = parse_query_route_scope((query,))
    assert not relation_non_phase_eligible(relations[0], obs_by, events_by, evidence_by, (query,), scope)
    assert index.anchor_for_edge((relations[0].earlier_observation_id, relations[0].later_observation_id)) is None
    assert index.anchor_for_edge((relations[2].earlier_observation_id, relations[2].later_observation_id)) is None


def test_unknown_time_witness_excluded_from_soft_membership():
    query = "Trace Commander Delta from Port A toward City D in 200 BCE."
    index, relations, *_ = _index(query, [("Port A", "Station B", None), ("Gate C", "City D", -200)])
    assert index.anchor_for_edge((relations[0].earlier_observation_id, relations[0].later_observation_id)) is None


def test_matching_time_witness_still_works():
    query = "Trace Commander Delta from Port A toward City D in 200 BCE."
    index, relations, *_ = _index(query, [
        ("Port A", "Station B", -200),
        ("Station B", "Gate C", -200),
        ("Gate C", "City D", -200),
    ])
    assert index.anchor_for_edge((relations[1].earlier_observation_id, relations[1].later_observation_id)) is not None


def test_no_time_soft_membership_regression():
    query = "Trace Commander Delta from Port A toward City D."
    index, relations, *_ = _index(query, [
        ("Port A", "Station B", None),
        ("Station B", "Gate C", None),
        ("Gate C", "City D", None),
    ])
    assert index.anchor_for_edge((relations[1].earlier_observation_id, relations[1].later_observation_id)) is not None


def test_equal_date_then_is_source_structural_order():
    e1, e2, evidence_by_id = _pair(
        "In 200 BCE Ariston marched from Roma to Capua.",
        "Then in 200 BCE Ariston sailed from Brundisium to Corcyra.",
    )
    grounded = _dated(-200)
    e1 = e1.model_copy(update={"temporal_grounding": grounded})
    e2 = e2.model_copy(update={"temporal_grounding": grounded})
    _, relations, _ = project_observation_ordering([e1, e2], [], [], list(evidence_by_id.values()))
    structural = [rel for rel in relations if rel.ordering_rule is ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER]
    temporal = [rel for rel in relations if rel.ordering_rule is ObservationOrderingAuthority.TEMPORAL_ORDER]
    assert structural and not temporal


def test_overlapping_date_then_is_source_structural_order():
    e1, e2, evidence_by_id = _pair(
        "In 205-195 BCE Ariston marched from Roma to Capua.",
        "Then in 202-198 BCE Ariston sailed from Brundisium to Corcyra.",
    )
    e1 = e1.model_copy(update={"temporal_grounding": _range(-205, -195)})
    e2 = e2.model_copy(update={"temporal_grounding": _range(-202, -198)})
    _, relations, _ = project_observation_ordering([e1, e2], [], [], list(evidence_by_id.values()))
    assert any(rel.ordering_rule is ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER for rel in relations)
    assert not any(rel.ordering_rule is ObservationOrderingAuthority.TEMPORAL_ORDER for rel in relations)


def test_equal_date_without_cue_has_no_inter_event_order():
    e1, e2, evidence_by_id = _pair(
        "In 200 BCE Ariston marched from Roma to Capua.",
        "Ariston sailed from Brundisium to Corcyra.",
    )
    grounded = _dated(-200)
    e1 = e1.model_copy(update={"temporal_grounding": grounded})
    e2 = e2.model_copy(update={"temporal_grounding": grounded})
    _, relations, _ = project_observation_ordering([e1, e2], [], [], list(evidence_by_id.values()))
    assert not any(
        rel.ordering_rule in {
            ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER,
            ObservationOrderingAuthority.TEMPORAL_ORDER,
        }
        for rel in relations
    )


def test_strict_nonoverlap_is_temporal_order():
    e1, e2, evidence_by_id = _pair(
        "In 200 BCE Ariston marched from Roma to Capua.",
        "In 199 BCE Ariston sailed from Brundisium to Corcyra.",
    )
    e1 = e1.model_copy(update={"temporal_grounding": _dated(-200)})
    e2 = e2.model_copy(update={"temporal_grounding": _dated(-199)})
    _, relations, _ = project_observation_ordering([e1, e2], [], [], list(evidence_by_id.values()))
    assert any(rel.ordering_rule is ObservationOrderingAuthority.TEMPORAL_ORDER for rel in relations)


def test_undated_structural_dated_query_unknown():
    statement_a = "Commander Alpha marched from Port A to City B."
    statement_b = "Then Commander Alpha marched from City B to City D."
    a = _event(statement_a, origin="Port A", dest="City B", event_id="e1")
    b = _event(statement_b, origin="City B", dest="City D", event_id="e2")
    obs = (
        _obs("o1", "Port A", EventPlaceRole.ORIGIN, event_id="e1"),
        _obs("o2", "City B", EventPlaceRole.DESTINATION, event_id="e2"),
    )
    rel = _rel("e1", "e2", authority=ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER)
    admission = _classify(a, obs, rel, "Trace Commander Alpha from Port A to City D in 200 BCE.", extra_events=(b,))
    assert admission.temporal_match is AuthorityState.UNKNOWN
    assert admission.admitted is False


def test_g6ds_production_grounded_dated_positive():
    e1, e2, evidence_by_id = _pair(POSITIVE_FIRST, POSITIVE_SECOND)
    grounded = HistoricalEventTemporalGrounding(
        raw_expression="200 BCE",
        normalized_start="-200",
        normalized_end="-200",
        precision=TemporalPrecision.YEAR,
        evidence_refs=["ev1"],
        status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
    )
    e1 = e1.model_copy(update={"temporal_grounding": grounded.model_copy(update={"evidence_refs": ["ev1"]})})
    e2 = e2.model_copy(update={"temporal_grounding": grounded.model_copy(update={"evidence_refs": ["ev2"]})})
    assert e1.temporal_grounding.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    assert e2.temporal_grounding.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6ds", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    assert any(rel.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER for rel in outcome.relations)
    assert outcome.route is not None
    names = [point.historical_place.canonical_name for point in outcome.route.ordered_points]
    assert names == ["Roma", "Capua", "Brundisium", "Corcyra"]


def test_g6ds_undated_negative_remains():
    e1, e2, evidence_by_id = _pair(
        "In 200 BCE Ariston marched from Roma to Capua.",
        "Then Ariston sailed from Brundisium to Corcyra.",
    )
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6ds", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    names = [] if outcome.route is None else [p.historical_place.canonical_name for p in outcome.route.ordered_points]
    assert names != ["Roma", "Capua", "Brundisium", "Corcyra"]


def test_canonical_false_not_restored_by_component():
    query = "Trace Commander Alpha from Port A to City B in 200 BCE."
    event = _event("Commander Alpha marched from Port A to City B.")
    obs = [_obs("o1", "Port A", EventPlaceRole.ORIGIN), _obs("o2", "City B", EventPlaceRole.DESTINATION)]
    rel = _rel()
    admission = _classify(event, obs, rel, query)
    assembly = assemble_observation_components(obs, [rel], [event], [_evidence(event.summary)], query_contexts=(query,))
    assert admission.admitted is False
    assert not any((rel.earlier_observation_id, rel.later_observation_id) in c.relation_ids for c in assembly.components)
