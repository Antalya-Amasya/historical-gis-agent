"""V1.1G2B/G3B: behavioral contract for independent waypoint observations."""
from __future__ import annotations

import pytest

from backend.app.geography.feature_semantics import exact_anchor_eligible
from backend.app.models import (
    EventActorStatus, EventGroundingStatus, EventPlaceResolutionStatus, EventPlaceRole,
    Evidence, HistoricalEvent, HistoricalEventActorGrounding, HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention, HistoricalEventTemporalGrounding, HistoricalEventType,
    HistoricalPlace, PlaceSpatialSemantics, TemporalGroundingStatus, TemporalPrecision,
)
from backend.app.routes.event_anchors import project_event_anchors
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor

ALPHA = HistoricalEventActorGrounding(actor_text="Commander Alpha", actor_tokens=["Commander Alpha"], actor_status=EventActorStatus.EXPLICIT)
BETA = HistoricalEventActorGrounding(actor_text="Commander Beta", actor_tokens=["Commander Beta"], actor_status=EventActorStatus.EXPLICIT)
UNKNOWN = HistoricalEventActorGrounding(actor_status=EventActorStatus.UNKNOWN)


def _place(name, *, semantics=PlaceSpatialSemantics.SETTLEMENT, coordinate_role="exact_site"):
    return HistoricalPlace(id=name.lower().replace(" ", "-"), canonical_name=name, latitude=41.0, longitude=12.0, source="fixture", confidence=0.8, spatial_semantics=semantics, coordinate_role=coordinate_role)


def _binding(name, role, refs, *, place=None):
    place = place or _place(name)
    return HistoricalEventPlaceBinding(mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=list(refs)), place=place, role=role, resolution_status=EventPlaceResolutionStatus.RESOLVED, evidence_refs=list(refs), resolver_provenance="fixture")


def _evidence(eid, text, *, document="doc-1", offset=100):
    return Evidence(id=eid, author="Fixture", work="Test", locator="1", excerpt=text[:240], text=text, metadata={"document_id": document, "spine_index": 1, "start_offset": offset})


def _event(eid, statement, bindings, refs, *, actor=ALPHA, years=None):
    grounding = HistoricalEventTemporalGrounding()
    if years is not None:
        grounding = HistoricalEventTemporalGrounding(raw_expression=f"{years} BCE", normalized_start=f"-{years}", normalized_end=f"-{years}", precision=TemporalPrecision.YEAR, evidence_refs=list(refs), status=TemporalGroundingStatus.EVIDENCE_GROUNDED)
    return HistoricalEvent(id=eid, name=eid, summary=statement, event_type=HistoricalEventType.MOVEMENT, grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED, evidence_refs=list(refs), source_statements=[statement], place_bindings=list(bindings), temporal_grounding=grounding, actor=actor)


def _build(events, items, *, query=()):
    return EventAnchorRouteBuilder().build_with_diagnostics(events, items, event_id="g2", name="Alpha", period="401 BCE", query_contexts=query)


def _anchors(events, items):
    return {(a.canonical_name, a.role) for a in project_event_anchors(events, items)[0]}


def _pair(*, connector=None, years_a=None, years_b=None):
    stmt_a, stmt_b = "Commander Alpha departed Harbor One.", "Commander Alpha reached Fort Two."
    if connector:
        stmt_b = f"{connector} {stmt_b}"
    events = [_event("e1", stmt_a, [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"], years=years_a), _event("e2", stmt_b, [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], years=years_b)]
    passage = f"{stmt_a} {stmt_b}"
    return events, [_evidence("ev-a", passage, offset=100), _evidence("ev-b", passage, offset=200)]


def _assert_waypoint_route(outcome, events, *, earlier, later, ev_a, ev_b, rule):
    assert len(events) == 2 and outcome.route is not None and len(outcome.route.route_components) == 1
    path = [point.historical_place.canonical_name for point in outcome.route.route_components[0].ordered_points]
    assert path == [earlier, later]
    relation = next(rel for rel in outcome.relations if rel.earlier == earlier and rel.later == later)
    assert relation.rule is rule and ev_a in relation.evidence_refs and ev_b in relation.evidence_refs
    claim = next(item for item in outcome.route.claims if item.claim_type == "WAYPOINT_ORDERING" and item.source_place == earlier and item.destination_place == later)
    assert claim.movement_relation is None and "no direct movement is asserted" in claim.text


def test_independent_one_ended_observations_form_waypoint_ordering_route():
    events, items = _pair(connector="Then")
    assert _anchors(events, items) == {("Harbor One", EventPlaceRole.ORIGIN), ("Fort Two", EventPlaceRole.DESTINATION)}
    _assert_waypoint_route(_build(events, items), events, earlier="Harbor One", later="Fort Two", ev_a="ev-a", ev_b="ev-b", rule=OrderingRule.SOURCE_STRUCTURAL_ORDER)


def test_origin_only_projects_single_anchor_without_destination():
    event = _event("depart", "Commander Alpha departed Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev"])], ["ev"])
    assert _anchors([event], [_evidence("ev", event.summary)]) == {("Harbor One", EventPlaceRole.ORIGIN)}


def test_destination_only_projects_single_anchor_without_origin():
    event = _event("reach", "Commander Alpha reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev"])], ["ev"])
    assert _anchors([event], [_evidence("ev", event.summary)]) == {("Fort Two", EventPlaceRole.DESTINATION)}


def test_actor_at_place_extractor_produces_presence_without_resolved_route():
    text = "Commander Alpha was at Camp Three."
    items = [_evidence("ev", text)]
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract(items)
    assert len(events) == 1 and events[0].event_type is HistoricalEventType.PRESENCE
    assert events[0].actor.actor_status is EventActorStatus.EXPLICIT
    assert _build(events, items).route is None
    manual = _event("at", text, [_binding("Camp Three", EventPlaceRole.EVENT_SITE, ["ev"])], ["ev"])
    assert _anchors([manual], items) == {("Camp Three", EventPlaceRole.EVENT_SITE)}
    assert _build([manual], items).route is None


def test_attested_direct_edge_preserves_same_movement_ordering():
    stmt = "Commander Alpha marched from Harbor One to Fort Two."
    event = _event("edge", stmt, [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev"]), _binding("Fort Two", EventPlaceRole.DESTINATION, ["ev"])], ["ev"])
    outcome = _build([event], [_evidence("ev", stmt)])
    assert outcome.route is not None and [rel.rule for rel in outcome.relations] == [OrderingRule.SAME_MOVEMENT_EVENT]
    assert outcome.route.claims[0].claim_type == "ORDERING" and outcome.route.claims[0].movement_relation is not None


def test_non_overlapping_dates_form_temporal_waypoint_ordering():
    events, items = _pair(years_a="402", years_b="401")
    _assert_waypoint_route(_build(events, items), events, earlier="Harbor One", later="Fort Two", ev_a="ev-a", ev_b="ev-b", rule=OrderingRule.TEMPORAL_ORDER)


def test_same_actor_temporal_order_without_query_context():
    events, items = _pair(years_a="402", years_b="401")
    _assert_waypoint_route(_build(events, items, query=()), events, earlier="Harbor One", later="Fort Two", ev_a="ev-a", ev_b="ev-b", rule=OrderingRule.TEMPORAL_ORDER)


def test_different_actors_with_dates_fail_closed_without_query():
    events = [_event("e1", "Commander Alpha departed Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"], years="402"), _event("e2", "Commander Beta reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], years="401", actor=BETA)]
    outcome = _build(events, [_evidence("ev-a", events[0].summary), _evidence("ev-b", events[1].summary, offset=200)])
    assert outcome.route is None and outcome.relations == ()


def test_no_chronology_fails_closed_without_route():
    events, items = _pair()
    outcome = _build(events, items)
    assert outcome.route is None and outcome.relations == () and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


@pytest.mark.parametrize("label,events,items", [
    ("wrong_actor", [_event("e1", "Commander Alpha departed Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"], years="402"), _event("e2", "Then Commander Beta reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], years="401", actor=BETA)], [_evidence("ev-a", "Commander Alpha departed Harbor One. Then Commander Beta reached Fort Two.", offset=100), _evidence("ev-b", "Commander Alpha departed Harbor One. Then Commander Beta reached Fort Two.", offset=200)]),
    ("unauthorized_pronoun", [_event("e1", "Commander Alpha departed Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"]), _event("e2", "He later reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], actor=UNKNOWN)], [_evidence("ev-a", "Commander Alpha departed Harbor One.", offset=100), _evidence("ev-b", "He later reached Fort Two.", offset=200)]),
])
def test_unauthorized_actor_does_not_form_shared_route(label, events, items):
    outcome = _build(events, items)
    assert outcome.route is None and not any(rel.earlier == "Harbor One" and rel.later == "Fort Two" for rel in outcome.relations)


def test_wrong_episode_rejects_without_query_context():
    events = [_event("e1", "Commander Alpha departed Harbor One during Campaign Alpha.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"], years="402"), _event("e2", "Then Commander Alpha reached Fort Two during Campaign Beta.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], years="401")]
    passage = f"{events[0].source_statements[0]} {events[1].source_statements[0]}"
    outcome = _build(events, [_evidence("ev-a", passage, offset=100), _evidence("ev-b", passage, offset=200)])
    assert outcome.route is None and outcome.relations == ()


def test_query_only_endpoint_is_not_projected():
    event = _event("at", "Commander Alpha was at Harbor One.", [_binding("Harbor One", EventPlaceRole.EVENT_SITE, ["ev"])], ["ev"])
    items = [_evidence("ev", event.summary)]
    query = ("Trace Commander Alpha from Harbor One to Fort Two.",)
    assert "Fort Two" not in {name for name, _ in _anchors([event], items)}
    assert _build([event], items, query=query).route is None


@pytest.mark.parametrize("label,events,items", [
    ("input_order", *_pair()[:2]),
    ("multiple_destinations", [_event("e1", "Commander Alpha departed Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"]), _binding("Alt Harbor", EventPlaceRole.DESTINATION, ["ev-a"]), _binding("Other Port", EventPlaceRole.DESTINATION, ["ev-a"]), _binding("Camp", EventPlaceRole.EVENT_SITE, ["ev-a"])], ["ev-a"], years="402"), _event("e2", "Commander Alpha reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], years="401")], [_evidence("ev-a", "Commander Alpha departed Harbor One.", offset=100), _evidence("ev-b", "Commander Alpha reached Fort Two.", offset=200)]),
    ("multiple_origins", [_event("e1", "Commander Alpha departed Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"], years="402"), _event("e2", "Commander Alpha reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"]), _binding("Alt Fort", EventPlaceRole.ORIGIN, ["ev-b"]), _binding("Other Origin", EventPlaceRole.ORIGIN, ["ev-b"]), _binding("Camp", EventPlaceRole.EVENT_SITE, ["ev-b"])], ["ev-b"], years="401")], [_evidence("ev-a", "Commander Alpha departed Harbor One.", offset=100), _evidence("ev-b", "Commander Alpha reached Fort Two.", offset=200)]),
    ("cross_document", *_pair()[:2]),
    ("broad_feature", [_event("cross", "Commander Alpha crossed the River Zeta.", [_binding("River Zeta", EventPlaceRole.EVENT_SITE, ["ev"], place=_place("River Zeta", semantics=PlaceSpatialSemantics.RIVER, coordinate_role="feature_centroid"))], ["ev"])], [_evidence("ev", "Commander Alpha crossed the River Zeta.")]),
])
def test_safety_controls_fail_closed(label, events, items):
    if label == "input_order":
        outcome = _build(list(reversed(events)), list(reversed(items)))
        assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]
        return
    if label == "cross_document":
        events, _ = _pair()
        items = [_evidence("ev-a", events[0].summary, document="doc-a"), _evidence("ev-b", events[1].summary, document="doc-b", offset=200)]
        assert _build(events, items).route is None
        return
    if label == "broad_feature":
        anchors, diagnostics = project_event_anchors(events, items)
        assert not anchors and any(code.startswith("NON_EXACT_FEATURE_ANCHOR") for code in diagnostics)
        assert not exact_anchor_eligible(events[0].place_bindings[0].place, strong_role=True)
        return
    assert _build(events, items).route is None


def test_retrospective_reversal_rejects_structural_observation_order():
    stmt_a, stmt_b = "Commander Alpha departed Harbor One.", "Then Commander Alpha reached Fort Two."
    events = [_event("e1", stmt_a, [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"]), _event("e2", f"Earlier, {stmt_b.lower()}", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"])]
    passage = f"{stmt_a} Earlier, commander alpha reached fort two."
    assert _build(events, [_evidence("ev-a", passage, offset=100), _evidence("ev-b", passage, offset=200)]).route is None
