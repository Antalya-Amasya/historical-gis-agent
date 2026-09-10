"""G6DS: admitted SOURCE_STRUCTURAL_ORDER reaches existing route assembly."""

from __future__ import annotations

from backend.app.models import EventActorStatus
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.tests.test_g6ae_relation_participant_authority import classify_movement, movement_event
from backend.tests.test_g6dr_connector_relation_admission import QUERY, _pair, _structural
from backend.tests.test_g6dq_explicit_connector_source_chronology import (
    _inter_order,
    explicit_actor,
    movement_event as dq_movement_event,
    passage_evidence,
    unknown_actor,
)

POSITIVE_FIRST = "In 200 BCE Ariston marched from Roma to Capua."
POSITIVE_SECOND = "Then in 200 BCE Ariston sailed from Brundisium to Corcyra."


def _build(query_contexts=QUERY):
    e1, e2, evidence_by_id = _pair(POSITIVE_FIRST, POSITIVE_SECOND)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2],
        list(evidence_by_id.values()),
        event_id="g6ds",
        name="Ariston",
        period="200 BCE",
        query_contexts=query_contexts,
    )
    return outcome, e1, e2, evidence_by_id


def test_admitted_structural_order_assembles_capua_before_brundisium():
    outcome, e1, e2, evidence_by_id = _build()
    events_by_id = {e1.id: e1, e2.id: e2}

    _, e1_detail = classify_movement(e1, events_by_id, evidence_by_id, QUERY)
    _, e2_detail = classify_movement(e2, events_by_id, evidence_by_id, QUERY)
    assert e1_detail["admitted"] is True and e2_detail["admitted"] is True
    assert e1.actor.actor_status is EventActorStatus.EXPLICIT
    assert e2.actor.actor_text == "Ariston"

    bridge = _structural(outcome.relations)
    assert len(bridge) == 1
    bridge = bridge[0]
    assert bridge.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER
    assert outcome.route is not None

    waypoint_names = [point.historical_place.canonical_name for point in outcome.route.ordered_points]
    assert waypoint_names == ["Roma", "Capua", "Brundisium", "Corcyra"]
    assert waypoint_names.index("Capua") < waypoint_names.index("Brundisium")

    capua_point = next(point for point in outcome.route.ordered_points if point.historical_place.canonical_name == "Capua")
    brundisium_point = next(point for point in outcome.route.ordered_points if point.historical_place.canonical_name == "Brundisium")
    assert capua_point.coordinate_role == "exact_site"
    assert brundisium_point.coordinate_role == "exact_site"

    bridge_claim = next(
        claim for claim in outcome.route.claims
        if claim.source_place == "Capua" and claim.destination_place == "Brundisium"
    )
    assert bridge_claim.claim_type == "WAYPOINT_ORDERING"
    assert bridge_claim.movement_relation is None
    assert "no direct movement is asserted" in bridge_claim.text

    provenance = next(
        item for item in outcome.diagnostics["ordering_provenance"]
        if item["rule"] == OrderingRule.SOURCE_STRUCTURAL_ORDER.value
    )
    assert provenance["historical_authority"] == "EVIDENCE_GROUNDED_WAYPOINT_ORDERING"
    assert provenance["connection_semantics"] == "ALGORITHMIC_GIS_RECONSTRUCTION_REQUIRED"
    assert outcome.diagnostics["component_count"] == 1
    assert outcome.diagnostics["route_point_count"] == 4
    assert len(outcome.route.geometry.coordinates) == 4
    assert bridge.event_ids == ("e1", "e2")


def test_rejected_episode_structural_order_does_not_leak_into_route():
    first = "In 200 BCE Ariston marched from Roma to Capua."
    second = "Then Ariston sailed from Brundisium to Corcyra."
    e1, e2, evidence_by_id = _pair(first, second)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6ds", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    assert _inter_order(e1, e2, evidence_by_id)[2] is OrderingRule.SOURCE_STRUCTURAL_ORDER
    assert _structural(outcome.relations) == []
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == ["Roma", "Capua"]
    assert "Brundisium" not in {point.historical_place.canonical_name for point in outcome.route.ordered_points}


def test_non_exact_bridge_endpoint_fails_closed_before_route():
    """Strong non-exact guard also covered by test_g6cx_strong_route_point_eligibility_guard."""
    e1, e2, evidence_by_id = _pair(POSITIVE_FIRST, POSITIVE_SECOND)
    origin_binding, destination_binding = e2.place_bindings
    e2 = e2.model_copy(update={
        "place_bindings": [
            origin_binding.model_copy(update={
                "place": origin_binding.place.model_copy(update={"coordinate_role": "feature_centroid"}),
            }),
            destination_binding,
        ],
    })
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6ds", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    assert _structural(outcome.relations) == []
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == ["Roma", "Capua"]
    assert any(code.startswith("NON_EXACT_FEATURE_ANCHOR:") for code in outcome.diagnostics["projection_diagnostics"])


def test_pompey_still_has_no_composed_route():
    cyprus = dq_movement_event(
        "event-4b6ee897cebc", "Pompey sailed toward Cyprus.", "Rhodes", "Cyprus",
        refs=["ev-cyprus"], actor=unknown_actor(),
    )
    pelusium = dq_movement_event(
        "event-bcd445242ec8", "He steered his course that way toward Pelusium.", "Alpha", "Pelusium",
        refs=["ev-pelusium"], actor=unknown_actor(),
    )
    evidence_by_id = {
        "ev-cyprus": passage_evidence("ev-cyprus", cyprus.summary, offset=100, document="pompey-doc"),
        "ev-pelusium": passage_evidence("ev-pelusium", pelusium.summary, offset=200, document="pompey-doc"),
    }
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [cyprus, pelusium], list(evidence_by_id.values()), event_id="g6ds", name="Pompey", period="48 BCE",
    )
    assert not any(rel.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER for rel in outcome.relations)
    assert outcome.route is None or not outcome.route.ordered_points
    if outcome.route is not None:
        names = {point.historical_place.canonical_name for point in outcome.route.ordered_points}
        assert "Cyprus" not in names or "Pelusium" not in names
