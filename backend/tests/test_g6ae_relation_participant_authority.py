"""G6AE: every relation participant must be episode-admissible for the query."""

from __future__ import annotations

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.episode_relevance import EpisodeRelevance, classify_event_anchor_episode
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    classify_relation_relevance,
    event_relevance,
    relation_admission_allowed,
)
from backend.tests.test_g4b_route_components import relation

COORDS = {
    "Roma": (41.9, 12.5),
    "Capua": (41.08, 14.25),
    "Brundisium": (40.6, 17.9),
    "Corcyra": (39.6, 19.9),
    "Alpha": (45.0, 5.0),
    "Beta": (45.5, 5.5),
    "Gamma": (46.0, 6.0),
}

QUERY_200 = "Trace Ariston's route in 200 BCE from Capua to Brundisium."
UNRESTRICTED_QUERY = "Trace Ariston's route."


def place(name: str) -> HistoricalPlace:
    lat, lon = COORDS[name]
    return HistoricalPlace(
        id=name.lower(),
        canonical_name=name,
        latitude=lat,
        longitude=lon,
        source="test",
        confidence=0.8,
    )


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Source",
        work="Work",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 10},
    )


def binding(name: str, role: EventPlaceRole, refs: list[str]) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=refs),
        place=place(name),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=refs,
        resolver_provenance="test",
    )


def movement_event(
    identifier: str,
    statement: str,
    origin: str,
    destination: str,
    *,
    year: str | None,
    refs: list[str],
) -> HistoricalEvent:
    grounding = HistoricalEventTemporalGrounding()
    if year is not None:
        grounding = HistoricalEventTemporalGrounding(
            raw_expression=f"{year} BCE",
            normalized_start=f"-{year}",
            normalized_end=f"-{year}",
            precision=TemporalPrecision.YEAR,
            evidence_refs=refs,
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        )
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        source_statements=[statement],
        place_bindings=[
            binding(origin, EventPlaceRole.ORIGIN, refs),
            binding(destination, EventPlaceRole.DESTINATION, refs),
        ],
        temporal_grounding=grounding,
    )


def classify_movement(
    event: HistoricalEvent,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...],
):
    rel = relation(
        event.place_bindings[0].place.canonical_name,
        event.place_bindings[1].place.canonical_name,
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=tuple(event.evidence_refs),
        event_ids=(event.id,),
    )
    return classify_event_anchor_episode(
        rel,
        events_by_id,
        evidence_by_id,
        contexts,
        subject_relevance=event_relevance(event, evidence_by_id, contexts),
    )


def build_route(
    events: list[HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    *,
    query_contexts: tuple[str, ...],
):
    return EventAnchorRouteBuilder().build_with_diagnostics(
        events,
        list(evidence_by_id.values()),
        event_id="g6ae",
        name="G6AE",
        period="200 BCE",
        query_contexts=query_contexts,
    )


def test_a_exact_wrong_period_bridge_is_rejected():
    e1 = movement_event(
        "e1",
        "In 200 BCE Ariston marched from Rome to Capua.",
        "Roma",
        "Capua",
        year="200",
        refs=["ev1"],
    )
    e2 = movement_event(
        "e2",
        "In 100 BCE Ariston sailed from Brundisium to Corcyra.",
        "Brundisium",
        "Corcyra",
        year="100",
        refs=["ev2"],
    )
    evidence_by_id = {
        "ev1": evidence("ev1", e1.summary),
        "ev2": evidence("ev2", e2.summary),
    }
    events_by_id = {e1.id: e1, e2.id: e2}
    contexts = (QUERY_200,)

    _, e1_detail = classify_movement(e1, events_by_id, evidence_by_id, contexts)
    _, e2_detail = classify_movement(e2, events_by_id, evidence_by_id, contexts)
    assert e1_detail["admitted"] is True
    assert e2_detail["admitted"] is False

    bridge = relation(
        "Capua",
        "Brundisium",
        OrderingRule.TEMPORAL_ORDER,
        refs=("ev1", "ev2"),
        event_ids=("e1", "e2"),
    )
    _, bridge_detail = classify_event_anchor_episode(
        bridge,
        events_by_id,
        evidence_by_id,
        contexts,
        subject_relevance=classify_relation_relevance(
            bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
        ),
    )
    assert bridge_detail["admitted"] is False
    assert not relation_admission_allowed(
        bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
    )


def test_b_both_participants_valid_bridge_may_remain():
    e1 = movement_event(
        "e1",
        "In 200 BCE Ariston marched from Rome to Capua.",
        "Roma",
        "Capua",
        year="200",
        refs=["ev1"],
    )
    e2 = movement_event(
        "e2",
        "In 200 BCE Ariston sailed from Brundisium to Corcyra.",
        "Brundisium",
        "Corcyra",
        year="200",
        refs=["ev2"],
    )
    evidence_by_id = {
        "ev1": evidence("ev1", e1.summary),
        "ev2": evidence("ev2", e2.summary),
    }
    events_by_id = {e1.id: e1, e2.id: e2}
    contexts = (QUERY_200,)

    bridge = relation(
        "Capua",
        "Brundisium",
        OrderingRule.TEMPORAL_ORDER,
        refs=("ev1", "ev2"),
        event_ids=("e1", "e2"),
    )
    _, bridge_detail = classify_event_anchor_episode(
        bridge,
        events_by_id,
        evidence_by_id,
        contexts,
        subject_relevance=classify_relation_relevance(
            bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
        ),
    )
    assert bridge_detail["admitted"] is True
    assert relation_admission_allowed(
        bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
    )


def test_c_structural_bridge_with_invalid_participant_is_rejected():
    e1 = movement_event(
        "e1",
        "In 200 BCE Ariston marched from Alpha to Beta.",
        "Alpha",
        "Beta",
        year="200",
        refs=["ev1"],
    )
    e2 = movement_event(
        "e2",
        "In 100 BCE Ariston marched from Beta to Gamma.",
        "Beta",
        "Gamma",
        year="100",
        refs=["ev2"],
    )
    evidence_by_id = {
        "ev1": evidence("ev1", e1.summary),
        "ev2": evidence("ev2", e2.summary),
    }
    events_by_id = {e1.id: e1, e2.id: e2}
    contexts = ("Trace Ariston's route in 200 BCE.",)
    bridge = relation(
        "Beta",
        "Gamma",
        OrderingRule.SOURCE_STRUCTURAL_ORDER,
        refs=("ev1", "ev2"),
        event_ids=("e1", "e2"),
    )
    _, bridge_detail = classify_event_anchor_episode(
        bridge,
        events_by_id,
        evidence_by_id,
        contexts,
        subject_relevance=classify_relation_relevance(
            bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
        ),
    )
    assert bridge_detail["admitted"] is False


def test_d_wrong_subject_participant_vetoes_bridge():
    e1 = movement_event(
        "e1",
        "Subject Alpha advanced from Rome to Capua.",
        "Roma",
        "Capua",
        year="200",
        refs=["ev1"],
    )
    e2 = movement_event(
        "e2",
        "Commander Z crossed from Brundisium to Corcyra.",
        "Brundisium",
        "Corcyra",
        year="200",
        refs=["ev2"],
    )
    evidence_by_id = {
        "ev1": evidence("ev1", e1.summary),
        "ev2": evidence("ev2", e2.summary),
    }
    events_by_id = {e1.id: e1, e2.id: e2}
    contexts = ("Subject Alpha campaign",)
    bridge = relation(
        "Capua",
        "Brundisium",
        OrderingRule.TEMPORAL_ORDER,
        refs=("ev1", "ev2"),
        event_ids=("e1", "e2"),
    )
    _, e1_detail = classify_movement(e1, events_by_id, evidence_by_id, contexts)
    _, e2_detail = classify_movement(e2, events_by_id, evidence_by_id, contexts)
    assert e1_detail["admitted"] is True
    assert e2_detail["admitted"] is False
    assert e2_detail["episode_classification"] == EpisodeRelevance.OTHER_CAMPAIGN.value
    assert not relation_admission_allowed(
        bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
    )


def test_e_constrained_unknown_participant_vetoes_bridge():
    e1 = movement_event(
        "e1",
        "In 200 BCE Ariston marched from Rome to Capua.",
        "Roma",
        "Capua",
        year="200",
        refs=["ev1"],
    )
    e2 = movement_event(
        "e2",
        "Ariston sailed from Brundisium to Corcyra.",
        "Brundisium",
        "Corcyra",
        year=None,
        refs=["ev2"],
    )
    evidence_by_id = {
        "ev1": evidence("ev1", e1.summary),
        "ev2": evidence("ev2", e2.summary),
    }
    events_by_id = {e1.id: e1, e2.id: e2}
    contexts = ("Trace Ariston's route in 200 BCE.",)
    bridge = relation(
        "Capua",
        "Brundisium",
        OrderingRule.TEMPORAL_ORDER,
        refs=("ev1", "ev2"),
        event_ids=("e1", "e2"),
    )
    _, bridge_detail = classify_event_anchor_episode(
        bridge,
        events_by_id,
        evidence_by_id,
        contexts,
        subject_relevance=classify_relation_relevance(
            bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
        ),
    )
    assert bridge_detail["admitted"] is False


def test_f_unrestricted_subject_multi_event_bridge_not_rejected_for_year_difference():
    e1 = movement_event(
        "e1",
        "Ariston's marched from Rome to Capua in 200 BCE.",
        "Roma",
        "Capua",
        year="200",
        refs=["ev1"],
    )
    e2 = movement_event(
        "e2",
        "Ariston's sailed from Brundisium to Corcyra in 100 BCE.",
        "Brundisium",
        "Corcyra",
        year="100",
        refs=["ev2"],
    )
    evidence_by_id = {
        "ev1": evidence("ev1", e1.summary),
        "ev2": evidence("ev2", e2.summary),
    }
    events_by_id = {e1.id: e1, e2.id: e2}
    contexts = (UNRESTRICTED_QUERY,)

    _, e1_detail = classify_movement(e1, events_by_id, evidence_by_id, contexts)
    _, e2_detail = classify_movement(e2, events_by_id, evidence_by_id, contexts)
    assert e1_detail["admitted"] is True
    assert e2_detail["admitted"] is True

    bridge = relation(
        "Capua",
        "Brundisium",
        OrderingRule.TEMPORAL_ORDER,
        refs=("ev1", "ev2"),
        event_ids=("e1", "e2"),
    )
    assert relation_admission_allowed(
        bridge, events_by_id, evidence_by_id, contexts, rule=bridge.rule,
    )


def test_route_level_unsafe_bridge_removed_but_valid_fragment_preserved():
    e1 = movement_event(
        "e1",
        "In 200 BCE Ariston marched from Rome to Capua.",
        "Roma",
        "Capua",
        year="200",
        refs=["ev1"],
    )
    e2 = movement_event(
        "e2",
        "In 100 BCE Ariston sailed from Brundisium to Corcyra.",
        "Brundisium",
        "Corcyra",
        year="100",
        refs=["ev2"],
    )
    evidence_by_id = {
        "ev1": evidence("ev1", e1.summary),
        "ev2": evidence("ev2", e2.summary),
    }
    outcome = build_route([e1, e2], evidence_by_id, query_contexts=(QUERY_200,))

    bridge_relations = [
        rel for rel in outcome.relations
        if rel.rule is OrderingRule.TEMPORAL_ORDER and rel.earlier == "Capua" and rel.later == "Brundisium"
    ]
    assert bridge_relations == []

    same_movement = [
        rel for rel in outcome.relations
        if rel.rule is OrderingRule.SAME_MOVEMENT_EVENT and rel.event_ids == ("e1",)
    ]
    assert same_movement
    if outcome.route is not None:
        assert outcome.route.ordered_points[0].historical_place.canonical_name == "Roma"
        assert outcome.route.ordered_points[1].historical_place.canonical_name == "Capua"
        assert all(
            point.historical_place.canonical_name != "Brundisium"
            for point in outcome.route.ordered_points
        )
