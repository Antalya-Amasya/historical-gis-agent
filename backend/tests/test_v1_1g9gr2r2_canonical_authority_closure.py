"""V1.1G9G-R2R2: canonical authority closure."""
from __future__ import annotations

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    GeographicFeatureKind,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.route_observations import (
    ObservationOrderingAuthority,
    ObservationOrderingRelation,
    RouteObservation,
    RouteObservationKind,
)

BROAD_QUERY = "Show Commander Alpha's historical route."
EXPLICIT_RED_QUERY = "Trace Commander Alpha during Campaign Red."


def _evidence(text: str, *, eid: str = "ev") -> Evidence:
    return Evidence(
        id=eid,
        author="Fixture",
        work="Test",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 100},
    )


def _place(name: str) -> HistoricalPlace:
    return HistoricalPlace(
        id=name.lower().replace(" ", "-"),
        canonical_name=name,
        latitude=40.0,
        longitude=10.0,
        source="fixture",
        confidence=0.8,
        spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
        coordinate_role="exact_site",
    )


def _binding(name: str, role: EventPlaceRole, *, eid: str = "ev1") -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=[eid]),
        place=_place(name),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=[eid],
        resolver_provenance="fixture",
    )


def _event(
    eid: str,
    statement: str,
    *,
    origin: str,
    destination: str,
    evidence_id: str,
) -> HistoricalEvent:
    return HistoricalEvent(
        id=eid,
        name=eid,
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=[evidence_id],
        source_statements=[statement],
        place_bindings=[
            _binding(origin, EventPlaceRole.ORIGIN, eid=evidence_id),
            _binding(destination, EventPlaceRole.DESTINATION, eid=evidence_id),
        ],
    )


def _observation(
    observation_id: str,
    *,
    label: str,
    event_id: str,
    evidence_id: str,
    place_role: EventPlaceRole,
    actor_text: str = "Commander Alpha",
    actor_status: EventActorStatus = EventActorStatus.EXPLICIT,
) -> RouteObservation:
    return RouteObservation(
        observation_id=observation_id,
        kind=RouteObservationKind.PLACE,
        event_id=event_id,
        label=label,
        actor_text=actor_text,
        actor_status=actor_status,
        evidence_refs=(evidence_id,),
        place_role=place_role,
    )


def _relation(
    earlier: str,
    later: str,
    *,
    event_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    authority: ObservationOrderingAuthority = ObservationOrderingAuthority.AFTER_SUBORDINATE,
) -> ObservationOrderingRelation:
    return ObservationOrderingRelation(
        earlier_observation_id=earlier,
        later_observation_id=later,
        ordering_rule=authority,
        event_ids=event_ids,
        evidence_refs=evidence_refs,
        authority=authority.value,
    )


def _inter_event_case(*, text_a: str, text_b: str, origin_a: str, dest_a: str, origin_b: str, dest_b: str):
    ev_a = _evidence(text_a, eid="ev-a")
    ev_b = _evidence(text_b, eid="ev-b")
    event_a = _event("event-a", text_a, origin=origin_a, destination=dest_a, evidence_id="ev-a")
    event_b = _event("event-b", text_b, origin=origin_b, destination=dest_b, evidence_id="ev-b")
    observations = [
        _observation("a-origin", label=origin_a, event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a-dest", label=dest_a, event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
        _observation("b-origin", label=origin_b, event_id="event-b", evidence_id="ev-b", place_role=EventPlaceRole.ORIGIN),
        _observation("b-dest", label=dest_b, event_id="event-b", evidence_id="ev-b", place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation(
        "a-dest",
        "b-origin",
        event_ids=("event-a", "event-b"),
        evidence_refs=("ev-a", "ev-b"),
        authority=ObservationOrderingAuthority.TEMPORAL_ORDER,
    )
    obs_by = {item.observation_id: item for item in observations}
    events_by = {event_a.id: event_a, event_b.id: event_b}
    evidence_by = {ev_a.id: ev_a, ev_b.id: ev_b}
    return observations, relation, [event_a, event_b], [ev_a, ev_b], obs_by, events_by, evidence_by


def _canonical_and_component(query: str, observations, relation, events, evidence):
    contexts = (query,)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {event.id: event for event in events}
    evidence_by = {item.id: item for item in evidence}
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, contexts,
    )
    assembly = assemble_observation_components(
        observations, [relation], events, evidence, query_contexts=contexts,
    )
    edge = (relation.earlier_observation_id, relation.later_observation_id)
    component_admitted = any(
        edge in component.relation_ids for component in assembly.components
    )
    return admission, assembly, component_admitted


def test_broad_query_parses_confirmed_subject():
    scope = parse_query_route_scope((BROAD_QUERY,))
    assert scope.subject == "Commander Alpha"


def test_one_sided_unknown_episode_compatibility():
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_case(
        text_a="During Campaign Red, Commander Alpha marched from Port A to River X.",
        text_b="Commander Alpha marched from River X to City B.",
        origin_a="Port A",
        dest_a="River X",
        origin_b="River X",
        dest_b="City B",
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (BROAD_QUERY,),
    )
    assert admission.subject_match is AuthorityState.MATCH
    assert admission.event_episode_compatibility is AuthorityState.UNKNOWN
    assert not admission.admitted
    assert "INTER_EVENT_EPISODE_UNKNOWN" in admission.reason_codes


def test_one_sided_unknown_component_agrees():
    observations, relation, events, evidence, _, _, _ = _inter_event_case(
        text_a="During Campaign Red, Commander Alpha marched from Port A to River X.",
        text_b="Commander Alpha marched from River X to City B.",
        origin_a="Port A",
        dest_a="River X",
        origin_b="River X",
        dest_b="City B",
    )
    admission, assembly, component_admitted = _canonical_and_component(
        BROAD_QUERY, observations, relation, events, evidence,
    )
    assert admission.admitted == component_admitted
    assert not admission.admitted
    assert not any(
        len(component.observation_ids) > 2
        for component in assembly.components
    )


@pytest.mark.parametrize(
    ("campaign_a", "campaign_b", "expected"),
    [
        ("Campaign Red", "Campaign Red", AuthorityState.MATCH),
        ("Campaign Red", "Campaign Blue", AuthorityState.WRONG),
        ("Campaign Red", "", AuthorityState.UNKNOWN),
    ],
)
def test_inter_event_compatibility_matrix(campaign_a, campaign_b, expected):
    text_a = f"During {campaign_a}, Commander Alpha marched from Port A to Port B." if campaign_a else "Commander Alpha marched from Port A to Port B."
    text_b = f"During {campaign_b}, Commander Alpha marched from Port B to Port C." if campaign_b else "Commander Alpha marched from Port B to Port C."
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_case(
        text_a=text_a,
        text_b=text_b,
        origin_a="Port A",
        dest_a="Port B",
        origin_b="Port B",
        dest_b="Port C",
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (BROAD_QUERY,),
    )
    assert admission.event_episode_compatibility is expected


def test_query_subject_beta_relation_rejects_canonically():
    text = "During Campaign Red, Commander Alpha marched from Port A to Port B."
    ev = _evidence(text)
    event = _event("event-a", text, origin="Port A", destination="Port B", evidence_id="ev-a")
    observations = [
        _observation("a1", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN, actor_text="Commander Beta"),
        _observation("a2", label="Port B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION, actor_text="Commander Beta"),
    ]
    relation = _relation("a1", "a2", event_ids=("event-a",), evidence_refs=("ev-a",))
    admission, assembly, component_admitted = _canonical_and_component(
        BROAD_QUERY, observations, relation, [event], [ev],
    )
    assert admission.subject_match is AuthorityState.WRONG
    assert not admission.admitted
    assert "QUERY_SUBJECT_REJECTED" in admission.reason_codes
    assert admission.admitted == component_admitted
    assert not component_admitted
    assert not any(item["reason"] == "ACTOR_AUTHORITY_REJECTED" for item in assembly.rejected_edges)


def test_conflicting_episode_provenance_rejects():
    text = (
        "During Campaign Red, Commander Alpha advanced toward City B; "
        "during Campaign Blue, he later marched elsewhere."
    )
    ev = _evidence(text)
    event = _event("event-a", text, origin="Port A", destination="City B", evidence_id="ev-a")
    observations = [
        _observation("a1", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a2", label="City B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation("a1", "a2", event_ids=("event-a",), evidence_refs=("ev-a",))
    admission = classify_observation_relation_admission(
        relation,
        {item.observation_id: item for item in observations},
        {event.id: event},
        {ev.id: ev},
        (EXPLICIT_RED_QUERY,),
    )
    assert not admission.admitted
    assert admission.episode_match is AuthorityState.WRONG


def test_mixed_window_does_not_union_campaign_terms_across_events():
    text_a = "During Campaign Red, Commander Alpha marched from Port A to River X."
    text_b = "During Campaign Blue, Commander Alpha marched from River X to City B."
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_case(
        text_a=text_a,
        text_b=text_b,
        origin_a="Port A",
        dest_a="River X",
        origin_b="River X",
        dest_b="City B",
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (BROAD_QUERY,),
    )
    assert admission.event_episode_compatibility is AuthorityState.WRONG


def test_input_order_stability():
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_case(
        text_a="During Campaign Red, Commander Alpha marched from Port A to River X.",
        text_b="Commander Alpha marched from River X to City B.",
        origin_a="Port A",
        dest_a="River X",
        origin_b="River X",
        dest_b="City B",
    )
    first = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (BROAD_QUERY,),
    )
    reversed_events = {
        event_b.id: event_b,
        event_a.id: event_a,
    } if False else events_by
    event_a = events[0]
    event_b = events[1]
    second = classify_observation_relation_admission(
        relation,
        obs_by,
        {event_b.id: event_b, event_a.id: event_a},
        evidence_by,
        (BROAD_QUERY,),
    )
    assert first == second


@pytest.mark.parametrize(
    ("query", "campaign_a", "campaign_b", "actor", "expected_admitted"),
    [
        (BROAD_QUERY, "Campaign Red", "Campaign Red", "Commander Alpha", True),
        (BROAD_QUERY, "Campaign Red", "Campaign Blue", "Commander Alpha", False),
        (BROAD_QUERY, "Campaign Red", "", "Commander Alpha", False),
        (BROAD_QUERY, "Campaign Red", "Campaign Red", "Commander Beta", False),
    ],
)
def test_canonical_component_agreement_matrix(
    query, campaign_a, campaign_b, actor, expected_admitted,
):
    text_a = f"During {campaign_a}, {actor} marched from Port A to Port B." if campaign_a else f"{actor} marched from Port A to Port B."
    text_b = f"During {campaign_b}, {actor} marched from Port B to Port C." if campaign_b else f"{actor} marched from Port B to Port C."
    observations, relation, events, evidence, _, _, _ = _inter_event_case(
        text_a=text_a,
        text_b=text_b,
        origin_a="Port A",
        dest_a="Port B",
        origin_b="Port B",
        dest_b="Port C",
    )
    if actor != "Commander Alpha":
        observations = [
            _observation(
                item.observation_id,
                label=item.label,
                event_id=item.event_id,
                evidence_id=item.evidence_refs[0],
                place_role=item.place_role or EventPlaceRole.PLACE,
                actor_text=actor,
            )
            for item in observations
        ]
    admission, _, component_admitted = _canonical_and_component(
        query, observations, relation, events, evidence,
    )
    assert admission.admitted == expected_admitted
    assert admission.admitted == component_admitted
