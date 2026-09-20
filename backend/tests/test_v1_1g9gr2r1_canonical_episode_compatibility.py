"""V1.1G9G-R2R1: canonical inter-event episode compatibility."""
from __future__ import annotations

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
) -> RouteObservation:
    return RouteObservation(
        observation_id=observation_id,
        kind=RouteObservationKind.PLACE,
        event_id=event_id,
        label=label,
        actor_text="Commander Alpha",
        actor_status=EventActorStatus.EXPLICIT,
        evidence_refs=(evidence_id,),
        place_role=place_role,
    )


def _relation(
    earlier: str,
    later: str,
    *,
    event_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    authority: ObservationOrderingAuthority = ObservationOrderingAuthority.TEMPORAL_ORDER,
) -> ObservationOrderingRelation:
    return ObservationOrderingRelation(
        earlier_observation_id=earlier,
        later_observation_id=later,
        ordering_rule=authority,
        event_ids=event_ids,
        evidence_refs=evidence_refs,
        authority=authority.value,
    )


def _inter_event_fixture(*, campaign_a: str, campaign_b: str):
    text_a = f"During {campaign_a}, Commander Alpha marched from Port A to Port B."
    text_b = f"During {campaign_b}, Commander Alpha marched from Port B to Port C."
    ev_a = _evidence(text_a, eid="ev-a")
    ev_b = _evidence(text_b, eid="ev-b")
    event_a = _event("event-a", text_a, origin="Port A", destination="Port B", evidence_id="ev-a")
    event_b = _event("event-b", text_b, origin="Port B", destination="Port C", evidence_id="ev-b")
    observations = [
        _observation("a-origin", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a-dest", label="Port B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
        _observation("b-origin", label="Port B", event_id="event-b", evidence_id="ev-b", place_role=EventPlaceRole.ORIGIN),
        _observation("b-dest", label="Port C", event_id="event-b", evidence_id="ev-b", place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation(
        "a-dest",
        "b-origin",
        event_ids=("event-a", "event-b"),
        evidence_refs=("ev-a", "ev-b"),
    )
    events = [event_a, event_b]
    evidence = [ev_a, ev_b]
    obs_by = {item.observation_id: item for item in observations}
    events_by = {event.id: event for event in events}
    evidence_by = {item.id: item for item in evidence}
    return observations, relation, events, evidence, obs_by, events_by, evidence_by


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


def test_broad_query_disjoint_episodes_canonical_rejects():
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_fixture(
        campaign_a="Campaign Red",
        campaign_b="Campaign Blue",
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (BROAD_QUERY,),
    )
    assert admission.subject_match is AuthorityState.MATCH
    assert admission.episode_match is AuthorityState.UNKNOWN
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert admission.movement_assertion is AuthorityState.MATCH
    assert admission.event_episode_compatibility is AuthorityState.WRONG
    assert not admission.admitted
    assert "INTER_EVENT_EPISODE_INCOMPATIBLE" in admission.reason_codes


def test_broad_query_disjoint_episodes_component_agrees_with_canonical():
    observations, relation, events, evidence, _, _, _ = _inter_event_fixture(
        campaign_a="Campaign Red",
        campaign_b="Campaign Blue",
    )
    admission, assembly, component_admitted = _canonical_and_component(
        BROAD_QUERY, observations, relation, events, evidence,
    )
    assert admission.admitted == component_admitted
    assert not admission.admitted
    assert not component_admitted
    assert any(
        item["reason"] == "INTER_EVENT_EPISODE_INCOMPATIBLE"
        for item in assembly.rejected_edges
    )


def test_broad_query_same_episode_may_admit_inter_event_relation():
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_fixture(
        campaign_a="Campaign Red",
        campaign_b="Campaign Red",
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (BROAD_QUERY,),
    )
    assert admission.event_episode_compatibility is AuthorityState.MATCH
    assert admission.admitted


def test_broad_query_same_episode_component_agrees_with_canonical():
    observations, relation, events, evidence, _, _, _ = _inter_event_fixture(
        campaign_a="Campaign Red",
        campaign_b="Campaign Red",
    )
    admission, assembly, component_admitted = _canonical_and_component(
        BROAD_QUERY, observations, relation, events, evidence,
    )
    assert admission.admitted == component_admitted
    assert admission.admitted
    assert component_admitted
    assert assembly.components


def test_broad_query_disjoint_episodes_preserve_separate_components():
    text_a = "During Campaign Red, Commander Alpha marched from Port A to Port B."
    text_b = "During Campaign Blue, Commander Alpha marched from Port X to Port Y."
    ev_a = _evidence(text_a, eid="ev-a")
    ev_b = _evidence(text_b, eid="ev-b")
    event_a = _event("event-a", text_a, origin="Port A", destination="Port B", evidence_id="ev-a")
    event_b = _event("event-b", text_b, origin="Port X", destination="Port Y", evidence_id="ev-b")
    observations = [
        _observation("a1", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a2", label="Port B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
        _observation("b1", label="Port X", event_id="event-b", evidence_id="ev-b", place_role=EventPlaceRole.ORIGIN),
        _observation("b2", label="Port Y", event_id="event-b", evidence_id="ev-b", place_role=EventPlaceRole.DESTINATION),
    ]
    rel_a = _relation(
        "a1",
        "a2",
        event_ids=("event-a",),
        evidence_refs=("ev-a",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    rel_b = _relation(
        "b1",
        "b2",
        event_ids=("event-b",),
        evidence_refs=("ev-b",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    events = [event_a, event_b]
    evidence = [ev_a, ev_b]
    assembly = assemble_observation_components(
        observations,
        [rel_a, rel_b],
        events,
        evidence,
        query_contexts=(BROAD_QUERY,),
    )
    assert len(assembly.components) == 2
    component_sizes = sorted(len(component.observation_ids) for component in assembly.components)
    assert component_sizes == [2, 2]


def test_explicit_query_same_episode_inter_event_may_admit():
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_fixture(
        campaign_a="Campaign Red",
        campaign_b="Campaign Red",
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (EXPLICIT_RED_QUERY,),
    )
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.event_episode_compatibility is AuthorityState.MATCH
    assert admission.admitted


def test_explicit_query_disjoint_episodes_rejects():
    observations, relation, events, evidence, obs_by, events_by, evidence_by = _inter_event_fixture(
        campaign_a="Campaign Red",
        campaign_b="Campaign Blue",
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, events_by, evidence_by, (EXPLICIT_RED_QUERY,),
    )
    assert not admission.admitted
    assert admission.event_episode_compatibility is AuthorityState.WRONG
    assert "INTER_EVENT_EPISODE_INCOMPATIBLE" in admission.reason_codes


def test_same_event_relation_not_rejected_for_inter_event_compatibility():
    text = "During Campaign Red, Commander Alpha marched from Port A to Port B."
    ev = _evidence(text)
    event = _event("event-a", text, origin="Port A", destination="Port B", evidence_id="ev-a")
    observations = [
        _observation("a1", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a2", label="Port B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation(
        "a1",
        "a2",
        event_ids=("event-a",),
        evidence_refs=("ev-a",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    admission = classify_observation_relation_admission(
        relation,
        {item.observation_id: item for item in observations},
        {event.id: event},
        {ev.id: ev},
        (EXPLICIT_RED_QUERY,),
    )
    assert admission.event_episode_compatibility is AuthorityState.MATCH
    assert admission.admitted


def test_same_event_wrong_query_episode_still_rejects():
    text = "During Campaign Blue, Commander Alpha marched from Port A to Port B."
    ev = _evidence(text)
    event = _event("event-a", text, origin="Port A", destination="Port B", evidence_id="ev-a")
    observations = [
        _observation("a1", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a2", label="Port B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation(
        "a1",
        "a2",
        event_ids=("event-a",),
        evidence_refs=("ev-a",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    admission = classify_observation_relation_admission(
        relation,
        {item.observation_id: item for item in observations},
        {event.id: event},
        {ev.id: ev},
        (EXPLICIT_RED_QUERY,),
    )
    assert admission.event_episode_compatibility is AuthorityState.MATCH
    assert admission.episode_match is AuthorityState.WRONG
    assert not admission.admitted
    assert "QUERY_EPISODE_REJECTED" in admission.reason_codes


def test_terra_failure_reproduction_agreement():
    observations, relation, events, evidence, _, _, _ = _inter_event_fixture(
        campaign_a="Campaign Red",
        campaign_b="Campaign Blue",
    )
    admission, assembly, component_admitted = _canonical_and_component(
        BROAD_QUERY, observations, relation, events, evidence,
    )
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted
