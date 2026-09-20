"""R2-FINAL-R1: canonical same-event admission is sole historical authority."""
from __future__ import annotations

from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEvent,
    HistoricalEventActorGrounding,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
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


def _place(canonical: str) -> HistoricalPlace:
    return HistoricalPlace(
        id=canonical.casefold().replace(" ", "-"),
        canonical_name=canonical,
        latitude=40.0,
        longitude=19.0,
        source="test",
        confidence=0.8,
    )


def _binding(role: EventPlaceRole, mention: str, canonical: str) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=mention, role=role, evidence_refs=["ev1"]),
        place=_place(canonical),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=["ev1"],
        resolver_provenance="test",
    )


def _event(
    statement: str,
    *,
    actor: str = "Commander Alpha",
    origin_mention: str = "Port A",
    origin_canonical: str = "Port A",
    dest_mention: str = "City B",
    dest_canonical: str = "City B",
    event_id: str = "move",
) -> HistoricalEvent:
    return HistoricalEvent(
        id=event_id,
        name=event_id,
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        source_statements=[statement],
        actor=HistoricalEventActorGrounding(
            actor_text=actor,
            actor_status=EventActorStatus.EXPLICIT,
        ),
        place_bindings=[
            _binding(EventPlaceRole.ORIGIN, origin_mention, origin_canonical),
            _binding(EventPlaceRole.DESTINATION, dest_mention, dest_canonical),
        ],
    )


def _obs(oid: str, label: str, role: EventPlaceRole, actor: str = "Commander Alpha") -> RouteObservation:
    return RouteObservation(
        observation_id=oid,
        kind=RouteObservationKind.PLACE,
        event_id="move",
        label=label,
        actor_text=actor,
        actor_status=EventActorStatus.EXPLICIT,
        evidence_refs=("ev1",),
        place_role=role,
    )


def _rel(authority: ObservationOrderingAuthority = ObservationOrderingAuthority.AFTER_SUBORDINATE) -> ObservationOrderingRelation:
    return ObservationOrderingRelation(
        earlier_observation_id="o1",
        later_observation_id="o2",
        ordering_rule=authority,
        event_ids=("move",),
        evidence_refs=("ev1",),
        authority="SAME_MOVEMENT_EVENT",
    )


def _evidence(text: str) -> Evidence:
    return Evidence(
        id="ev1",
        author="A",
        work="W",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "d", "spine_index": 1, "start_offset": 1},
    )


def _classify(event: HistoricalEvent, obs, rel, query: str):
    ev = _evidence(event.summary)
    return classify_observation_relation_admission(
        rel,
        {item.observation_id: item for item in obs},
        {event.id: event},
        {ev.id: ev},
        (query,),
    )


def _assemble(event: HistoricalEvent, obs, rel, query: str):
    ev = _evidence(event.summary)
    return assemble_observation_components(
        list(obs),
        [rel],
        [event],
        [ev],
        query_contexts=(query,),
    )


def _component_has_rel(assembly, rel) -> bool:
    edge = (rel.earlier_observation_id, rel.later_observation_id)
    return any(edge in component.relation_ids for component in assembly.components)


def test_binding_identity_direct_same_event_canonical_admit_before_component():
    statement = "Commander Alpha marched from Harborwest to Cityeast."
    event = _event(
        statement,
        origin_mention="Portwest",
        origin_canonical="Harborwest",
        dest_mention="Cityeast",
        dest_canonical="Cityeast",
    )
    obs = (
        _obs("o1", "Harborwest", EventPlaceRole.ORIGIN),
        _obs("o2", "Cityeast", EventPlaceRole.DESTINATION),
    )
    rel = _rel()
    query = "Trace Commander Alpha from Portwest to Cityeast."
    admission = _classify(event, obs, rel, query)
    assert admission.admitted is True
    assert admission.route_phase_match is AuthorityState.MATCH
    assembly = _assemble(event, obs, rel, query)
    assert _component_has_rel(assembly, rel)


def _negative_case(statement: str, actor: str, query: str, origin: str, dest: str):
    event = _event(
        statement,
        actor=actor,
        origin_mention=origin,
        origin_canonical=origin,
        dest_mention=dest,
        dest_canonical=dest,
    )
    obs = (
        _obs("o1", origin, EventPlaceRole.ORIGIN, actor=actor),
        _obs("o2", dest, EventPlaceRole.DESTINATION, actor=actor),
    )
    rel = _rel()
    admission = _classify(event, obs, rel, query)
    assembly = _assemble(event, obs, rel, query)
    assert admission.admitted is False
    assert not _component_has_rel(assembly, rel)


def test_wrong_subject_rejects_canonical_and_component():
    _negative_case(
        "Commander Beta marched from Port A to City B.",
        "Commander Beta",
        "Trace Commander Alpha from Port A to City B.",
        "Port A",
        "City B",
    )


def test_wrong_episode_rejects_canonical_and_component():
    _negative_case(
        "During Campaign Silver, Commander Alpha marched from Port A to City B.",
        "Commander Alpha",
        "Trace Commander Alpha from Port A to City B during Campaign Gold.",
        "Port A",
        "City B",
    )


def test_episode_unknown_rejects_canonical_and_component():
    _negative_case(
        "Commander Alpha marched from Port A to City B.",
        "Commander Alpha",
        "Trace Commander Alpha from Port A to City B during Campaign Gold.",
        "Port A",
        "City B",
    )


def test_unrelated_phase_rejects_canonical_and_component():
    _negative_case(
        "Commander Alpha marched from Station B to Gate C.",
        "Commander Alpha",
        "Trace Commander Alpha from Port A toward City D.",
        "Station B",
        "Gate C",
    )


def test_reverse_rejects_canonical_and_component():
    _negative_case(
        "Commander Alpha marched from City B to Port A.",
        "Commander Alpha",
        "Trace Commander Alpha from Port A to City B.",
        "City B",
        "Port A",
    )


def test_ambiguous_query_rejects_canonical_and_component():
    _negative_case(
        "Commander Alpha marched from Port A to City B.",
        "Commander Alpha",
        "Trace Commander Alpha from Port A to City B then City C to City D.",
        "Port A",
        "City B",
    )


def test_generic_direct_same_event_admits_canonically():
    statement = "Commander Alpha marched from Port A to City B."
    event = _event(statement)
    obs = (
        _obs("o1", "Port A", EventPlaceRole.ORIGIN),
        _obs("o2", "City B", EventPlaceRole.DESTINATION),
    )
    rel = _rel()
    query = "Trace Commander Alpha from Port A to City B."
    admission = _classify(event, obs, rel, query)
    assert admission.admitted is True
    assert admission.subject_match is AuthorityState.MATCH
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.movement_assertion is AuthorityState.MATCH
    assert _component_has_rel(_assemble(event, obs, rel, query), rel)
