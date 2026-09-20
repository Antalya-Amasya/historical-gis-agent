"""R2-FINAL-R1.1: query-time is a canonical QueryRouteScope authority."""
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
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.query_scope_parser import QuerySpanRole, parse_query_scope
from backend.app.routes.route_observations import (
    ObservationOrderingAuthority,
    ObservationOrderingRelation,
    RouteObservation,
    RouteObservationKind,
)
from backend.tests.test_g6ds_route_assembly_acceptance import _pair, POSITIVE_FIRST, POSITIVE_SECOND, QUERY


def _place(name: str) -> HistoricalPlace:
    return HistoricalPlace(
        id=name.casefold().replace(" ", "-"),
        canonical_name=name,
        latitude=40.0,
        longitude=19.0,
        source="test",
        confidence=0.8,
    )


def _binding(role: EventPlaceRole, name: str) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=["ev1"]),
        place=_place(name),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=["ev1"],
        resolver_provenance="test",
    )


def _grounding(year: int | None, *, conflict: bool = False, start: int | None = None, end: int | None = None):
    if conflict:
        return HistoricalEventTemporalGrounding(status=TemporalGroundingStatus.CONFLICT)
    if year is None and start is None:
        return HistoricalEventTemporalGrounding()
    start_y = start if start is not None else year
    end_y = end if end is not None else year
    return HistoricalEventTemporalGrounding(
        raw_expression=str(start_y),
        normalized_start=str(start_y),
        normalized_end=str(end_y),
        precision=TemporalPrecision.YEAR_RANGE if start_y != end_y else TemporalPrecision.YEAR,
        evidence_refs=["ev1"],
        status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
    )


def _event(
    statement: str,
    *,
    actor: str = "Commander Alpha",
    origin: str = "Port A",
    dest: str = "City B",
    event_id: str = "move",
    year: int | None = None,
    start: int | None = None,
    end: int | None = None,
    conflict: bool = False,
) -> HistoricalEvent:
    return HistoricalEvent(
        id=event_id,
        name=event_id,
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        source_statements=[statement],
        actor=HistoricalEventActorGrounding(actor_text=actor, actor_status=EventActorStatus.EXPLICIT),
        place_bindings=[_binding(EventPlaceRole.ORIGIN, origin), _binding(EventPlaceRole.DESTINATION, dest)],
        temporal_grounding=_grounding(year, conflict=conflict, start=start, end=end),
    )


def _obs(oid: str, label: str, role: EventPlaceRole, event_id: str = "move", actor: str = "Commander Alpha"):
    return RouteObservation(
        observation_id=oid,
        kind=RouteObservationKind.PLACE,
        event_id=event_id,
        label=label,
        actor_text=actor,
        actor_status=EventActorStatus.EXPLICIT,
        evidence_refs=("ev1",),
        place_role=role,
    )


def _rel(*event_ids: str, authority=ObservationOrderingAuthority.AFTER_SUBORDINATE, earlier="o1", later="o2"):
    ids = event_ids or ("move",)
    return ObservationOrderingRelation(
        earlier_observation_id=earlier,
        later_observation_id=later,
        ordering_rule=authority,
        event_ids=ids,
        evidence_refs=("ev1",),
        authority=authority.value if authority is not ObservationOrderingAuthority.AFTER_SUBORDINATE else "SAME_MOVEMENT_EVENT",
    )


def _evidence(text: str) -> Evidence:
    return Evidence(
        id="ev1", author="A", work="W", locator="1", excerpt=text, text=text,
        metadata={"document_id": "d", "spine_index": 1, "start_offset": 1},
    )


def _classify(event, obs, rel, query: str, extra_events=()):
    ev = _evidence(event.summary)
    events = {event.id: event}
    events.update({item.id: item for item in extra_events})
    return classify_observation_relation_admission(
        rel, {item.observation_id: item for item in obs}, events, {ev.id: ev}, (query,),
    )


def test_parser_owns_explicit_query_time_not_episode():
    query = "Trace Commander Alpha's route in 200 BCE from Port A to City B."
    parsed = parse_query_scope(query)
    scope = parse_query_route_scope((query,))
    assert scope.subject == "Commander Alpha"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.episode is None
    assert scope.has_episode_constraint is False
    assert scope.has_temporal_constraint is True
    assert scope.temporal_start == -200
    assert scope.temporal_end == -200
    assert any(span.role is QuerySpanRole.TIME for span in parsed.spans)


def test_parser_year_forms():
    assert parse_query_route_scope(("Trace Alpha from A to B in 200 BC.",)).temporal_start == -200
    assert parse_query_route_scope(("Trace Alpha from A to B in AD 50.",)).temporal_start == 50
    assert parse_query_route_scope(("Trace Alpha from A to B in 50 CE.",)).temporal_start == 50
    ranged = parse_query_route_scope(("Trace Alpha from A to B in 210-190 BCE.",))
    assert ranged.temporal_start == -210
    assert ranged.temporal_end == -190


def test_time_does_not_steal_endpoints_or_episode():
    parsed = parse_query_scope("Trace Commander Alpha from Year Hill to Port A during Campaign Gold.")
    assert parsed.origin == "Year Hill"
    assert parsed.destination == "Port A"
    assert parsed.episode == "Campaign Gold"
    assert parsed.has_temporal_constraint is False


def test_same_event_temporal_matrix():
    statement = "Commander Alpha marched from Port A to City B."
    query_dated = "Trace Commander Alpha from Port A to City B in 200 BCE."
    query_open = "Trace Commander Alpha from Port A to City B."
    obs = (_obs("o1", "Port A", EventPlaceRole.ORIGIN), _obs("o2", "City B", EventPlaceRole.DESTINATION))
    rel = _rel()
    match = _classify(_event(statement, year=-200), obs, rel, query_dated)
    assert match.temporal_match is AuthorityState.MATCH and match.admitted is True
    wrong = _classify(_event(statement, year=-150), obs, rel, query_dated)
    assert wrong.temporal_match is AuthorityState.WRONG and wrong.admitted is False
    unknown = _classify(_event(statement), obs, rel, query_dated)
    assert unknown.temporal_match is AuthorityState.UNKNOWN and unknown.admitted is False
    inactive = _classify(_event(statement), obs, rel, query_open)
    assert inactive.admitted is True


def test_range_contains_query_year():
    statement = "Commander Alpha marched from Port A to City B."
    obs = (_obs("o1", "Port A", EventPlaceRole.ORIGIN), _obs("o2", "City B", EventPlaceRole.DESTINATION))
    admission = _classify(_event(statement, start=-210, end=-190), obs, _rel(), "Trace Commander Alpha from Port A to City B in 200 BCE.")
    assert admission.temporal_match is AuthorityState.MATCH
    assert admission.admitted is True


def test_episode_and_time_are_independent():
    statement = "During Campaign Gold, Commander Alpha marched from Port A to City B."
    obs = (_obs("o1", "Port A", EventPlaceRole.ORIGIN), _obs("o2", "City B", EventPlaceRole.DESTINATION))
    rel = _rel()
    query = "Trace Commander Alpha from Port A to City B during Campaign Gold in 200 BCE."
    wrong_year = _classify(_event(statement, year=-150), obs, rel, query)
    assert wrong_year.episode_match is AuthorityState.MATCH
    assert wrong_year.temporal_match is AuthorityState.WRONG
    assert wrong_year.admitted is False


def test_inter_event_does_not_lend_date():
    query = "Trace Commander Alpha from Port A toward City D in 200 BCE."
    a = _event("Commander Alpha was at Port A.", origin="Port A", dest="Port A", event_id="e1", year=-200)
    b = _event("Then Commander Alpha reached City B.", origin="City B", dest="City B", event_id="e2")
    obs = (
        _obs("o1", "Port A", EventPlaceRole.ORIGIN, event_id="e1"),
        _obs("o2", "City B", EventPlaceRole.DESTINATION, event_id="e2"),
    )
    rel = _rel("e1", "e2", authority=ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER)
    admission = _classify(a, obs, rel, query, extra_events=(b,))
    assert admission.temporal_match is AuthorityState.UNKNOWN
    assert admission.admitted is False


def test_g6ds_undated_bridge_rejected():
    first = "In 200 BCE Ariston marched from Roma to Capua."
    second = "Then Ariston sailed from Brundisium to Corcyra."
    e1, e2, evidence_by_id = _pair(first, second)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6ds", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    names = [] if outcome.route is None else [p.historical_place.canonical_name for p in outcome.route.ordered_points]
    assert "Corcyra" not in names
    assert not (names == ["Roma", "Capua", "Brundisium", "Corcyra"])


def test_g6ds_dated_positive_preserved():
    e1, e2, evidence_by_id = _pair(POSITIVE_FIRST, POSITIVE_SECOND)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6ds", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    assert outcome.route is not None
    names = [p.historical_place.canonical_name for p in outcome.route.ordered_points]
    assert names == ["Roma", "Capua", "Brundisium", "Corcyra"]


def test_no_time_query_undated_not_blocked():
    e1, e2, evidence_by_id = _pair(POSITIVE_FIRST, POSITIVE_SECOND)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6ds", name="Ariston", period="200 BCE",
        query_contexts=("Trace Ariston's route from Capua to Brundisium.",),
    )
    assert outcome.route is not None


def test_canonical_temporal_unknown_not_restored_by_component():
    statement = "Commander Alpha marched from Port A to City B."
    event = _event(statement)
    obs = [_obs("o1", "Port A", EventPlaceRole.ORIGIN), _obs("o2", "City B", EventPlaceRole.DESTINATION)]
    rel = _rel()
    query = "Trace Commander Alpha from Port A to City B in 200 BCE."
    admission = _classify(event, obs, rel, query)
    assembly = assemble_observation_components(obs, [rel], [event], [_evidence(statement)], query_contexts=(query,))
    assert admission.admitted is False
    assert not any((rel.earlier_observation_id, rel.later_observation_id) in c.relation_ids for c in assembly.components)
