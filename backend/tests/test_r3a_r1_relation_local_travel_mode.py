"""R3-A-R1: same-event travel mode is ordering-local, not event-wide."""
from __future__ import annotations

from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateStatus
from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    EventRouteOrdering,
    EventRouteOrderingAuthority,
    EventRouteOrderingEndpointKind,
    EventRouteOrderingRef,
    Evidence,
    HistoricalEvent,
    HistoricalEventActorGrounding,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
    HistoricalTravelMode,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.route_observations import project_observation_ordering


def _place(name: str, lat: float, lon: float) -> HistoricalPlace:
    return HistoricalPlace(
        id=name.casefold().replace(" ", "-"),
        canonical_name=name,
        latitude=lat,
        longitude=lon,
        source="test",
        confidence=0.8,
        coordinate_role="exact_site",
    )


def _binding(name: str, role: EventPlaceRole, lat: float, lon: float) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=["ev1"]),
        place=_place(name, lat, lon),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=["ev1"],
        resolver_provenance="test",
    )


def _ref(kind: EventRouteOrderingEndpointKind, name: str) -> EventRouteOrderingRef:
    return EventRouteOrderingRef(endpoint_kind=kind, surface=name, canonical=name)


def _ordering(earlier: str, later: str, statement: str) -> EventRouteOrdering:
    return EventRouteOrdering(
        earlier=_ref(EventRouteOrderingEndpointKind.ORIGIN, earlier),
        later=_ref(EventRouteOrderingEndpointKind.DESTINATION, later),
        authority=EventRouteOrderingAuthority.AFTER_SUBORDINATE,
        source_statement=statement,
        evidence_refs=["ev1"],
    )


def _event(orderings: list[EventRouteOrdering], statements: list[str] | None = None) -> HistoricalEvent:
    places = [
        ("Port A", EventPlaceRole.ORIGIN, 40.0, 19.0),
        ("Island B", EventPlaceRole.DESTINATION, 40.1, 19.5),
        ("City C", EventPlaceRole.ORIGIN, 41.0, 12.0),
        ("City D", EventPlaceRole.DESTINATION, 41.5, 12.5),
    ]
    texts = statements if statements is not None else [item.source_statement for item in orderings]
    return HistoricalEvent(
        id="move",
        name="move",
        summary=texts[0] if texts else "",
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        source_statements=texts,
        actor=HistoricalEventActorGrounding(actor_text="Commander Alpha", actor_status=EventActorStatus.EXPLICIT),
        place_bindings=[_binding(name, role, lat, lon) for name, role, lat, lon in places],
        route_orderings=orderings,
    )


def _evidence() -> Evidence:
    return Evidence(
        id="ev1", author="A", work="W", locator="1", excerpt="x", text="x",
        metadata={"document_id": "d", "spine_index": 1, "start_offset": 1},
    )


def _modes(event: HistoricalEvent) -> dict[tuple[str, str], HistoricalTravelMode]:
    observations, relations, _ = project_observation_ordering([event], [], [], [_evidence()])
    by_id = {item.observation_id: item for item in observations}
    return {
        (by_id[rel.earlier_observation_id].label, by_id[rel.later_observation_id].label): rel.travel_mode
        for rel in relations
        if rel.event_ids == ("move",)
    }


SAILED = "Commander Alpha sailed from Port A to Island B."
MOVED = "Commander Alpha moved from City C to City D."
MARCHED = "Commander Alpha marched from City C to City D."
MARCHED_AB = "Commander Alpha marched from Port A to Island B."


def test_sailed_and_generic_same_event_are_local():
    event = _event([_ordering("Port A", "Island B", SAILED), _ordering("City C", "City D", MOVED)])
    modes = _modes(event)
    assert modes[("Port A", "Island B")] is HistoricalTravelMode.SEA
    assert modes[("City C", "City D")] is HistoricalTravelMode.UNKNOWN


def test_reverse_ordering_and_statement_list_do_not_change_modes():
    forward = _modes(_event([_ordering("Port A", "Island B", SAILED), _ordering("City C", "City D", MOVED)]))
    reversed_orderings = _modes(_event([_ordering("City C", "City D", MOVED), _ordering("Port A", "Island B", SAILED)]))
    reversed_statements = _modes(_event(
        [_ordering("Port A", "Island B", SAILED), _ordering("City C", "City D", MOVED)],
        statements=[MOVED, SAILED],
    ))
    assert forward == reversed_orderings == reversed_statements
    assert forward[("Port A", "Island B")] is HistoricalTravelMode.SEA
    assert forward[("City C", "City D")] is HistoricalTravelMode.UNKNOWN


def test_marched_and_generic_same_event_are_local():
    modes = _modes(_event([_ordering("Port A", "Island B", MARCHED_AB), _ordering("City C", "City D", MOVED)]))
    assert modes[("Port A", "Island B")] is HistoricalTravelMode.LAND
    assert modes[("City C", "City D")] is HistoricalTravelMode.UNKNOWN


def test_sailed_and_marched_keep_distinct_modes():
    modes = _modes(_event([_ordering("Port A", "Island B", SAILED), _ordering("City C", "City D", MARCHED)]))
    assert modes[("Port A", "Island B")] is HistoricalTravelMode.SEA
    assert modes[("City C", "City D")] is HistoricalTravelMode.LAND


def test_missing_local_statement_is_unknown():
    modes = _modes(_event([_ordering("Port A", "Island B", ""), _ordering("City C", "City D", MOVED)], statements=[SAILED, MOVED]))
    assert modes[("Port A", "Island B")] is HistoricalTravelMode.UNKNOWN


def test_local_conflict_is_unknown():
    conflict = "Commander Alpha marched from Port A to Island B and sailed onward."
    modes = _modes(_event([_ordering("Port A", "Island B", conflict), _ordering("City C", "City D", MARCHED)]))
    assert modes[("Port A", "Island B")] is HistoricalTravelMode.UNKNOWN
    assert modes[("City C", "City D")] is HistoricalTravelMode.LAND


def test_mixed_three_leg_modes():
    event = HistoricalEvent(
        id="move",
        name="move",
        summary=MARCHED_AB,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        source_statements=[MARCHED_AB, SAILED, MOVED],
        actor=HistoricalEventActorGrounding(actor_text="Commander Alpha", actor_status=EventActorStatus.EXPLICIT),
        place_bindings=[
            _binding("Port A", EventPlaceRole.ORIGIN, 40.0, 19.0),
            _binding("Island B", EventPlaceRole.DESTINATION, 40.1, 19.5),
            _binding("City C", EventPlaceRole.DESTINATION, 41.0, 12.0),
            _binding("City D", EventPlaceRole.DESTINATION, 41.5, 12.5),
        ],
        route_orderings=[
            _ordering("Port A", "Island B", MARCHED_AB),
            EventRouteOrdering(
                earlier=_ref(EventRouteOrderingEndpointKind.DESTINATION, "Island B"),
                later=_ref(EventRouteOrderingEndpointKind.DESTINATION, "City C"),
                authority=EventRouteOrderingAuthority.AFTER_SUBORDINATE,
                source_statement=SAILED,
                evidence_refs=["ev1"],
            ),
            EventRouteOrdering(
                earlier=_ref(EventRouteOrderingEndpointKind.DESTINATION, "City C"),
                later=_ref(EventRouteOrderingEndpointKind.DESTINATION, "City D"),
                authority=EventRouteOrderingAuthority.AFTER_SUBORDINATE,
                source_statement=MOVED,
                evidence_refs=["ev1"],
            ),
        ],
    )
    modes = _modes(event)
    assert modes[("Port A", "Island B")] is HistoricalTravelMode.LAND
    assert modes[("Island B", "City C")] is HistoricalTravelMode.SEA
    assert modes[("City C", "City D")] is HistoricalTravelMode.UNKNOWN


def test_claims_and_planner_preserve_distinct_modes():
    event = _event([_ordering("Port A", "Island B", SAILED), _ordering("City C", "City D", MOVED)])
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event], [_evidence()], event_id="r3a", name="Alpha", period="200 BCE",
    )
    assert outcome.route is not None
    by_pair = {(claim.source_place, claim.destination_place): claim.travel_mode for claim in outcome.route.claims}
    assert by_pair[("Port A", "Island B")] is HistoricalTravelMode.SEA
    assert by_pair[("City C", "City D")] is HistoricalTravelMode.UNKNOWN

    calls: list[tuple[str, str]] = []

    class CandidateService:
        def build(self, *_args):
            source, dest = _args[0], _args[1]
            calls.append((source.historical_place.canonical_name, dest.historical_place.canonical_name))
            return type("Result", (), {"candidate": None, "status": RomanRoadCandidateStatus.DISCONNECTED, "limitation": "gap"})()

    sea_blocked = False
    unknown_eligible = False
    for component in outcome.route.route_components:
        fragment = outcome.route.model_copy(update={"ordered_points": component.ordered_points, "geometry": outcome.route.geometry})
        result = RomanRoadRouteOrchestrator(CandidateService()).build_roman_road_candidates(fragment)
        for leg in result.legs:
            if leg.travel_mode is HistoricalTravelMode.SEA:
                assert leg.failure_status == "MARITIME_PLANNER_UNAVAILABLE"
                sea_blocked = True
            if leg.travel_mode is HistoricalTravelMode.UNKNOWN:
                unknown_eligible = True
    assert sea_blocked
    assert unknown_eligible
    assert ("Port A", "Island B") not in calls
    assert ("City C", "City D") in calls
