"""R2-FINAL-C1: coordinate authority separated from historical waypoint admission."""

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
    PlaceSpatialSemantics,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.tests.test_g6dr_connector_relation_admission import QUERY, _pair as g6ds_pair, _structural
from backend.tests.test_g6ds_route_assembly_acceptance import POSITIVE_FIRST


COORDS = {
    "Port A": (40.0, 18.0),
    "River B": (40.2, 17.8),
    "Island C": (40.4, 17.6),
    "Fort X": (41.2, 16.8),
    "Gate Q": (41.4, 16.4),
    "City B": (41.0, 15.0),
    "City D": (42.0, 14.0),
    "Station B": (40.5, 17.5),
    "Gate C": (41.5, 16.5),
    "Roma": (41.9, 12.5),
    "Capua": (41.08, 14.25),
    "Brundisium": (40.6, 17.9),
    "Corcyra": (39.6, 19.9),
}


def _evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="test",
        work="test",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1"},
    )


def _place(name: str, *, coordinate_role: str = "exact_site") -> HistoricalPlace:
    latitude, longitude = COORDS[name]
    return HistoricalPlace(
        id=name.lower().replace(" ", "-"),
        canonical_name=name,
        latitude=latitude,
        longitude=longitude,
        source="test",
        confidence=0.8,
        spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
        coordinate_role=coordinate_role,
    )


def _binding(name: str, role: EventPlaceRole, refs: list[str], *, coordinate_role: str = "exact_site"):
    place = _place(name, coordinate_role=coordinate_role)
    mention = HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=refs)
    return HistoricalEventPlaceBinding(
        mention=mention,
        place=place,
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=refs,
        resolver_provenance="test",
        limitations=[f"fixture:{coordinate_role}"],
    )


def _movement(
    identifier: str,
    origin: str,
    destination: str,
    refs: list[str],
    *,
    origin_role: str = "exact_site",
    destination_role: str = "exact_site",
    actor_text: str = "Commander Alpha",
) -> HistoricalEvent:
    statement = f"{actor_text} marched from {origin} to {destination}."
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        source_statements=[statement],
        place_bindings=[
            _binding(origin, EventPlaceRole.ORIGIN, refs, coordinate_role=origin_role),
            _binding(destination, EventPlaceRole.DESTINATION, refs, coordinate_role=destination_role),
        ],
        actor=HistoricalEventActorGrounding(
            actor_text=actor_text,
            actor_status=EventActorStatus.EXPLICIT,
            actor_tokens=(actor_text,),
        ),
    )


def _build(events: list[HistoricalEvent], evidence_by_id: dict[str, Evidence], *, query_contexts=None):
    return EventAnchorRouteBuilder().build_with_diagnostics(
        events,
        list(evidence_by_id.values()),
        event_id="c1",
        name=events[0].actor.actor_text or "Actor",
        period="49 BCE",
        query_contexts=query_contexts,
    )


def test_representative_point_destination_survives_canonical_route():
    text = "Commander Alpha marched from Port A to City B."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Port A", "City B", ["ev1"], destination_role="representative_point")
    outcome = _build([event], {"ev1": evidence})
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == ["Port A", "City B"]
    city_b = outcome.route.ordered_points[-1]
    assert city_b.coordinate_role == "representative_point"
    assert city_b.historical_place.coordinate_role == "representative_point"
    assert outcome.diagnostics["observation_component_count"] >= 1


def test_feature_centroid_destination_survives_without_exact_upgrade():
    text = "Commander Alpha marched from Port A to City B."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Port A", "City B", ["ev1"], destination_role="feature_centroid")
    outcome = _build([event], {"ev1": evidence})
    assert outcome.route is not None
    city_b = outcome.route.ordered_points[-1]
    assert city_b.coordinate_role == "feature_centroid"
    assert city_b.historical_place.coordinate_role != "exact_site"


def test_exact_site_control_unchanged():
    text = "Commander Alpha marched from Port A to City B."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Port A", "City B", ["ev1"])
    outcome = _build([event], {"ev1": evidence})
    assert outcome.route is not None
    assert all(point.coordinate_role == "exact_site" for point in outcome.route.ordered_points)


def test_unresolved_place_remains_absent():
    text = "Commander Alpha marched from Port A to City B."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Port A", "City B", ["ev1"])
    destination = event.place_bindings[1].model_copy(update={"resolution_status": EventPlaceResolutionStatus.UNRESOLVED, "place": None})
    event = event.model_copy(update={"place_bindings": [event.place_bindings[0], destination]})
    outcome = _build([event], {"ev1": evidence})
    assert outcome.route is None or len(outcome.route.ordered_points) < 2


def test_g6ds_approximate_waypoint_chain_preserves_topology_and_roles():
    first = POSITIVE_FIRST
    second = "Then in 200 BCE Ariston sailed from Brundisium to Corcyra."
    e1, e2, evidence_by_id = g6ds_pair(first, second)
    origin_binding, destination_binding = e1.place_bindings
    e1 = e1.model_copy(update={
        "place_bindings": [
            origin_binding,
            destination_binding.model_copy(update={
                "place": destination_binding.place.model_copy(update={"coordinate_role": "representative_point"}),
            }),
        ],
    })
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2],
        list(evidence_by_id.values()),
        event_id="g6ds-c1",
        name="Ariston",
        period="200 BCE",
        query_contexts=QUERY,
    )
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == [
        "Roma", "Capua", "Brundisium", "Corcyra",
    ]
    capua = next(point for point in outcome.route.ordered_points if point.historical_place.canonical_name == "Capua")
    assert capua.coordinate_role == "representative_point"
    assert capua.historical_place.coordinate_role != "exact_site"
    assert _structural(outcome.relations)


def test_g6ds_feature_centroid_waypoint_preserves_roles_without_upgrade():
    first = POSITIVE_FIRST
    second = "Then in 200 BCE Ariston sailed from Brundisium to Corcyra."
    e1, e2, evidence_by_id = g6ds_pair(first, second)
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
        [e1, e2],
        list(evidence_by_id.values()),
        event_id="g6ds-c1",
        name="Ariston",
        period="200 BCE",
        query_contexts=QUERY,
    )
    assert outcome.route is not None
    brundisium = next(
        point for point in outcome.route.ordered_points
        if point.historical_place.canonical_name == "Brundisium"
    )
    assert brundisium.coordinate_role == "feature_centroid"
    assert brundisium.historical_place.coordinate_role != "exact_site"
    assert any(code.startswith("NON_EXACT_FEATURE_ANCHOR:") for code in outcome.diagnostics["projection_diagnostics"])


def test_mandatory_unrelated_negative_remains_closed():
    query = ("Trace Commander Delta from Port A toward City D.",)
    text = "Commander Delta marched from Station B to Gate C."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Station B", "Gate C", ["ev1"], actor_text="Commander Delta")
    outcome = _build([event], {"ev1": evidence}, query_contexts=query)
    assert outcome.route is None
    assert outcome.diagnostics["reason_codes"] == ["CANONICAL_ROUTE_ABSENT"]


def test_libo_control_still_builds_route():
    from backend.tests.test_v1b3_same_movement_direct_subject_admission import (
        test_libo_oricum_brundisium_real_control_builds_route,
    )

    test_libo_oricum_brundisium_real_control_builds_route()
