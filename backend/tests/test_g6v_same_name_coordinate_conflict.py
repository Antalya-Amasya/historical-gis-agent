"""G6V: same-name coordinate conflicts must remain local to affected components."""
from __future__ import annotations

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.tests.test_event_anchor_routes import build, evidence, movement


def place_at(
    name: str,
    latitude: float,
    longitude: float,
    *,
    place_id: str | None = None,
) -> HistoricalPlace:
    return HistoricalPlace(
        id=place_id or name.lower(),
        canonical_name=name,
        latitude=latitude,
        longitude=longitude,
        source="test registry",
        confidence=0.8,
        coordinate_role="exact_site",
    )


def binding_at(
    name: str,
    role: EventPlaceRole,
    refs: list[str],
    latitude: float,
    longitude: float,
    *,
    place_id: str | None = None,
) -> HistoricalEventPlaceBinding:
    mention = HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=refs)
    return HistoricalEventPlaceBinding(
        mention=mention,
        place=place_at(name, latitude, longitude, place_id=place_id),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=refs,
        resolver_provenance="registry",
    )


def movement_event(
    identifier: str,
    origin: str,
    destination: str,
    refs: list[str],
    *,
    origin_coords: tuple[float, float],
    destination_coords: tuple[float, float],
    origin_place_id: str | None = None,
    destination_place_id: str | None = None,
) -> HistoricalEvent:
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=f"{identifier} summary",
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        place_bindings=[
            binding_at(origin, EventPlaceRole.ORIGIN, refs, *origin_coords, place_id=origin_place_id),
            binding_at(
                destination,
                EventPlaceRole.DESTINATION,
                refs,
                *destination_coords,
                place_id=destination_place_id,
            ),
        ],
    )


def site_event(
    identifier: str,
    name: str,
    refs: list[str],
    *,
    latitude: float,
    longitude: float,
    place_id: str | None = None,
) -> HistoricalEvent:
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=f"{identifier} summary",
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        place_bindings=[
            binding_at(name, EventPlaceRole.EVENT_SITE, refs, latitude, longitude, place_id=place_id),
        ],
    )


def component_chains(outcome) -> set[tuple[str, ...]]:
    return {
        tuple(point.historical_place.canonical_name for point in component.ordered_points)
        for component in outcome.route.route_components
    }


def build_route(events, items):
    return EventAnchorRouteBuilder().build_with_diagnostics(
        events,
        items,
        event_id="test",
        name="Test",
        period="58 BCE",
    )


def test_a_gpt6_conflict_plus_unrelated_valid_component_survives():
    """GPT-6 F11: unrelated Gamma→Delta must survive a distant Beta coordinate conflict."""
    events = [
        movement_event(
            "ab",
            "Alpha",
            "Beta",
            ["a"],
            origin_coords=(45.0, 5.0),
            destination_coords=(45.5, 5.5),
            destination_place_id="beta-apollonia",
        ),
        site_event(
            "beta_other",
            "Beta",
            ["c"],
            latitude=44.0,
            longitude=4.0,
            place_id="beta-antioch",
        ),
        movement_event(
            "gd",
            "Gamma",
            "Delta",
            ["b"],
            origin_coords=(46.0, 6.0),
            destination_coords=(46.5, 6.5),
        ),
    ]
    outcome = build_route(events, [evidence("a"), evidence("b"), evidence("c", document="other")])
    assert outcome.route is not None
    assert ("Gamma", "Delta") in component_chains(outcome)
    assert outcome.diagnostics["reason_codes"] == ["PARTIAL_ROUTE"]


def test_b_same_name_different_resolved_place_ids_not_merged():
    events = [
        movement_event(
            "left",
            "Alpha",
            "Beta",
            ["a"],
            origin_coords=(45.0, 5.0),
            destination_coords=(45.5, 5.5),
            destination_place_id="beta-west",
        ),
        movement_event(
            "right",
            "Gamma",
            "Beta",
            ["b"],
            origin_coords=(46.0, 6.0),
            destination_coords=(44.0, 4.0),
            destination_place_id="beta-east",
        ),
    ]
    outcome = build_route(events, [evidence("a"), evidence("b", document="other")])
    assert outcome.route is not None
    beta_points = [
        point
        for component in outcome.route.route_components
        for point in component.ordered_points
        if point.historical_place.canonical_name == "Beta"
    ]
    assert len(beta_points) == 2
    assert len({(point.historical_place.latitude, point.historical_place.longitude) for point in beta_points}) == 2


def test_c_same_name_same_resolved_place_groups_normally():
    events = [
        movement_event(
            "ab",
            "Alpha",
            "Beta",
            ["a"],
            origin_coords=(45.0, 5.0),
            destination_coords=(45.5, 5.5),
            destination_place_id="beta-shared",
        ),
        movement_event(
            "bc",
            "Beta",
            "Gamma",
            ["b"],
            origin_coords=(45.5, 5.5),
            destination_coords=(46.0, 6.0),
            origin_place_id="beta-shared",
        ),
    ]
    outcome = build_route(events, [evidence("a"), evidence("b")])
    assert outcome.route is not None
    beta_coords = {
        (point.historical_place.latitude, point.historical_place.longitude)
        for component in outcome.route.route_components
        for point in component.ordered_points
        if point.historical_place.canonical_name == "Beta"
    }
    assert beta_coords == {(45.5, 5.5)}


def test_d_local_conflict_does_not_abort_global_build():
    events = [
        movement_event(
            "ab",
            "Alpha",
            "Beta",
            ["a"],
            origin_coords=(45.0, 5.0),
            destination_coords=(45.5, 5.5),
            destination_place_id="beta-one",
        ),
        movement_event(
            "bc",
            "Beta",
            "Gamma",
            ["b"],
            origin_coords=(44.0, 4.0),
            destination_coords=(46.0, 6.0),
            origin_place_id="beta-two",
        ),
        movement_event(
            "gd",
            "Gamma",
            "Delta",
            ["c"],
            origin_coords=(46.0, 6.0),
            destination_coords=(46.5, 6.5),
        ),
    ]
    outcome = build_route(events, [evidence("a"), evidence("b"), evidence("c", document="other")])
    assert outcome.route is not None
    assert ("Alpha", "Beta", "Gamma") not in component_chains(outcome)
    assert ("Gamma", "Delta") in component_chains(outcome)


def test_e_independent_valid_component_preserved_when_other_invalid():
    outcome = build_route(
        [
            movement_event(
                "ab",
                "Alpha",
                "Beta",
                ["a"],
                origin_coords=(45.0, 5.0),
                destination_coords=(45.5, 5.5),
                destination_place_id="beta-one",
            ),
            movement_event(
                "bc",
                "Beta",
                "Gamma",
                ["b"],
                origin_coords=(44.0, 4.0),
                destination_coords=(46.0, 6.0),
                origin_place_id="beta-two",
            ),
            movement("gd", "Gamma", "Delta", ["c"]),
        ],
        [evidence("a"), evidence("b"), evidence("c", document="other")],
    )
    assert outcome.route is not None
    assert ("Alpha", "Beta") in component_chains(outcome) or ("Beta", "Gamma") in component_chains(outcome)
    assert ("Gamma", "Delta") in component_chains(outcome)


def test_f_provenance_isolated_for_same_name_occurrences():
    events = [
        movement_event(
            "ab",
            "Alpha",
            "Beta",
            ["a"],
            origin_coords=(45.0, 5.0),
            destination_coords=(45.5, 5.5),
            destination_place_id="beta-west",
        ),
        movement_event(
            "gb",
            "Gamma",
            "Beta",
            ["b"],
            origin_coords=(46.0, 6.0),
            destination_coords=(44.0, 4.0),
            destination_place_id="beta-east",
        ),
    ]
    outcome = build_route(events, [evidence("a"), evidence("b", document="other")])
    assert outcome.route is not None
    alpha_beta = next(
        component
        for component in outcome.route.route_components
        if [point.historical_place.canonical_name for point in component.ordered_points] == ["Alpha", "Beta"]
    )
    gamma_beta = next(
        component
        for component in outcome.route.route_components
        if [point.historical_place.canonical_name for point in component.ordered_points] == ["Gamma", "Beta"]
    )
    assert alpha_beta.ordered_points[-1].evidence_refs == ["a"]
    assert gamma_beta.ordered_points[-1].evidence_refs == ["b"]
