"""G6CX: strong route-point eligibility guard for event-first and legacy fallback."""

from __future__ import annotations

import pytest

from backend.app.geography.feature_semantics import exact_anchor_eligible
from backend.app.models import (
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.event_anchors import project_event_anchors
from backend.app.routes.extractor import HistoricalRouteExtractor


def _evidence(identifier: str = "e1") -> Evidence:
    return Evidence(id=identifier, author="a", work="w", locator="l", excerpt="x", text="x")


def _place(
    *,
    semantics: PlaceSpatialSemantics,
    coordinate_role: str,
    name: str = "Place",
) -> HistoricalPlace:
    return HistoricalPlace(
        id=f"fixture-{name.lower()}",
        canonical_name=name,
        latitude=41.0,
        longitude=12.0,
        source="fixture",
        confidence=0.8,
        spatial_semantics=semantics,
        coordinate_role=coordinate_role,
    )


def _binding(role: EventPlaceRole, place: HistoricalPlace) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=place.canonical_name, role=role, evidence_refs=["e1"]),
        place=place,
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=["e1"],
        resolver_provenance="fixture",
        limitations=["fixture"],
    )


def _event(*bindings: HistoricalEventPlaceBinding) -> HistoricalEvent:
    return HistoricalEvent(
        id="evt-g6cx",
        name="movement",
        summary="movement",
        event_type=HistoricalEventType.MOVEMENT,
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
        evidence_refs=["e1"],
        source_statements=["movement"],
        place_bindings=list(bindings),
    )


@pytest.mark.parametrize(
    ("semantics", "coordinate_role", "role"),
    [
        (PlaceSpatialSemantics.REGION, "regional_centroid", EventPlaceRole.ORIGIN),
        (PlaceSpatialSemantics.ISLAND, "feature_centroid", EventPlaceRole.DESTINATION),
        (PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid", EventPlaceRole.DESTINATION),
        (PlaceSpatialSemantics.UNKNOWN, "representative_point", EventPlaceRole.ORIGIN),
        (PlaceSpatialSemantics.SETTLEMENT, "representative_point", EventPlaceRole.EVENT_SITE),
    ],
)
def test_strong_roles_reject_non_exact_coordinates(
    semantics: PlaceSpatialSemantics,
    coordinate_role: str,
    role: EventPlaceRole,
):
    place = _place(semantics=semantics, coordinate_role=coordinate_role, name=f"{semantics.value}-{coordinate_role}")
    anchors, diagnostics = project_event_anchors([_event(_binding(role, place))], [_evidence()])
    assert anchors == []
    assert any("NON_EXACT_FEATURE_ANCHOR" in code for code in diagnostics)
    assert not exact_anchor_eligible(place, strong_role=True)


@pytest.mark.parametrize(
    ("semantics", "coordinate_role", "role"),
    [
        (PlaceSpatialSemantics.PORT, "exact_site", EventPlaceRole.ORIGIN),
        (PlaceSpatialSemantics.PASS, "exact_site", EventPlaceRole.DESTINATION),
    ],
)
def test_strong_roles_allow_exact_site_coordinates(
    semantics: PlaceSpatialSemantics,
    coordinate_role: str,
    role: EventPlaceRole,
):
    place = _place(semantics=semantics, coordinate_role=coordinate_role, name=f"{semantics.value}-{coordinate_role}")
    anchors, diagnostics = project_event_anchors([_event(_binding(role, place))], [_evidence()])
    assert len(anchors) == 1
    assert anchors[0].role is role
    assert diagnostics == []
    assert exact_anchor_eligible(place, strong_role=True)


def test_resolved_region_remains_on_event_after_anchor_rejection():
    region = _place(semantics=PlaceSpatialSemantics.REGION, coordinate_role="regional_centroid", name="Italia")
    event = _event(_binding(EventPlaceRole.ORIGIN, region))
    anchors, diagnostics = project_event_anchors([event], [_evidence()])
    assert anchors == []
    assert any("NON_EXACT_FEATURE_ANCHOR" in code for code in diagnostics)
    binding = event.place_bindings[0]
    assert binding.resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert binding.place is region
    assert binding.place.spatial_semantics is PlaceSpatialSemantics.REGION


class LegacyGeography:
    def __init__(self, places: dict[str, dict]):
        self.places = places

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        payload = self.places.get(name)
        if payload is None:
            return {"found": False}
        return {"found": True, **payload}


def _legacy_outcome(*texts: str, places: dict[str, dict]):
    evidence = [
        Evidence(
            id=f"ev{i}",
            author="a",
            work="w",
            locator=str(i),
            excerpt=text,
            text=text,
            metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": i * 100},
        )
        for i, text in enumerate(texts)
    ]
    return HistoricalRouteExtractor(LegacyGeography(places)).build_with_diagnostics(
        evidence,
        event_id="g6cx",
        name="G6CX",
        period="100 BCE",
    )


def test_legacy_rejects_region_centroid_route_point():
    places = {
        "Italia": {
            "id": "fixture-italia",
            "canonical_name": "Italia",
            "latitude": 41.9,
            "longitude": 12.5,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "region",
            "coordinate_role": "regional_centroid",
        },
        "Ostia": {
            "id": "fixture-ostia",
            "canonical_name": "Ostia",
            "latitude": 41.76,
            "longitude": 12.28,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "port",
            "coordinate_role": "exact_site",
        },
    }
    outcome = _legacy_outcome("The army marched from Ostia to Italia.", places=places)
    assert outcome.route is None
    assert outcome.diagnostics["reason_codes"] == ["NON_EXACT_ROUTE_POINT"]


def test_legacy_rejects_island_centroid_route_point():
    places = {
        "Cyprus": {
            "id": "fixture-cyprus",
            "canonical_name": "Cyprus",
            "latitude": 35.0,
            "longitude": 33.0,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "island",
            "coordinate_role": "feature_centroid",
        },
        "Ostia": {
            "id": "fixture-ostia",
            "canonical_name": "Ostia",
            "latitude": 41.76,
            "longitude": 12.28,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "port",
            "coordinate_role": "exact_site",
        },
    }
    outcome = _legacy_outcome("The army marched from Ostia to Cyprus.", places=places)
    assert outcome.route is None
    assert outcome.diagnostics["reason_codes"] == ["NON_EXACT_ROUTE_POINT"]


def test_legacy_preserves_exact_port_site_route_point():
    places = {
        "Ostia": {
            "id": "fixture-ostia",
            "canonical_name": "Ostia",
            "latitude": 41.76,
            "longitude": 12.28,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "port",
            "coordinate_role": "exact_site",
        },
        "Brundisium": {
            "id": "fixture-brundisium",
            "canonical_name": "Brundisium",
            "latitude": 40.6,
            "longitude": 17.9,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "port",
            "coordinate_role": "exact_site",
        },
    }
    outcome = _legacy_outcome("The army marched from Ostia to Brundisium.", places=places)
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == ["Ostia", "Brundisium"]


def test_legacy_mixed_route_rejects_on_first_non_exact_anchor():
    places = {
        "Ostia": {
            "id": "fixture-ostia",
            "canonical_name": "Ostia",
            "latitude": 41.76,
            "longitude": 12.28,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "port",
            "coordinate_role": "exact_site",
        },
        "Italia": {
            "id": "fixture-italia",
            "canonical_name": "Italia",
            "latitude": 41.9,
            "longitude": 12.5,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "region",
            "coordinate_role": "regional_centroid",
        },
        "Brundisium": {
            "id": "fixture-brundisium",
            "canonical_name": "Brundisium",
            "latitude": 40.6,
            "longitude": 17.9,
            "source": "fixture",
            "confidence": 0.8,
            "spatial_semantics": "port",
            "coordinate_role": "exact_site",
        },
    }
    outcome = _legacy_outcome(
        "The army marched from Ostia to Italia.",
        "The army marched from Italia to Brundisium.",
        places=places,
    )
    assert outcome.route is None
    assert outcome.diagnostics["reason_codes"] == ["NON_EXACT_ROUTE_POINT"]
