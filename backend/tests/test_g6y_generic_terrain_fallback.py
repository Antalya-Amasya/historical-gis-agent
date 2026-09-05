"""G6Y: production terrain fallback must not depend on historical subject/campaign identity."""

from backend.app.models import (
    Evidence,
    GeoJsonLineString,
    HistoricalPlace,
    HistoricalRoute,
    HistoricalRouteIntent,
    HistoricalRoutePoint,
    PlaceSpatialSemantics,
)
from backend.app.route_orchestrator import HistoricalRouteOrchestrator


def _mediterranean_route() -> HistoricalRoute:
    points = [
        HistoricalRoutePoint(
            sequence=1,
            historical_place=HistoricalPlace(
                id="nova",
                canonical_name="Carthago Nova",
                longitude=-0.98452,
                latitude=37.599896,
                source="fixture",
                confidence=0.9,
                spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
            ),
            event_summary="Departed from New Carthage.",
            evidence_refs=["e-nova"],
            confidence=0.9,
            coordinate_role="exact_site",
        ),
        HistoricalRoutePoint(
            sequence=2,
            historical_place=HistoricalPlace(
                id="padus",
                canonical_name="Padus",
                longitude=12.432028,
                latitude=44.952389,
                source="fixture",
                confidence=0.9,
                spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
            ),
            event_summary="Reached the plains of the Padus.",
            evidence_refs=["e-padus"],
            confidence=0.9,
            coordinate_role="representative_point",
        ),
    ]
    return HistoricalRoute(
        id="med-route",
        event_id="med-event",
        name="Mediterranean movement fixture",
        period="218 BCE",
        ordered_points=points,
        geometry=GeoJsonLineString(
            coordinates=[(p.historical_place.longitude, p.historical_place.latitude) for p in points]
        ),
        evidence_refs=["e-nova", "e-padus"],
        historical_confidence=0.9,
    )


def _evidence() -> list[Evidence]:
    return [
        Evidence(id="e-nova", author="Fixture", work="Fixture", locator="nova", excerpt="nova"),
        Evidence(id="e-padus", author="Fixture", work="Fixture", locator="padus", excerpt="padus"),
    ]


def _present(campaign_id: str, entity: str) -> dict:
    orchestrator = HistoricalRouteOrchestrator(cell_size_m=5_000)
    response = orchestrator.present(
        HistoricalRouteIntent(campaign_id=campaign_id, entity=entity, route_type="movement"),
        _mediterranean_route(),
        _evidence(),
    )
    return {
        "coordinates": response.route_geojson["geometry"]["coordinates"],
        "applied_constraints": response.route_geojson["properties"]["applied_constraints"],
        "corridor_features": [
            feature
            for feature in response.geojson["features"]
            if feature["properties"].get("layer_type") == "uncertainty_corridor"
        ],
        "quality": response.route_geojson["properties"]["route_quality"],
    }


def test_a_subject_invariance_same_gis_inputs_same_fallback():
    """Same route geometry; only entity label differs."""
    demo = _present("hannibal_italy_campaign", "hannibal_alpine_crossing")
    other = _present("generic_campaign", "arbitrary_subject")
    assert demo["applied_constraints"] == other["applied_constraints"]
    assert demo["coordinates"] == other["coordinates"]
    assert len(demo["corridor_features"]) == len(other["corridor_features"]) == 1


def test_b_campaign_invariance_same_endpoints_same_fallback():
    """Same endpoints/terrain; different campaign labels."""
    hannibal = _present("hannibal_italy_campaign", "hannibal_alpine_crossing")
    punic = _present("second_punic_war", "some_operation")
    assert hannibal["applied_constraints"] == punic["applied_constraints"]
    assert hannibal["coordinates"] == punic["coordinates"]


def test_c_generic_failure_does_not_emit_named_campaign_corridor():
    """Mock fallback must not expose Hannibal-only corridor regions."""
    result = _present("generic_campaign", "arbitrary_subject")
    labels = [feature["properties"]["label"] for feature in result["corridor_features"]]
    assert "Iberian east-coast approach" not in labels
    assert "Po plain approach" not in labels
    assert result["applied_constraints"]
    assert "mock_reviewed_corridor_mask" not in result["applied_constraints"]
    assert "mock_route_search_bounds" in result["applied_constraints"]


def test_d_evidence_constraints_preserve_waypoint_positions():
    route = _mediterranean_route()
    response = HistoricalRouteOrchestrator(cell_size_m=5_000).present(
        HistoricalRouteIntent(campaign_id="generic_campaign", entity="any", route_type="movement"),
        route,
        _evidence(),
    )
    coordinates = response.route_geojson["geometry"]["coordinates"]
    waypoint_coordinates = [
        [point.historical_place.longitude, point.historical_place.latitude]
        for point in route.ordered_points
    ]
    assert coordinates[0] == waypoint_coordinates[0]
    assert coordinates[-1] == waypoint_coordinates[-1]
    assert len(coordinates) >= len(waypoint_coordinates)


def test_e_production_orchestrator_path_is_exercised():
    """Exercise HistoricalRouteOrchestrator.present, not a private helper."""
    response = HistoricalRouteOrchestrator(cell_size_m=5_000).present(
        HistoricalRouteIntent(campaign_id="generic_campaign", entity="any", route_type="movement"),
        _mediterranean_route(),
        _evidence(),
    )
    assert response.route_geojson["properties"]["route_type"] == "terrain_aware_historical_reconstruction"
    assert response.route_geojson["properties"]["terrain_source"] == "offline_mock_terrain"


def test_f_ordinary_generic_route_reconstruction_still_works():
    """Non-demo coordinates still reconstruct through the production orchestrator."""
    points = [
        HistoricalRoutePoint(
            sequence=1,
            historical_place=HistoricalPlace(
                id="west",
                canonical_name="West",
                longitude=0.0,
                latitude=45.0,
                source="fixture",
                confidence=0.9,
            ),
            event_summary="fixture",
            evidence_refs=["e-west"],
            confidence=0.9,
        ),
        HistoricalRoutePoint(
            sequence=2,
            historical_place=HistoricalPlace(
                id="east",
                canonical_name="East",
                longitude=2.0,
                latitude=45.0,
                source="fixture",
                confidence=0.9,
            ),
            event_summary="fixture",
            evidence_refs=["e-east"],
            confidence=0.9,
        ),
    ]
    historical_route = HistoricalRoute(
        id="generic",
        event_id="generic",
        name="Generic",
        period="fixture",
        ordered_points=points,
        geometry=GeoJsonLineString(coordinates=[(0.0, 45.0), (2.0, 45.0)]),
        evidence_refs=["e-west", "e-east"],
        historical_confidence=0.9,
    )
    evidence = [
        Evidence(id="e-west", author="Fixture", work="Fixture", locator="w", excerpt="w"),
        Evidence(id="e-east", author="Fixture", work="Fixture", locator="e", excerpt="e"),
    ]
    response = HistoricalRouteOrchestrator(cell_size_m=25_000).present(
        HistoricalRouteIntent(campaign_id="generic_campaign", entity="generic", route_type="movement"),
        historical_route,
        evidence,
    )
    assert len(response.route_geojson["geometry"]["coordinates"]) >= 2
    assert response.route_geojson["properties"]["route_quality"]["waypoint_order_preserved"] is True
