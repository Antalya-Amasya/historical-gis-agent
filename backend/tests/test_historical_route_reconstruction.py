import pytest

from backend.app.candidate_routes.grid import GridPoint
from backend.app.candidate_routes.historical_reconstruction import (
    HistoricalRouteReconstructionError,
    HistoricalRouteReconstructor,
    OfflineMockTerrainGraphProvider,
    ReviewedHistoricalWaypoint,
)
from backend.app.candidate_routes.models import ArmyProfile


def hannibal_alps_waypoints():
    # Explicit offline fixture coordinates supplied by the test; no resolver or geocoder is involved.
    return [
        ReviewedHistoricalWaypoint(id="rhodanus-crossing", canonical_name="Rhodanus crossing region", longitude=5.00, latitude=45.20, evidence_refs=["polybius-book-3"], confidence=0.8),
        ReviewedHistoricalWaypoint(id="alps-approach", canonical_name="Alps approach region", longitude=5.30, latitude=45.45, evidence_refs=["polybius-book-3"], confidence=0.75),
    ]


def test_hannibal_alps_offline_reconstruction_returns_terrain_aware_candidate_geojson():
    waypoints = hannibal_alps_waypoints()
    graph = OfflineMockTerrainGraphProvider().build_graph(waypoints, padding_km=5, cell_size_m=5_000)
    reconstruction = HistoricalRouteReconstructor().reconstruct(
        waypoints, graph, army_profile=ArmyProfile(name="carthaginian_demo"), route_id="hannibal-alps-demo",
    )
    feature = reconstruction.to_geojson()
    assert feature["geometry"]["type"] == "LineString"
    assert feature["geometry"]["coordinates"][0] == [5.0, 45.2]
    assert feature["geometry"]["coordinates"][-1] == [5.3, 45.45]
    assert feature["properties"]["route_type"] == "terrain_aware_historical_reconstruction"
    assert reconstruction.candidate_paths[0].terrain_source == "offline_mock_terrain"
    assert reconstruction.evidence_refs == ["polybius-book-3"]


def test_reconstructor_uses_existing_astar_to_avoid_blocked_offline_terrain_cell():
    waypoints = hannibal_alps_waypoints()
    graph = OfflineMockTerrainGraphProvider().build_graph(waypoints, padding_km=5, cell_size_m=2_000)
    start, goal = graph.grid_point_for(waypoints[0]), graph.grid_point_for(waypoints[1])
    blocked = GridPoint((start.x + goal.x) // 2, start.y)
    assert blocked not in (start, goal)
    graph.grid.set_cell(blocked, blocked=True, terrain="mountain", terrain_multiplier=20, cell_size_m=graph.spec.cell_size_m)
    reconstruction = HistoricalRouteReconstructor().reconstruct(waypoints, graph)
    traversed = {
        graph.spec.geographic_to_grid(longitude, latitude)
        for longitude, latitude in reconstruction.candidate_paths[0].geometry.coordinates[1:-1]
    }
    assert blocked not in traversed


def test_reconstruction_requires_reviewed_evidence_and_is_deterministic():
    waypoints = hannibal_alps_waypoints()
    graph = OfflineMockTerrainGraphProvider().build_graph(waypoints, padding_km=5, cell_size_m=5_000)
    first = HistoricalRouteReconstructor().reconstruct(waypoints, graph).model_dump()
    second = HistoricalRouteReconstructor().reconstruct(waypoints, graph).model_dump()
    assert first == second
    missing_evidence = ReviewedHistoricalWaypoint.model_construct(
        id="unreviewed", canonical_name="Unreviewed", longitude=5.1, latitude=45.3, evidence_refs=[], confidence=0.5,
    )
    with pytest.raises(HistoricalRouteReconstructionError, match="no evidence"):
        HistoricalRouteReconstructor().reconstruct([waypoints[0], missing_evidence], graph)
