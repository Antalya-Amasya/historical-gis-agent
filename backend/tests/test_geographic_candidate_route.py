from math import isclose

import pytest

from backend.app.candidate_routes import ArmyProfile, CandidateRouteEngine, NoPathError
from backend.app.candidate_routes.geographic import (
    DEFAULT_MAX_GRID_CELLS,
    GeographicCandidateRouteService,
    GeographicGridSpec,
    GridTooLargeError,
    LocalProjection,
    PointOutsideGridError,
    SyntheticGeographicTerrainProvider,
    TerrainOverride,
)
from backend.app.candidate_routes.grid import GridPoint
from backend.app.models import HistoricalPlace, HistoricalRoutePoint


def historical_point(identifier: str, name: str, lon: float, lat: float, evidence_ref: str) -> HistoricalRoutePoint:
    place = HistoricalPlace(id=identifier, canonical_name=name, longitude=lon, latitude=lat, source="auditable fixture", confidence=0.9)
    return HistoricalRoutePoint(sequence=1, historical_place=place, event_summary="Evidence-grounded anchor", evidence_refs=[evidence_ref], confidence=0.9)


def profile(**kwargs):
    return ArmyProfile(**kwargs)


def spec():
    return GeographicGridSpec.from_anchor_coordinates((4.0, 45.0), (4.1, 45.0), padding_km=2, cell_size_m=1_000)


def test_local_projection_round_trip_and_local_distance():
    projection = LocalProjection(4.05, 45.0)
    lon, lat = 4.07321, 45.01234
    x, y = projection.to_local(lon, lat)
    restored_lon, restored_lat = projection.to_geographic(x, y)
    assert abs(restored_lon - lon) < 1e-10 and abs(restored_lat - lat) < 1e-10
    assert projection.distance_m((4.0, 45.0), (4.1, 45.0)) > 7_000


def test_bounds_mapping_uses_cell_centers_and_rejects_outside_points():
    grid_spec = spec()
    first = grid_spec.geographic_to_grid(4.0, 45.0)
    second = grid_spec.geographic_to_grid(4.1, 45.0)
    assert first != second
    assert grid_spec.width * grid_spec.height <= DEFAULT_MAX_GRID_CELLS
    assert grid_spec.cell_size_m == 1_000
    lon, lat = grid_spec.grid_to_geographic(first)
    assert grid_spec.projection.distance_m((4.0, 45.0), (lon, lat)) <= 2 ** 0.5 * grid_spec.cell_size_m / 2
    with pytest.raises(PointOutsideGridError):
        grid_spec.geographic_to_grid(5.0, 45.0)
    with pytest.raises(PointOutsideGridError):
        grid_spec.grid_to_geographic(GridPoint(grid_spec.width, 0))


def test_grid_size_guard_runs_before_any_large_grid_allocation():
    with pytest.raises(GridTooLargeError):
        GeographicGridSpec.from_anchor_coordinates((0.0, 0.0), (10.0, 10.0), padding_km=0, cell_size_m=1, max_grid_cells=100)


def test_synthetic_geographic_provider_default_and_rule_are_offline():
    grid_spec = spec()
    default = SyntheticGeographicTerrainProvider().build(grid_spec)
    assert default.terrain_cell(GridPoint(0, 0)).terrain == "plain"
    provider = SyntheticGeographicTerrainProvider(lambda point, lon, lat: TerrainOverride(elevation_m=50, terrain="hill", terrain_multiplier=3) if point == GridPoint(1, 1) else None)
    changed = provider.build(grid_spec).terrain_cell(GridPoint(1, 1))
    assert changed.elevation_m == 50 and changed.terrain == "hill" and changed.terrain_multiplier == 3
    assert changed.provenance == "derived_geographic_data"


def test_real_coordinate_flat_path_returns_wgs84_lon_lat_geojson_and_metadata():
    first = historical_point("a", "Anchor A", 4.0, 45.0, "e-a")
    second = historical_point("b", "Anchor B", 4.1, 45.0, "e-b")
    route = GeographicCandidateRouteService().build_between(first, second, profile(), padding_km=2, cell_size_m=2_000)
    start_lon, start_lat = route.geometry.coordinates[0]
    end_lon, end_lat = route.geometry.coordinates[-1]
    assert route.geometry.type == "LineString"
    assert abs(start_lon - 4.0) < 0.03 and abs(start_lat - 45.0) < 0.03
    assert abs(end_lon - 4.1) < 0.03 and abs(end_lat - 45.0) < 0.03
    assert route.coordinate_system == "EPSG:4326" and route.projection_method == "local_equirectangular"
    assert route.grid_cell_size_m == 2_000 and route.terrain_source == "synthetic_geographic"
    assert route.geometry.coordinates[0][0] < 10 and route.geometry.coordinates[0][1] > 40


def test_geographic_mountain_band_diverts_wgs84_path_off_direct_latitude():
    def mountain_rule(point, lon, lat):
        if 4.03 < lon < 4.09 and 44.999 < lat < 45.01:
            return TerrainOverride(elevation_m=100, terrain="mountain", terrain_multiplier=20)
        return None

    first = historical_point("a", "Anchor A", 4.0, 45.0, "e-a")
    second = historical_point("b", "Anchor B", 4.12, 45.0, "e-b")
    route = GeographicCandidateRouteService(SyntheticGeographicTerrainProvider(mountain_rule)).build_between(
        first, second, profile(distance_weight=1, slope_weight=0.1, terrain_weight=10, barrier_weight=1), padding_km=2, cell_size_m=1_000,
    )
    direct_lat = route.geometry.coordinates[0][1]
    assert any(abs(lat - direct_lat) > 0.005 for _, lat in route.geometry.coordinates[1:-1])
    assert route.provenance == "algorithmic_candidate"


def test_geographic_barrier_routes_through_gap_or_raises_when_sealed():
    grid_spec = spec()
    grid = SyntheticGeographicTerrainProvider().build(grid_spec)
    start, goal = grid_spec.geographic_to_grid(4.0, 45.0), grid_spec.geographic_to_grid(4.1, 45.0)
    wall_x = (start.x + goal.x) // 2
    for y in range(grid_spec.height - 1):
        grid.set_cell(GridPoint(wall_x, y), blocked=True)
    path = CandidateRouteEngine().find_path(grid, start, goal, profile())
    assert GridPoint(wall_x, grid_spec.height - 1) in path.points
    assert all(not grid.cell(point).blocked for point in path.points)

    sealed = SyntheticGeographicTerrainProvider().build(grid_spec)
    for y in range(grid_spec.height):
        sealed.set_cell(GridPoint(wall_x, y), blocked=True)
    with pytest.raises(NoPathError):
        CandidateRouteEngine().find_path(sealed, start, goal, profile())


def test_geographic_candidate_keeps_evidence_anchor_and_algorithmic_provenance_separate():
    first = historical_point("a", "Anchor A", 4.0, 45.0, "e-a")
    second = historical_point("b", "Anchor B", 4.1, 45.0, "e-b")
    route = GeographicCandidateRouteService().build_between(first, second, profile(), padding_km=2, cell_size_m=2_000)
    assert route.from_anchor.provenance == "evidence_grounded_anchor"
    assert route.to_anchor.provenance == "evidence_grounded_anchor"
    assert route.provenance == "algorithmic_candidate"
    assert route.evidence_refs == ["e-a", "e-b"]
    assert "not historical facts" in route.assumptions[-1]
