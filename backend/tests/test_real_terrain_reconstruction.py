import struct

import pytest

from backend.app.candidate_routes.historical_reconstruction import (
    HistoricalRouteReconstructor,
    OfflineMockTerrainGraphProvider,
    RealTerrainGraphProvider,
    ReviewedHistoricalWaypoint,
)
from backend.app.candidate_routes.terrain import (
    DEMTerrainProvider, OfflineMockTerrainProvider, RealTerrainProvider, TerrainDataUnavailableError,
)


def write_alps_fixture(tmp_path):
    """Small local HGT-format fixture; values are test data, not a claimed Alpine DEM."""
    rows = [
        [900, 800, 700, 600, 500],
        [800, 700, 600, 500, 400],
        [700, 600, 500, 400, 300],
        [600, 500, 400, 300, 200],
        [500, 400, 300, 200, 100],
    ]
    path = tmp_path / "N45E005.hgt"
    path.write_bytes(struct.pack(f">{len(rows) * len(rows)}h", *(value for row in rows for value in row)))
    return path


def reviewed_alps_waypoints():
    return [
        ReviewedHistoricalWaypoint(
            id="rhodanus-crossing", canonical_name="Rhodanus crossing region",
            longitude=5.10, latitude=45.10, evidence_refs=["polybius-book-3"], confidence=0.8,
        ),
        ReviewedHistoricalWaypoint(
            id="alps-approach", canonical_name="Alps approach region",
            longitude=5.80, latitude=45.80, evidence_refs=["polybius-book-3"], confidence=0.75,
        ),
    ]


def test_real_offline_hgt_provider_builds_terrain_graph_and_preserves_reviewed_facts(tmp_path):
    waypoints = reviewed_alps_waypoints()
    provider = DEMTerrainProvider.from_hgt(write_alps_fixture(tmp_path))
    graph = RealTerrainGraphProvider(provider).build_graph(
        waypoints, padding_km=0, cell_size_m=25_000,
    )
    reconstruction = HistoricalRouteReconstructor().reconstruct(waypoints, graph, route_id="alps-hgt")

    assert isinstance(provider, RealTerrainProvider)
    assert provider.dataset_id == "local_srtm_hgt"
    assert provider.coordinate_reference == "EPSG:4326"
    assert provider.resolution_m == pytest.approx(111_320.0 / 4)
    assert reconstruction.terrain_source == "offline_srtm_hgt"
    assert reconstruction.geometry.coordinates[0] == (5.10, 45.10)
    assert reconstruction.geometry.coordinates[-1] == (5.80, 45.80)
    assert reconstruction.evidence_refs == ["polybius-book-3"]
    assert reconstruction.candidate_paths[0].from_anchor.evidence_refs == ["polybius-book-3"]
    assert reconstruction.candidate_paths[0].coordinate_system == "EPSG:4326"
    assert reconstruction.candidate_paths[0].cost_breakdown.slope_cost > 0


def test_real_dem_changes_only_terrain_cost_relative_to_offline_mock(tmp_path):
    waypoints = reviewed_alps_waypoints()
    real_graph = RealTerrainGraphProvider(DEMTerrainProvider.from_hgt(write_alps_fixture(tmp_path))).build_graph(
        waypoints, padding_km=0, cell_size_m=25_000,
    )
    mock_graph = OfflineMockTerrainGraphProvider().build_graph(waypoints, padding_km=0, cell_size_m=25_000)
    reconstructor = HistoricalRouteReconstructor()
    real = reconstructor.reconstruct(waypoints, real_graph, route_id="real")
    mock = reconstructor.reconstruct(waypoints, mock_graph, route_id="mock")

    assert real.terrain_source == "offline_srtm_hgt"
    assert mock.terrain_source == "offline_mock_terrain"
    assert real.evidence_refs == mock.evidence_refs
    assert real.candidate_paths[0].from_anchor == mock.candidate_paths[0].from_anchor
    assert real.candidate_paths[0].to_anchor == mock.candidate_paths[0].to_anchor
    assert real.candidate_paths[0].cost_breakdown.slope_cost > mock.candidate_paths[0].cost_breakdown.slope_cost


def test_real_dem_provider_keeps_out_of_tile_data_explicit(tmp_path):
    provider = DEMTerrainProvider.from_hgt(write_alps_fixture(tmp_path))
    with pytest.raises(TerrainDataUnavailableError, match="outside"):
        provider.get_elevation(6.01, 45.5)


def test_named_offline_mock_provider_remains_a_deterministic_test_implementation():
    provider = OfflineMockTerrainProvider()
    assert provider.source == "offline_mock_terrain"
    assert provider.get_elevation(5.5, 45.5) == 0.0
