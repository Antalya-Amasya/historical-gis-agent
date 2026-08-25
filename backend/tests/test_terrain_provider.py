import struct

import pytest

from backend.app.candidate_routes.costs import SyntheticCostModel
from backend.app.candidate_routes.geographic import GeographicGridSpec
from backend.app.candidate_routes.grid import GridCell, GridPoint
from backend.app.candidate_routes.models import ArmyProfile
from backend.app.candidate_routes.terrain import (
    DEMTerrainProvider,
    HgtRaster,
    SyntheticTerrainProvider,
    TerrainDataUnavailableError,
    UnsupportedDemError,
)


def write_fixture_hgt(tmp_path, rows):
    path = tmp_path / "N45E004.hgt"
    path.write_bytes(struct.pack(f">{len(rows) * len(rows)}h", *(value for row in rows for value in row)))
    return path


def local_spec():
    return GeographicGridSpec.from_anchor_coordinates(
        (4.10, 45.10), (4.20, 45.20), padding_km=0, cell_size_m=4_000,
    )


def test_synthetic_provider_remains_offline_and_builds_plain_grid():
    provider = SyntheticTerrainProvider()
    grid = provider.build_grid(local_spec(), resolution_m=4_000)
    assert provider.source == "synthetic_geographic"
    assert provider.get_elevation(4.1, 45.1) == 0.0
    assert all(cell.terrain == "plain" and cell.cell_size_m == 4_000 for cell in grid._cells.values())


def test_hgt_dem_provider_reads_real_offline_raster_and_builds_grid(tmp_path):
    provider = DEMTerrainProvider.from_hgt(write_fixture_hgt(tmp_path, [
        [0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
        [30, 31, 32, 33, 34],
        [40, 41, 42, 43, 44],
    ]))
    assert provider.get_elevation(4.5, 45.5) == 22.0
    grid = provider.build_grid(local_spec(), resolution_m=4_000)
    elevations = [cell.elevation_m for cell in grid._cells.values()]
    assert provider.source == "offline_srtm_hgt"
    assert len(elevations) == grid.width * grid.height
    assert all(cell.terrain == "dem" and not cell.blocked for cell in grid._cells.values())


def test_hgt_boundary_no_data_and_invalid_resolution_are_explicit(tmp_path):
    raster = HgtRaster.from_file(write_fixture_hgt(tmp_path, [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, -32768, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]))
    provider = DEMTerrainProvider(raster)
    with pytest.raises(TerrainDataUnavailableError):
        provider.get_elevation(5.1, 45.5)
    with pytest.raises(TerrainDataUnavailableError):
        provider.get_elevation(4.5, 45.5)
    with pytest.raises(ValueError):
        provider.build_grid(local_spec(), resolution_m=0)
    with pytest.raises(ValueError):
        provider.build_grid(local_spec(), resolution_m=1_000)


def test_invalid_hgt_fixture_is_rejected(tmp_path):
    invalid = tmp_path / "not-a-tile.hgt"
    invalid.write_bytes(b"bad")
    with pytest.raises(UnsupportedDemError):
        HgtRaster.from_file(invalid)


def test_cost_uses_local_slope_factor_not_absolute_elevation():
    current = GridCell(GridPoint(0, 0), elevation_m=0, cell_size_m=1_000)
    flat = GridCell(GridPoint(1, 0), elevation_m=0, cell_size_m=1_000)
    steep = GridCell(GridPoint(1, 0), elevation_m=200, cell_size_m=1_000)
    model = SyntheticCostModel()
    profile = ArmyProfile(distance_weight=1, slope_weight=0, terrain_weight=2, barrier_weight=1)
    flat_cost = model.edge_cost(current, flat, profile)
    steep_cost = model.edge_cost(current, steep, profile)
    assert flat_cost.terrain_cost == 0
    assert steep_cost.terrain_cost == 8
    assert flat_cost.total_cost < steep_cost.total_cost
