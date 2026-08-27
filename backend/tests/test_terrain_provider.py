import struct

import pytest

from backend.app.candidate_routes.costs import SyntheticCostModel
from backend.app.candidate_routes.geographic import GeographicGridSpec
from backend.app.candidate_routes.grid import GridCell, GridPoint
from backend.app.candidate_routes.models import ArmyProfile
from backend.app.candidate_routes.terrain import (
    DEMTerrainProvider,
    HgtRaster,
    MosaicDEMProvider,
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


def test_mosaic_dem_provider_queries_adjacent_tiles_and_caches_them(tmp_path):
    """Shared edge values make the synthetic two-tile surface continuous at E005."""
    west_tile = tmp_path / "N45E004.hgt"
    east_tile = tmp_path / "N45E005.hgt"
    west_tile.write_bytes(struct.pack(">9h", *([100, 150, 200] * 3)))
    east_tile.write_bytes(struct.pack(">9h", *([200, 250, 300] * 3)))
    provider = MosaicDEMProvider(tmp_path)

    assert provider.get_elevation(4.99, 45.5) == 200.0
    assert provider.get_elevation(5.0, 45.5) == 200.0
    assert provider.get_elevation(5.01, 45.5) == 200.0
    assert set(provider._rasters) == {"N45E004.hgt", "N45E005.hgt"}


def test_mosaic_dem_provider_uses_srtm_south_west_tile_names_for_negative_coordinates(tmp_path):
    tile = tmp_path / "S01W001.hgt"
    tile.write_bytes(struct.pack(">4h", 7, 7, 7, 7))
    provider = MosaicDEMProvider(tmp_path)

    assert provider.tile_name_for(-0.1, -0.1) == "S01W001.hgt"
    assert provider.get_elevation(-0.1, -0.1) == 7.0


def test_cost_uses_physical_slope_and_changes_with_cell_resolution():
    current_1km = GridCell(GridPoint(0, 0), elevation_m=0, cell_size_m=1_000)
    current_5km = GridCell(GridPoint(0, 0), elevation_m=0, cell_size_m=5_000)
    uphill_1km = GridCell(GridPoint(1, 0), elevation_m=100, cell_size_m=1_000)
    uphill_5km = GridCell(GridPoint(1, 0), elevation_m=100, cell_size_m=5_000)
    model = SyntheticCostModel()
    profile = ArmyProfile(distance_weight=1, slope_weight=1, terrain_weight=0, barrier_weight=1)

    one_km_cost = model.edge_cost(current_1km, uphill_1km, profile)
    five_km_cost = model.edge_cost(current_5km, uphill_5km, profile)

    assert one_km_cost.slope_cost == 1_500
    assert five_km_cost.slope_cost == 0
    assert one_km_cost.total_cost > five_km_cost.total_cost


def test_slope_cost_falls_back_to_one_kilometre_when_cell_size_is_unspecified():
    current = GridCell(GridPoint(0, 0), elevation_m=0)
    uphill = GridCell(GridPoint(1, 0), elevation_m=100)
    cost = SyntheticCostModel().edge_cost(current, uphill, ArmyProfile(slope_weight=1))

    assert cost.slope_cost == 1_500
