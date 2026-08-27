import numpy as np
import pytest

from backend.app.candidate_routes.historical_reconstruction import RealTerrainGraphProvider, ReviewedHistoricalWaypoint
from backend.app.gis.srtm import SrtmElevationService, SrtmElevationUnavailableError


def write_tile(directory, name, values):
    (directory / name).write_bytes(np.asarray(values, dtype=">i2").tobytes())


def test_crossing_integer_longitude_loads_adjacent_tiles_without_a_seam(tmp_path):
    size = SrtmElevationService.SAMPLES_PER_EDGE
    west_tile = np.tile(np.linspace(100, 200, size, dtype=np.int16), (size, 1))
    east_tile = np.tile(np.linspace(200, 300, size, dtype=np.int16), (size, 1))
    write_tile(tmp_path, "N45E004.hgt", west_tile)
    write_tile(tmp_path, "N45E005.hgt", east_tile)
    service = SrtmElevationService(tmp_path)

    west_of_seam = service.get_elevation(45.5, 4.999)
    at_seam = service.get_elevation(45.5, 5.0)
    east_of_seam = service.get_elevation(45.5, 5.001)

    assert max(west_of_seam, at_seam, east_of_seam) - min(west_of_seam, at_seam, east_of_seam) <= 1.0
    assert at_seam == east_of_seam == 200.0
    assert service._load_tile.cache_info().currsize == 2


def test_missing_tile_is_explicitly_unavailable_not_sea_level(tmp_path):
    with pytest.raises(SrtmElevationUnavailableError, match="MISSING"):
        SrtmElevationService(tmp_path).get_elevation(45.5, 4.5)


def test_valid_zero_and_void_samples_are_semantically_distinct(tmp_path):
    size = SrtmElevationService.SAMPLES_PER_EDGE
    valid_zero = np.zeros((size, size), dtype=np.int16)
    write_tile(tmp_path, "N45E004.hgt", valid_zero)
    assert SrtmElevationService(tmp_path).get_elevation(45.5, 4.5) == 0.0
    valid_zero[size // 2, size // 2] = SrtmElevationService.VOID_ELEVATION
    write_tile(tmp_path, "N45E004.hgt", valid_zero)
    service = SrtmElevationService(tmp_path)
    with pytest.raises(SrtmElevationUnavailableError, match="NODATA"):
        service.get_elevation(45.5, 4.5)


def test_hgt_row_index_is_flipped_from_north_to_south(tmp_path):
    size = SrtmElevationService.SAMPLES_PER_EDGE
    tile = np.zeros((size, size), dtype=np.int16)
    tile[0, 0] = 900
    tile[-1, 0] = 100
    write_tile(tmp_path, "N45E004.hgt", tile)
    service = SrtmElevationService(tmp_path)

    assert service.get_elevation(45.9999, 4.0) == 900.0
    assert service.get_elevation(45.0001, 4.0) == 100.0


def test_real_terrain_graph_provider_builds_cells_from_srtm_service(tmp_path):
    size = SrtmElevationService.SAMPLES_PER_EDGE
    tile = np.full((size, size), 321, dtype=np.int16)
    write_tile(tmp_path, "N45E004.hgt", tile)
    waypoints = [
        ReviewedHistoricalWaypoint(id="west", canonical_name="West", longitude=4.1, latitude=45.1, evidence_refs=["e"], confidence=1),
        ReviewedHistoricalWaypoint(id="east", canonical_name="East", longitude=4.2, latitude=45.2, evidence_refs=["e"], confidence=1),
    ]

    graph = RealTerrainGraphProvider(SrtmElevationService(tmp_path)).build_graph(
        waypoints, padding_km=0, cell_size_m=5_000,
    )

    assert graph.source == "srtm3_real_terrain"
    assert all(cell.elevation_m == 321 and cell.terrain_multiplier == 1.0 for cell in graph.grid._cells.values())
    assert graph.applied_constraints == ["dem_availability_blocking", "dem_nodata_blocking", "slope_cost"]
