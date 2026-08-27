import json
import struct

import pytest

from backend.app.gis.dem import ElevationStatus, HgtTileStore
from backend.app.gis.dem_manifest import UNKNOWN, build_hgt_inventory, load_dem_manifest
from backend.app.gis.srtm import SrtmElevationService
from backend.app.candidate_routes.terrain import MosaicDEMProvider


def write_hgt(directory, name, values):
    (directory / name).write_bytes(struct.pack(f">{len(values)}h", *values))


def test_canonical_store_distinguishes_valid_zero_missing_nodata_and_invalid(tmp_path):
    write_hgt(tmp_path, "N00E000.hgt", [0, 0, 0, -32768])
    valid = HgtTileStore(tmp_path, samples_per_edge=2).sample(0.99, 0.0)
    nodata = HgtTileStore(tmp_path, samples_per_edge=2).sample(0.01, 0.99)
    missing = HgtTileStore(tmp_path, samples_per_edge=2).sample(1.1, 0.1)
    (tmp_path / "N02E000.hgt").write_bytes(b"corrupt")
    invalid = HgtTileStore(tmp_path, samples_per_edge=2).sample(2.1, 0.1)

    assert valid.status is ElevationStatus.VALID and valid.elevation_m == 0.0
    assert nodata.status is ElevationStatus.NODATA and nodata.elevation_m is None
    assert missing.status is ElevationStatus.MISSING and missing.elevation_m is None
    assert invalid.status is ElevationStatus.INVALID and invalid.elevation_m is None


def test_srtm_and_mosaic_adapters_share_canonical_sample_values(tmp_path):
    values = [10] * (1201 * 1201)
    values[0] = 900
    values[1200 * 1201] = 100
    write_hgt(tmp_path, "N45E004.hgt", values)
    srtm = SrtmElevationService(tmp_path)
    mosaic = MosaicDEMProvider(tmp_path)

    assert srtm.sample(45.9999, 4.0).elevation_m == mosaic.get_elevation(4.0, 45.9999) == 900.0
    assert srtm.sample(45.0001, 4.0).elevation_m == mosaic.get_elevation(4.0, 45.0001) == 100.0


def test_bounded_cache_evicts_least_recent_tile(tmp_path):
    for name, value in (("N00E000.hgt", 1), ("N00E001.hgt", 2), ("N00E002.hgt", 3)):
        write_hgt(tmp_path, name, [value] * 4)
    store = HgtTileStore(tmp_path, samples_per_edge=2, cache_size=2)
    for longitude in (0.1, 1.1, 2.1):
        assert store.sample(0.1, longitude).status is ElevationStatus.VALID
    assert store.cached_tile_ids == ("N00E001.hgt", "N00E002.hgt")


def test_adapter_rejects_a_non_positive_cache_size(tmp_path):
    with pytest.raises(ValueError, match="cache_size"):
        MosaicDEMProvider(tmp_path, cache_size=0)


def test_inventory_and_missing_manifest_preserve_unknown_provenance(tmp_path):
    write_hgt(tmp_path, "N45E004.hgt", [1] * 4)
    inventory = build_hgt_inventory(tmp_path)
    assert inventory.tile_count == 1
    assert inventory.provenance.source_name == UNKNOWN
    assert inventory.provenance.acquisition_method == UNKNOWN
    assert load_dem_manifest(None)["provenance"]["source_name"] == UNKNOWN

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"provenance": {"source_name": UNKNOWN}}), encoding="utf-8")
    assert load_dem_manifest(manifest)["provenance"]["source_name"] == UNKNOWN
    assert MosaicDEMProvider(tmp_path, samples_per_edge=2).provenance_metadata["provenance"]["source_name"] == UNKNOWN
