import hashlib
import json
from pathlib import Path
from shutil import copytree

from backend.app.gis.natural_earth_surface import NaturalEarthAvailability, NaturalEarthSurfaceClassifier
from backend.app.gis.surface import SurfaceType, WaterDomain


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "natural_earth_surface"
LAYER_FILES = {
    "land": "ne_10m_land.geojson", "ocean": "ne_10m_ocean.geojson",
    "lakes": "ne_10m_lakes.geojson", "minor_islands": "ne_10m_minor_islands.geojson",
}


def _manifest(root: Path) -> Path:
    payload = {
        "schema_version": 1,
        "dataset_family": "Natural Earth",
        "dataset_version": "5.1.1",
        "layers": {
            name: {"path": filename, "sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest()}
            for name, filename in LAYER_FILES.items()
        },
    }
    path = root / "surface-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _provider(tmp_path: Path):
    root = tmp_path / "surface"
    copytree(FIXTURE_DIR, root)
    return root, NaturalEarthSurfaceClassifier.from_manifest(root, _manifest(root))


def test_manifest_loader_audits_all_required_layers_and_domains(tmp_path):
    _, provider = _provider(tmp_path)
    assert provider.audit.availability is NaturalEarthAvailability.AVAILABLE
    assert provider.classify(1.0, 1.0).surface_type is SurfaceType.LAND
    assert provider.classify(-1.0, -1.0).water_domain is WaterDomain.OCEAN
    assert provider.classify(4.0, 4.0).water_domain is WaterDomain.INLAND_WATER
    assert "Modern surface approximation" in provider.classify(-1.0, -1.0).metadata["modern_geography_warning"]


def test_manifest_hash_mismatch_and_absence_fail_closed(tmp_path):
    root, provider = _provider(tmp_path)
    assert provider.audit.availability is NaturalEarthAvailability.AVAILABLE
    (root / LAYER_FILES["minor_islands"]).write_bytes(b"changed")
    mismatched = NaturalEarthSurfaceClassifier.from_manifest(root, root / "surface-manifest.json")
    missing = NaturalEarthSurfaceClassifier.from_manifest(root, root / "absent.json")
    assert mismatched.audit.availability is NaturalEarthAvailability.HASH_MISMATCH
    assert mismatched.classify(1.0, 1.0).surface_type is SurfaceType.UNKNOWN
    assert missing.audit.availability is NaturalEarthAvailability.MISSING


def test_island_and_segment_land_intersection_are_not_sample_only(tmp_path):
    _, provider = _provider(tmp_path)
    assert provider.classify(1.0, 13.0).surface_type is SurfaceType.LAND
    assert provider.validate_land_segment(((-2.0, -2.0), (2.0, 2.0))).intersects_land is True
    assert provider.validate_land_segment(((11.0, 1.0), (15.0, 1.0))).intersects_land is True
    assert provider.validate_land_segment(((8.0, -2.0), (8.0, -1.0))).intersects_land is False


def test_boundary_and_unconfigured_production_path_fail_closed(tmp_path):
    root, provider = _provider(tmp_path)
    assert provider.classify(5.0, 0.0).surface_type is SurfaceType.UNKNOWN
    assert provider.validate_land_segment(((0.0, 5.0), (2.0, 5.0))).intersects_land is None
    unconfigured = NaturalEarthSurfaceClassifier.production(tmp_path / "no-config")
    assert unconfigured.audit.availability is NaturalEarthAvailability.UNAVAILABLE
    assert unconfigured.classify(0.0, 0.0).surface_type is SurfaceType.UNKNOWN
