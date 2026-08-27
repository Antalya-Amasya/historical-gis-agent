import json
from pathlib import Path
from shutil import copytree

from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.geographic import GeographicGridSpec, TerrainGrid
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.models import ArmyProfile
from backend.app.gis.natural_earth_surface import (
    MODERN_GEOGRAPHY_WARNING,
    NaturalEarthSurfaceClassifier,
)
from backend.app.gis.natural_earth_preprocess import normalize_natural_earth_geojson
from backend.app.gis.surface import SurfaceType


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "natural_earth_surface"


def classifier() -> NaturalEarthSurfaceClassifier:
    return NaturalEarthSurfaceClassifier(FIXTURE_DIR)


def test_fixture_classifies_land_ocean_lake_and_minor_island():
    subject = classifier()

    assert subject.classify(1.0, 1.0).surface_type is SurfaceType.LAND
    assert subject.classify(-1.0, -1.0).surface_type is SurfaceType.WATER
    assert subject.classify(4.0, 4.0).surface_type is SurfaceType.WATER
    assert subject.classify(1.0, 13.0).surface_type is SurfaceType.LAND


def test_boundary_is_unknown_not_land_or_water():
    result = classifier().classify(5.0, 0.0)

    assert result.surface_type is SurfaceType.UNKNOWN
    assert result.status == "boundary_ambiguous"
    assert result.confidence == 0.0


def test_overlapping_ocean_and_land_is_fail_closed(tmp_path):
    dataset = tmp_path / "dataset"
    copytree(FIXTURE_DIR, dataset)
    ocean_path = dataset / "ne_10m_ocean.geojson"
    payload = json.loads(ocean_path.read_text(encoding="utf-8"))
    payload["features"].append({
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[6, 6], [8, 6], [8, 8], [6, 8], [6, 6]]]},
    })
    ocean_path.write_text(json.dumps(payload), encoding="utf-8")

    result = NaturalEarthSurfaceClassifier(dataset).classify(7.0, 7.0)

    assert result.surface_type is SurfaceType.UNKNOWN
    assert result.status == "conflicting_geometry"


def test_invalid_coordinate_and_missing_or_corrupt_dataset_are_fail_closed(tmp_path):
    invalid = classifier().classify(91.0, 0.0)
    missing = NaturalEarthSurfaceClassifier(tmp_path).classify(0.0, 0.0)
    corrupt = tmp_path / "corrupt"
    copytree(FIXTURE_DIR, corrupt)
    (corrupt / "ne_10m_land.geojson").write_text("not json", encoding="utf-8")

    assert invalid.surface_type is SurfaceType.UNKNOWN
    assert invalid.status == "invalid_coordinate"
    assert missing.surface_type is SurfaceType.UNKNOWN
    assert missing.status == "dataset_unavailable"
    assert NaturalEarthSurfaceClassifier(corrupt).classify(1.0, 1.0).status == "dataset_unavailable"


def test_hash_mismatch_is_fail_closed():
    result = NaturalEarthSurfaceClassifier(FIXTURE_DIR, expected_hashes={"land": "0" * 64}).classify(1.0, 1.0)

    assert result.surface_type is SurfaceType.UNKNOWN
    assert result.status == "dataset_unavailable"


def test_explicit_preprocessing_repairs_invalid_source_geometry(tmp_path):
    raw = tmp_path / "raw"
    normalized = tmp_path / "normalized"
    copytree(FIXTURE_DIR, raw)
    ocean_path = raw / "ne_10m_ocean.geojson"
    payload = json.loads(ocean_path.read_text(encoding="utf-8"))
    payload["features"][0]["geometry"] = {"type": "Polygon", "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]}
    ocean_path.write_text(json.dumps(payload), encoding="utf-8")

    audit = normalize_natural_earth_geojson(raw, normalized)
    subject = NaturalEarthSurfaceClassifier(normalized, expected_hashes=audit.normalized_hashes)
    subject.set_raw_source_hashes(audit.raw_hashes)

    assert audit.repaired_geometry_counts["ocean"] == 1
    assert subject.audit.availability_error is None
    assert subject.classify(9.0, 9.0).metadata["raw_source_sha256"] == audit.raw_hashes["land"]


def test_provenance_is_complete_and_classification_is_deterministic():
    subject = classifier()
    first = subject.classify(1.0, 1.0)
    second = subject.classify(1.0, 1.0)

    assert first == second
    assert first.metadata["dataset_name"] == "Natural Earth 1:10m Physical Vectors"
    assert first.metadata["source_file"] == "ne_10m_land.geojson"
    assert first.metadata["source_sha256"]
    assert first.metadata["classification_method"]
    assert first.metadata["resolution_or_scale"] == "1:10m"
    assert first.metadata["modern_geography_warning"] == MODERN_GEOGRAPHY_WARNING
    assert first.metadata["boundary_status"] == "interior"


def test_terrain_grid_decoration_preserves_elevation_and_blocking():
    grid = TerrainGrid(GeographicGridSpec.from_anchor_coordinates(
        (1.0, 1.0), (1.02, 1.02), padding_km=0, cell_size_m=10_000,
    ))
    point = GridPoint(0, 0)
    grid.set_cell(point, elevation_m=0.0, blocked=False)

    grid.apply_surface_classifier(classifier())
    decorated = grid.cell(point)

    assert decorated.surface.surface_type is SurfaceType.LAND
    assert decorated.elevation_m == 0.0
    assert decorated.blocked is False


def test_classifier_does_not_automatically_change_land_only_astar():
    grid = SyntheticGrid.flat(2, 1)
    path = CandidateRouteEngine().find_path(grid, GridPoint(0, 0), GridPoint(1, 0), ArmyProfile())

    assert path.points == (GridPoint(0, 0), GridPoint(1, 0))
