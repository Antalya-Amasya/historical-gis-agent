from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.geographic import GeographicGridSpec, TerrainGrid
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.models import ArmyProfile
from backend.app.gis.surface import (
    MockSurfaceClassifier,
    NoSurfaceClassifier,
    SurfaceClassification,
    SurfaceType,
)


def _grid() -> TerrainGrid:
    return TerrainGrid(GeographicGridSpec.from_anchor_coordinates(
        (4.10, 45.10), (4.20, 45.20), padding_km=0, cell_size_m=10_000,
    ))


def _classification(surface_type: SurfaceType) -> SurfaceClassification:
    return SurfaceClassification(surface_type, "test_surface_data", confidence=1.0)


def test_valid_zero_elevation_can_be_explicitly_land_or_water():
    grid = SyntheticGrid.flat(2, 1)
    grid.set_cell(GridPoint(0, 0), elevation_m=0.0, surface=_classification(SurfaceType.LAND))
    grid.set_cell(GridPoint(1, 0), elevation_m=0.0, surface=_classification(SurfaceType.WATER))

    assert grid.cell(GridPoint(0, 0)).surface.surface_type is SurfaceType.LAND
    assert grid.cell(GridPoint(1, 0)).surface.surface_type is SurfaceType.WATER


def test_missing_or_nodata_elevation_does_not_determine_surface_type():
    grid = SyntheticGrid.flat(2, 1)
    grid.set_cell(GridPoint(0, 0), terrain="missing_dem", blocked=True, surface=_classification(SurfaceType.UNKNOWN))
    grid.set_cell(GridPoint(1, 0), terrain="nodata_dem", blocked=True, surface=_classification(SurfaceType.UNKNOWN))

    assert grid.cell(GridPoint(0, 0)).surface.surface_type is SurfaceType.UNKNOWN
    assert grid.cell(GridPoint(1, 0)).surface.surface_type is SurfaceType.UNKNOWN


def test_mock_classifier_does_not_use_elevation_to_classify_water():
    classifier = MockSurfaceClassifier()
    classifier.register(45.0, 4.0, SurfaceType.WATER)

    assert classifier.classify(45.0, 4.0).surface_type is SurfaceType.WATER
    assert classifier.classify(45.0, 4.1).surface_type is SurfaceType.UNKNOWN


def test_positive_elevation_can_still_be_classified_as_water():
    grid = SyntheticGrid.flat(1, 1)
    grid.set_cell(GridPoint(0, 0), elevation_m=250.0, surface=_classification(SurfaceType.WATER))

    assert grid.cell(GridPoint(0, 0)).surface.surface_type is SurfaceType.WATER


def test_unknown_and_blocked_are_not_water():
    assert SurfaceType.UNKNOWN is not SurfaceType.WATER
    assert SurfaceType.BLOCKED is not SurfaceType.WATER


def test_mock_classifier_can_explicitly_register_blocked_surface():
    classifier = MockSurfaceClassifier()
    classifier.register(45.0, 4.0, SurfaceType.BLOCKED)

    assert classifier.classify(45.0, 4.0).surface_type is SurfaceType.BLOCKED


def test_no_surface_classifier_reports_unavailable_unknown_not_sea():
    result = NoSurfaceClassifier().classify(45.0, 4.0)

    assert result.surface_type is SurfaceType.UNKNOWN
    assert result.status == "unavailable"


def test_terrain_grid_can_explicitly_receive_a_classifier_with_provenance():
    grid = _grid()
    point = GridPoint(0, 0)
    longitude, latitude = grid.spec.grid_to_geographic(point)
    classifier = MockSurfaceClassifier()
    classifier.register(latitude, longitude, SurfaceType.LAND, source="fixture_mask", confidence=0.8)

    grid.apply_surface_classifier(classifier)
    result = grid.terrain_cell(point).surface

    assert result.surface_type is SurfaceType.LAND
    assert result.source == "fixture_mask"
    assert result.confidence == 0.8


def test_existing_land_only_astar_ignores_surface_labels():
    grid = SyntheticGrid.flat(2, 1)
    grid.set_cell(GridPoint(0, 0), surface=_classification(SurfaceType.WATER))
    grid.set_cell(GridPoint(1, 0), surface=_classification(SurfaceType.UNKNOWN))

    path = CandidateRouteEngine().find_path(grid, GridPoint(0, 0), GridPoint(1, 0), ArmyProfile())

    assert path.points == (GridPoint(0, 0), GridPoint(1, 0))
