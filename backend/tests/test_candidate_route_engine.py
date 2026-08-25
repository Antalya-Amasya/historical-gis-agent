import json
from math import isinf

import pytest

from backend.app.candidate_routes import ArmyProfile, CandidateRouteAnchor, CandidateRouteEngine, NoPathError
from backend.app.candidate_routes.costs import SyntheticCostModel
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.models import HistoricalPlace, HistoricalRoutePoint


def anchors():
    return (
        CandidateRouteAnchor(historical_place_id="anchor-a", canonical_name="Anchor A", evidence_refs=["e-a"]),
        CandidateRouteAnchor(historical_place_id="anchor-b", canonical_name="Anchor B", evidence_refs=["e-b", "e-a"]),
    )


def profile(**values):
    return ArmyProfile(**values)


def test_cost_model_keeps_components_separate_and_blocks_cells():
    grid = SyntheticGrid.flat(2, 1)
    start, goal = GridPoint(0, 0), GridPoint(1, 0)
    grid.set_cell(start, elevation_m=10)
    grid.set_cell(goal, elevation_m=25, terrain="rough", terrain_multiplier=3)
    model = SyntheticCostModel()
    result = model.edge_cost(grid.cell(start), grid.cell(goal), profile(distance_weight=2, slope_weight=0.5, terrain_weight=4, barrier_weight=1))
    assert result.distance_cost == 2
    assert result.slope_cost == 7.5
    assert result.terrain_cost == 8
    assert result.barrier_cost == 0 and result.total_cost == 17.5
    grid.set_cell(goal, blocked=True)
    blocked = model.edge_cost(grid.cell(start), grid.cell(goal), profile())
    assert isinf(blocked.barrier_cost) and isinf(blocked.total_cost)


def test_flat_world_uses_shortest_deterministic_path_and_serializes_candidate():
    grid = SyntheticGrid.flat(5, 3)
    first, second = anchors()
    route = CandidateRouteEngine().build_route(
        from_anchor=first, to_anchor=second, start=GridPoint(0, 1), goal=GridPoint(4, 1), grid=grid, profile=profile(),
    )
    assert route.metrics.distance_km == 4 and route.metrics.segment_count == 4
    assert route.geometry.coordinates == [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0)]
    assert route.provenance == "algorithmic_candidate" and route.coordinate_system == "synthetic_grid"
    assert json.loads(json.dumps(route.model_dump(mode="json")))["metrics"]["cell_count"] == 5


def test_mountain_penalty_makes_astar_avoid_direct_cells():
    grid = SyntheticGrid.flat(7, 5)
    mountains = {GridPoint(x, 2) for x in (2, 3, 4)}
    for point in mountains:
        grid.set_cell(point, elevation_m=100, terrain="mountain", terrain_multiplier=20)
    path = CandidateRouteEngine().find_path(
        grid, GridPoint(0, 2), GridPoint(6, 2), profile(distance_weight=1, slope_weight=0.05, terrain_weight=10, barrier_weight=1),
    )
    assert not set(path.points) & mountains
    assert len(path.points) - 1 > 6


def test_blocked_barrier_is_never_crossed_and_no_path_is_explicit():
    grid = SyntheticGrid.flat(7, 5)
    barrier = {GridPoint(3, y) for y in range(4)}
    for point in barrier:
        grid.set_cell(point, blocked=True)
    path = CandidateRouteEngine().find_path(grid, GridPoint(0, 2), GridPoint(6, 2), profile())
    assert not set(path.points) & barrier
    assert GridPoint(3, 4) in path.points

    sealed = SyntheticGrid.flat(3, 3)
    for y in range(3):
        sealed.set_cell(GridPoint(1, y), blocked=True)
    with pytest.raises(NoPathError):
        CandidateRouteEngine().find_path(sealed, GridPoint(0, 1), GridPoint(2, 1), profile())


def test_weight_sensitivity_changes_path_in_same_geography():
    grid = SyntheticGrid.flat(7, 5)
    hills = {GridPoint(x, 2) for x in (2, 3, 4)}
    for point in hills:
        grid.set_cell(point, elevation_m=10, terrain="hill", terrain_multiplier=1)
    engine = CandidateRouteEngine()
    distance_first = engine.find_path(grid, GridPoint(0, 2), GridPoint(6, 2), profile(distance_weight=1, slope_weight=0, terrain_weight=0, barrier_weight=1))
    slope_first = engine.find_path(grid, GridPoint(0, 2), GridPoint(6, 2), profile(distance_weight=0.2, slope_weight=5, terrain_weight=0, barrier_weight=1))
    assert set(distance_first.points) & hills
    assert not set(slope_first.points) & hills
    assert distance_first.cost_breakdown.total_cost != slope_first.cost_breakdown.total_cost


def test_historical_anchor_provenance_is_preserved_but_candidate_is_not_historical_fact():
    place = HistoricalPlace(id="pleiades-1", canonical_name="Evidence Anchor", latitude=1, longitude=2, source="Pleiades", source_id="1", source_url="https://pleiades.stoa.org/places/1", confidence=0.9)
    point = HistoricalRoutePoint(sequence=1, historical_place=place, event_summary="Evidence mention", evidence_refs=["evidence-1"], confidence=0.9)
    anchor = CandidateRouteAnchor.from_historical_point(point)
    other = CandidateRouteAnchor(historical_place_id="pleiades-2", canonical_name="Second Anchor", evidence_refs=["evidence-2"])
    route = CandidateRouteEngine().build_route(from_anchor=anchor, to_anchor=other, start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=profile())
    assert anchor.provenance == "evidence_grounded_anchor"
    assert route.provenance == "algorithmic_candidate"
    assert route.evidence_refs == ["evidence-1", "evidence-2"]
    assert "not an evidence-grounded historical track" in route.assumptions[1]
