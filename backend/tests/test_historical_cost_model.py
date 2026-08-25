from backend.app.candidate_routes import (
    ArmyProfile,
    CandidateRouteAnchor,
    CandidateRouteEngine,
    HistoricalCostModel,
)
from backend.app.candidate_routes.grid import GridCell, GridPoint, SyntheticGrid


def anchors():
    return (
        CandidateRouteAnchor(historical_place_id="a", canonical_name="Anchor A", evidence_refs=["e-a"]),
        CandidateRouteAnchor(historical_place_id="b", canonical_name="Anchor B", evidence_refs=["e-b"]),
    )


def route_over(grid):
    first, second = anchors()
    return CandidateRouteEngine().build_route(
        from_anchor=first,
        to_anchor=second,
        start=GridPoint(0, 0),
        goal=GridPoint(grid.width - 1, 0),
        grid=grid,
        profile=ArmyProfile(),
    )


def roman():
    return ArmyProfile(
        name="roman_legion", movement_type="infantry", mountain_tolerance=0.5,
        river_crossing_penalty=2.0, rough_terrain_penalty=1.5, supply_range=30,
    )


def carthaginian():
    return ArmyProfile(
        name="carthaginian_army", movement_type="mixed", mountain_tolerance=0.8,
        river_crossing_penalty=1.5, rough_terrain_penalty=1.2, supply_range=40,
    )


def test_same_candidate_route_scores_differ_by_profile():
    grid = SyntheticGrid.flat(2, 1)
    grid.set_cell(GridPoint(1, 0), elevation_m=100, terrain="mountain", terrain_multiplier=5)
    route = route_over(grid)
    roman_score = route.evaluate(roman())
    carthaginian_score = route.evaluate(carthaginian())
    assert roman_score.total_cost > carthaginian_score.total_cost
    assert roman_score.profile_name == "roman_legion"
    assert any("uphill" in item for item in roman_score.explanation)


def test_mountain_edge_penalty_exceeds_plain_edge_penalty():
    model = HistoricalCostModel()
    profile = roman()
    current = GridCell(GridPoint(0, 0), elevation_m=0, cell_size_m=1_000)
    plain = GridCell(GridPoint(1, 0), elevation_m=0, terrain="plain", cell_size_m=1_000)
    mountain = GridCell(GridPoint(1, 0), elevation_m=200, terrain="mountain", terrain_multiplier=5, cell_size_m=1_000)
    assert model.calculate_movement_penalty(current, mountain, profile) > model.calculate_movement_penalty(current, plain, profile)
    assert model.edge_cost(current, mountain, profile).historical_cost > 0


def test_supply_range_adds_cost_only_after_route_exceeds_range():
    short = route_over(SyntheticGrid.flat(6, 1))
    long = route_over(SyntheticGrid.flat(41, 1))
    profile = ArmyProfile(name="short_supply", supply_range=30)
    short_score = short.evaluate(profile)
    long_score = long.evaluate(profile)
    assert short.metrics.distance_km <= profile.supply_range
    assert long.metrics.distance_km > profile.supply_range
    assert long_score.historical_cost > short_score.historical_cost
    assert any("supply range exceeded" in item for item in long_score.explanation)


def test_baseline_evaluation_preserves_phase_62_cost_and_is_deterministic():
    grid = SyntheticGrid.flat(2, 1)
    grid.set_cell(GridPoint(1, 0), elevation_m=10, terrain="rough", terrain_multiplier=3)
    route = route_over(grid)
    baseline = route.evaluate()
    first = route.evaluate(roman())
    second = route.evaluate(roman())
    assert baseline.historical_cost == 0
    assert baseline.total_cost == route.cost_breakdown.total_cost
    assert first.model_dump() == second.model_dump()
