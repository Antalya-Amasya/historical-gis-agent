from backend.app.candidate_routes import (
    ArmyProfile,
    CandidateRouteAnchor,
    CandidateRouteGenerator,
    CandidateRouteSet,
    RankingProfile,
    RouteRankingModel,
)
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid


def anchors():
    return (
        CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
    )


def varied_grid():
    grid = SyntheticGrid.flat(7, 3)
    for x in (2, 3, 4):
        grid.set_cell(GridPoint(x, 1), elevation_m=100, terrain="mountain", terrain_multiplier=20)
    return grid


def generated_set():
    first, second = anchors()
    return CandidateRouteGenerator().generate(
        from_anchor=first, to_anchor=second,
        start=GridPoint(0, 1), goal=GridPoint(6, 1), grid=varied_grid(),
        army_profile=ArmyProfile(name="roman_legion", mountain_tolerance=0.3, rough_terrain_penalty=2),
    )


def test_generator_creates_three_deterministic_strategy_variants():
    result = generated_set()
    assert len(result.routes) == 3
    assert result.generation_method == "deterministic_variants"
    assert {route.generation_method for route in result.routes} == {
        "shortest_distance_route", "terrain_optimized_route", "historical_profile_route",
    }
    shortest = next(route for route in result.routes if route.generation_method == "shortest_distance_route")
    terrain = next(route for route in result.routes if route.generation_method == "terrain_optimized_route")
    assert shortest.geometry.coordinates != terrain.geometry.coordinates


def test_ranking_puts_lowest_weighted_cost_first():
    routes = generated_set()
    ranked = RouteRankingModel().rank(routes, army_profile=ArmyProfile(name="roman_legion", mountain_tolerance=0.3))
    values = [item.ranking_score for item in ranked.routes]
    assert values == sorted(values)
    assert ranked.routes[0].rank == 1
    assert ranked.routes[0].reasons


def test_ranking_weights_can_change_order_for_existing_routes():
    base = generated_set().routes[0]
    distance_route = base.model_copy(update={
        "id": "distance-route",
        "cost_breakdown": base.cost_breakdown.model_copy(update={"distance_cost": 10, "slope_cost": 0, "terrain_cost": 0, "total_cost": 10}),
    })
    terrain_route = base.model_copy(update={
        "id": "terrain-route",
        "cost_breakdown": base.cost_breakdown.model_copy(update={"distance_cost": 1, "slope_cost": 0, "terrain_cost": 10, "total_cost": 11}),
    })
    route_set = CandidateRouteSet(routes=[distance_route, terrain_route])
    model = RouteRankingModel()
    distance_first = model.rank(route_set, ranking_profile=RankingProfile(distance_weight=1, terrain_weight=0, historical_weight=0))
    terrain_first = model.rank(route_set, ranking_profile=RankingProfile(distance_weight=0, terrain_weight=1, historical_weight=0))
    assert distance_first.routes[0].route.id == "terrain-route"
    assert terrain_first.routes[0].route.id == "distance-route"


def test_generation_and_ranking_are_deterministic():
    first = generated_set()
    second = generated_set()
    assert first.model_dump() == second.model_dump()
    ranking = RouteRankingModel()
    assert ranking.rank(first).model_dump() == ranking.rank(second).model_dump()
