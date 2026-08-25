import inspect

from backend.app.candidate_routes import (
    ArmyProfile,
    CandidateRouteAnchor,
    HistoricalPlanningRequest,
    HistoricalRoutePlanner,
    RankingProfile,
)
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
import backend.app.candidate_routes.planner as planner_module


def request(*, mountain_tolerance=1.0, ranking_profile=None):
    grid = SyntheticGrid.flat(7, 3)
    for x in (2, 3, 4):
        grid.set_cell(GridPoint(x, 1), elevation_m=100, terrain="mountain", terrain_multiplier=20)
    return HistoricalPlanningRequest(
        start_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        end_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start_grid_point=GridPoint(0, 1),
        end_grid_point=GridPoint(6, 1),
        grid=grid,
        army_profile=ArmyProfile(name="campaign", mountain_tolerance=mountain_tolerance, rough_terrain_penalty=2),
        ranking_profile=ranking_profile or RankingProfile(),
    )


def test_planner_runs_full_candidate_evaluation_ranking_pipeline():
    result = HistoricalRoutePlanner().plan(request())
    assert len(result.ranked_routes.routes) == 3
    assert len(result.evaluations) == 3
    assert result.selected_route is result.ranked_routes.routes[0]
    assert result.selected_route.rank == 1
    assert result.explanation == result.selected_route.reasons


def test_ranking_profile_or_movement_preferences_can_change_selected_route():
    planner = HistoricalRoutePlanner()
    distance_first = planner.plan(request(mountain_tolerance=1.0, ranking_profile=RankingProfile(distance_weight=1, terrain_weight=0, historical_weight=0)))
    historical_first = planner.plan(request(mountain_tolerance=0.0, ranking_profile=RankingProfile(distance_weight=0, terrain_weight=0, historical_weight=1)))
    assert distance_first.selected_route is not None and historical_first.selected_route is not None
    assert distance_first.selected_route.route.geometry.coordinates != historical_first.selected_route.route.geometry.coordinates


def test_planner_is_deterministic_and_has_no_agent_rag_or_mcp_dependencies():
    planner = HistoricalRoutePlanner()
    first = planner.plan(request(mountain_tolerance=0.3))
    second = planner.plan(request(mountain_tolerance=0.3))
    assert first.model_dump() == second.model_dump()
    source = inspect.getsource(planner_module)
    assert "agent" not in source.lower()
    assert "rag" not in source.lower()
    assert "mcp" not in source.lower()
