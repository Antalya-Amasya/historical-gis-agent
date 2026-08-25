import inspect

import pytest

from backend.app.candidate_routes import (
    ArmyProfile,
    CandidateRouteAnchor,
    HistoricalPlanningRequest,
    MultiSegmentHistoricalRoutePlanner,
    MultiSegmentPlanningError,
    MultiSegmentPlanningRequest,
)
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
import backend.app.candidate_routes.multi_segment as multi_module


def segment(start_id, end_id, start, end, *, start_evidence=True, end_evidence=True):
    return HistoricalPlanningRequest(
        start_anchor=CandidateRouteAnchor(
            historical_place_id=start_id, canonical_name=start_id,
            evidence_refs=[f"e-{start_id}"] if start_evidence else [],
        ),
        end_anchor=CandidateRouteAnchor(
            historical_place_id=end_id, canonical_name=end_id,
            evidence_refs=[f"e-{end_id}"] if end_evidence else [],
        ),
        start_grid_point=start,
        end_grid_point=end,
        grid=SyntheticGrid.flat(5, 3),
        army_profile=ArmyProfile(name="campaign"),
    )


def request():
    return MultiSegmentPlanningRequest(segments=[
        segment("a", "b", GridPoint(0, 1), GridPoint(2, 1)),
        segment("b", "c", GridPoint(2, 1), GridPoint(4, 1)),
    ])


def test_two_ordered_segments_plan_and_preserve_anchor_evidence():
    result = MultiSegmentHistoricalRoutePlanner().plan(request())
    assert len(result.segment_results) == 2
    assert len(result.selected_segments) == 2
    assert result.selected_segments[0].route.from_anchor.historical_place_id == "a"
    assert result.selected_segments[0].route.to_anchor.historical_place_id == "b"
    assert result.selected_segments[1].route.to_anchor.historical_place_id == "c"
    assert result.selected_segments[0].route.evidence_refs == ["e-a", "e-b"]


def test_total_cost_is_sum_of_selected_route_scores():
    result = MultiSegmentHistoricalRoutePlanner().plan(request())
    assert result.total_cost == sum(route.score.total_cost for route in result.selected_segments)
    assert "sum of selected RouteScore.total_cost" in result.explanation[1]


def test_missing_evidence_or_grid_coordinate_fails_whole_request():
    missing_evidence = MultiSegmentPlanningRequest(segments=[
        segment("a", "b", GridPoint(0, 1), GridPoint(2, 1)),
        segment("b", "c", GridPoint(2, 1), GridPoint(4, 1), end_evidence=False),
    ])
    with pytest.raises(MultiSegmentPlanningError):
        MultiSegmentHistoricalRoutePlanner().plan(missing_evidence)
    missing_coordinate = MultiSegmentPlanningRequest(segments=[
        segment("a", "b", GridPoint(0, 1), GridPoint(2, 1)),
        segment("b", "c", GridPoint(2, 1), GridPoint(9, 1)),
    ])
    with pytest.raises(MultiSegmentPlanningError):
        MultiSegmentHistoricalRoutePlanner().plan(missing_coordinate)


def test_multi_segment_result_is_deterministic_and_isolated():
    planner = MultiSegmentHistoricalRoutePlanner()
    assert planner.plan(request()).model_dump() == planner.plan(request()).model_dump()
    source = inspect.getsource(multi_module).lower()
    for forbidden in ("agent", "rag", "mcp", "http"):
        assert forbidden not in source
