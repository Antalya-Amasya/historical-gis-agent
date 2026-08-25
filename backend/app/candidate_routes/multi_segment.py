"""Deterministic all-or-nothing planning over supplied historical route segments."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from .models import RankedRoute
from .planner import HistoricalPlanningRequest, HistoricalRoutePlanner, RoutePlanningResult


class MultiSegmentPlanningError(ValueError):
    """A single invalid segment prevents returning a misleading partial route plan."""


@dataclass(frozen=True)
class MultiSegmentPlanningRequest:
    segments: list[HistoricalPlanningRequest]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("multi-segment planning requires at least one segment")


class MultiSegmentRoutePlanningResult(BaseModel):
    segment_results: list[RoutePlanningResult] = Field(default_factory=list)
    selected_segments: list[RankedRoute] = Field(default_factory=list)
    total_cost: float = Field(ge=0)
    explanation: list[str] = Field(default_factory=list)


class MultiSegmentHistoricalRoutePlanner:
    """Sequential, all-or-nothing coordination of the existing single-segment planner."""

    def __init__(self, planner: HistoricalRoutePlanner | None = None):
        self.planner = planner or HistoricalRoutePlanner()

    def plan(self, request: MultiSegmentPlanningRequest) -> MultiSegmentRoutePlanningResult:
        segment_results: list[RoutePlanningResult] = []
        selected_segments: list[RankedRoute] = []
        for index, segment in enumerate(request.segments, start=1):
            self._validate_segment(segment, index)
            try:
                result = self.planner.plan(segment)
            except (ValueError, KeyError) as exc:
                raise MultiSegmentPlanningError(f"segment {index} failed: {exc}") from exc
            if result.selected_route is None:
                raise MultiSegmentPlanningError(f"segment {index} generated no selected route")
            segment_results.append(result)
            selected_segments.append(result.selected_route)
        total_cost = sum(item.score.total_cost for item in selected_segments)
        return MultiSegmentRoutePlanningResult(
            segment_results=segment_results,
            selected_segments=selected_segments,
            total_cost=total_cost,
            explanation=[
                "selected rank-1 route for each segment in supplied historical order",
                "total cost is the sum of selected RouteScore.total_cost values",
            ],
        )

    @staticmethod
    def _validate_segment(segment: HistoricalPlanningRequest, index: int) -> None:
        if not segment.start_anchor.evidence_refs or not segment.end_anchor.evidence_refs:
            raise MultiSegmentPlanningError(f"segment {index} has an anchor without evidence references")
        if not segment.grid.contains(segment.start_grid_point) or not segment.grid.contains(segment.end_grid_point):
            raise MultiSegmentPlanningError(f"segment {index} grid coordinate is unavailable")
