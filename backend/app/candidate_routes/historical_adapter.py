"""Offline adapter from evidence-grounded HistoricalRoute to candidate planning input."""
from __future__ import annotations

from .location import (
    CoordinateResolutionError, MockCoordinateResolver, RouteCoordinateResolver, validate_resolved_coordinate,
)
from .grid import SyntheticGrid
from .models import ArmyProfile, CandidateRouteAnchor, RankingProfile
from .planner import HistoricalPlanningRequest, HistoricalRoutePlanner, PlanningConstraints, RoutePlanningResult


class HistoricalRouteEvidenceError(ValueError):
    """A route without supplied evidence must not enter candidate planning."""


class HistoricalRouteAdapter:
    """Converts only supplied route anchors and evidence into an explicit planning request."""

    def __init__(self, coordinate_resolver: RouteCoordinateResolver):
        self.coordinate_resolver = coordinate_resolver

    @staticmethod
    def to_anchor(point: HistoricalRoutePoint) -> CandidateRouteAnchor:
        return CandidateRouteAnchor(
            historical_place_id=point.historical_place.id,
            canonical_name=point.historical_place.canonical_name,
            evidence_refs=list(point.evidence_refs),
        )

    def to_planning_request(
        self,
        historical_route: HistoricalRoute,
        *,
        grid: SyntheticGrid,
        army_profile: ArmyProfile,
        ranking_profile: RankingProfile | None = None,
        constraints: PlanningConstraints | None = None,
    ) -> HistoricalPlanningRequest:
        if not historical_route.evidence_refs:
            raise HistoricalRouteEvidenceError("historical route has no route-level evidence references")
        if len(historical_route.ordered_points) < 2:
            raise HistoricalRouteEvidenceError("historical route needs at least two ordered evidence-grounded points")
        start = self.to_anchor(historical_route.ordered_points[0])
        end = self.to_anchor(historical_route.ordered_points[-1])
        if not start.evidence_refs or not end.evidence_refs:
            raise HistoricalRouteEvidenceError("selected historical route anchors must retain evidence references")
        constraints = constraints or PlanningConstraints()
        start_resolution = self.coordinate_resolver.resolve(start)
        end_resolution = self.coordinate_resolver.resolve(end)
        location_warnings = [
            *validate_resolved_coordinate(start, start_resolution, allow_disputed_locations=constraints.allow_disputed_locations),
            *validate_resolved_coordinate(end, end_resolution, allow_disputed_locations=constraints.allow_disputed_locations),
        ]
        return HistoricalPlanningRequest(
            start_anchor=start,
            end_anchor=end,
            start_grid_point=start_resolution.point,
            end_grid_point=end_resolution.point,
            grid=grid,
            army_profile=army_profile,
            ranking_profile=ranking_profile or RankingProfile(),
            constraints=constraints,
            location_warnings=location_warnings,
        )


class HistoricalRoutePlanningService:
    """Offline composition of route adaptation and the existing deterministic planner."""

    def __init__(self, adapter: HistoricalRouteAdapter, planner: HistoricalRoutePlanner | None = None):
        self.adapter = adapter
        self.planner = planner or HistoricalRoutePlanner()

    def plan(
        self,
        historical_route: HistoricalRoute,
        *,
        grid: SyntheticGrid,
        army_profile: ArmyProfile,
        ranking_profile: RankingProfile | None = None,
        constraints: PlanningConstraints | None = None,
    ) -> RoutePlanningResult:
        request = self.adapter.to_planning_request(
            historical_route,
            grid=grid,
            army_profile=army_profile,
            ranking_profile=ranking_profile,
            constraints=constraints,
        )
        return self.planner.plan(request)
