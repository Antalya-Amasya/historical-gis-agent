"""Pure algorithmic orchestration from supplied anchors to ranked candidates."""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .evaluation import RouteEvaluationResult, evaluate_route
from .generator import CandidateRouteGenerator
from .grid import GridPoint, SyntheticGrid
from .models import (
    ArmyProfile,
    CandidateRouteAnchor,
    CandidateRouteConstraints,
    RankedRoute,
    RankedRouteSet,
    RankingProfile,
)
from .ranking import RouteRankingModel


class PlanningConstraints(BaseModel):
    """Persisted planning intent; unsupported fields remain declarative in v1."""

    avoid_rivers: bool = False
    max_distance: float | None = Field(default=None, gt=0)
    candidate_strategies: CandidateRouteConstraints = Field(default_factory=CandidateRouteConstraints)
    allow_disputed_locations: bool = False


@dataclass(frozen=True)
class HistoricalPlanningRequest:
    """All anchors and grid coordinates are supplied by the caller; no evidence lookup occurs here."""

    start_anchor: CandidateRouteAnchor
    end_anchor: CandidateRouteAnchor
    start_grid_point: GridPoint
    end_grid_point: GridPoint
    grid: SyntheticGrid
    army_profile: ArmyProfile
    ranking_profile: RankingProfile = field(default_factory=RankingProfile)
    constraints: PlanningConstraints = field(default_factory=PlanningConstraints)
    location_warnings: list[str] = field(default_factory=list)


class RoutePlanningResult(BaseModel):
    ranked_routes: RankedRouteSet
    selected_route: RankedRoute | None = None
    evaluations: list[RouteEvaluationResult] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    location_warnings: list[str] = Field(default_factory=list)


class HistoricalRoutePlanner:
    """Coordinates existing candidate modules only; it does not implement pathfinding or scoring."""

    def __init__(
        self,
        generator: CandidateRouteGenerator | None = None,
        ranking_model: RouteRankingModel | None = None,
    ) -> None:
        self.generator = generator or CandidateRouteGenerator()
        self.ranking_model = ranking_model or RouteRankingModel()

    def plan(self, request: HistoricalPlanningRequest) -> RoutePlanningResult:
        candidate_set = self.generator.generate(
            from_anchor=request.start_anchor,
            to_anchor=request.end_anchor,
            start=request.start_grid_point,
            goal=request.end_grid_point,
            grid=request.grid,
            army_profile=request.army_profile,
            constraints=request.constraints.candidate_strategies,
        )
        evaluations = [evaluate_route(route, request.army_profile) for route in candidate_set.routes]
        ranked = self.ranking_model.rank(
            candidate_set,
            army_profile=request.army_profile,
            ranking_profile=request.ranking_profile,
        )
        selected = ranked.routes[0] if ranked.routes else None
        explanation = list(selected.reasons) if selected else ["no candidate route was generated"]
        return RoutePlanningResult(
            ranked_routes=ranked,
            selected_route=selected,
            evaluations=evaluations,
            explanation=explanation,
            location_warnings=list(request.location_warnings),
        )
