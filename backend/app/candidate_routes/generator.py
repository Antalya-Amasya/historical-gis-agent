"""Deterministic candidate-route variants over one caller-supplied grid."""
from __future__ import annotations

from .costs import SyntheticCostModel
from .engine import CandidateRouteEngine
from .grid import GridPoint, SyntheticGrid
from .historical_costs import HistoricalCostModel
from .models import ArmyProfile, CandidateRoute, CandidateRouteAnchor, CandidateRouteConstraints, CandidateRouteSet


class CandidateRouteGenerator:
    """Generates configured A* variants only; no randomization, LLM, or route ranking."""

    def generate(
        self,
        *,
        from_anchor: CandidateRouteAnchor,
        to_anchor: CandidateRouteAnchor,
        start: GridPoint,
        goal: GridPoint,
        grid: SyntheticGrid,
        army_profile: ArmyProfile | None = None,
        constraints: CandidateRouteConstraints | None = None,
    ) -> CandidateRouteSet:
        constraints = constraints or CandidateRouteConstraints()
        historical_profile = army_profile or ArmyProfile()
        routes: list[CandidateRoute] = []
        if constraints.include_shortest_distance:
            routes.append(self._build(
                "shortest_distance_route",
                CandidateRouteEngine(SyntheticCostModel()),
                ArmyProfile(name="shortest_distance", distance_weight=1, slope_weight=0, terrain_weight=0, barrier_weight=1),
                from_anchor, to_anchor, start, goal, grid,
            ))
        if constraints.include_terrain_optimized:
            routes.append(self._build(
                "terrain_optimized_route",
                CandidateRouteEngine(SyntheticCostModel()),
                ArmyProfile(name="terrain_optimized", distance_weight=1, slope_weight=1, terrain_weight=5, barrier_weight=1),
                from_anchor, to_anchor, start, goal, grid,
            ))
        if constraints.include_historical_profile:
            routes.append(self._build(
                "historical_profile_route",
                CandidateRouteEngine(HistoricalCostModel()), historical_profile,
                from_anchor, to_anchor, start, goal, grid,
            ))
        return CandidateRouteSet(
            routes=routes,
            metadata={
                "strategy_count": len(routes),
                "profile_name": historical_profile.name,
                "deterministic": True,
            },
        )

    @staticmethod
    def _build(
        generation_method: str,
        engine: CandidateRouteEngine,
        profile: ArmyProfile,
        from_anchor: CandidateRouteAnchor,
        to_anchor: CandidateRouteAnchor,
        start: GridPoint,
        goal: GridPoint,
        grid: SyntheticGrid,
    ) -> CandidateRoute:
        route = engine.build_route(
            from_anchor=from_anchor, to_anchor=to_anchor, start=start, goal=goal, grid=grid, profile=profile,
        )
        return route.model_copy(update={
            "id": f"{route.id}-{generation_method}",
            "generation_method": generation_method,
        })
