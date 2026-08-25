"""Composable deterministic edge costs for the synthetic candidate-route engine."""
from __future__ import annotations

from math import inf

from .grid import GridCell
from .models import ArmyProfile, RouteCostBreakdown


class SyntheticCostModel:
    """Non-negative costs make the Manhattan distance heuristic admissible for 4-neighbor A*."""

    def edge_cost(self, current: GridCell, neighbor: GridCell, profile: ArmyProfile) -> RouteCostBreakdown:
        if current.blocked or neighbor.blocked:
            return RouteCostBreakdown(
                distance_cost=0.0, slope_cost=0.0, terrain_cost=0.0,
                barrier_cost=inf, total_cost=inf,
            )
        distance_cost = profile.distance_weight
        slope_cost = profile.slope_weight * abs(neighbor.elevation_m - current.elevation_m)
        terrain_cost = profile.terrain_weight * (neighbor.terrain_multiplier - 1.0)
        barrier_cost = 0.0
        total_cost = distance_cost + slope_cost + terrain_cost + barrier_cost
        return RouteCostBreakdown(
            distance_cost=distance_cost, slope_cost=slope_cost, terrain_cost=terrain_cost,
            barrier_cost=barrier_cost, total_cost=total_cost,
        )

    @staticmethod
    def heuristic_distance_cost(dx: int, dy: int, profile: ArmyProfile) -> float:
        return (abs(dx) + abs(dy)) * profile.distance_weight
