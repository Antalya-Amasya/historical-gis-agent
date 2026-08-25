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
        terrain_factor = max(neighbor.terrain_multiplier, self.terrain_factor(current, neighbor))
        terrain_cost = profile.terrain_weight * (terrain_factor - 1.0)
        barrier_cost = 0.0
        total_cost = distance_cost + slope_cost + terrain_cost + barrier_cost
        return RouteCostBreakdown(
            distance_cost=distance_cost, slope_cost=slope_cost, terrain_cost=terrain_cost,
            barrier_cost=barrier_cost, total_cost=total_cost,
        )

    @staticmethod
    def terrain_factor(current: GridCell, neighbor: GridCell) -> float:
        """Classify local slope only when a geographic cell size is available.

        Synthetic-grid fixtures keep their explicit terrain multiplier unchanged.
        """
        if neighbor.cell_size_m is None:
            return 1.0
        slope_ratio = abs(neighbor.elevation_m - current.elevation_m) / neighbor.cell_size_m
        if slope_ratio < 0.05:
            return 1.0
        if slope_ratio < 0.15:
            return 2.0
        return 5.0

    @staticmethod
    def heuristic_distance_cost(dx: int, dy: int, profile: ArmyProfile) -> float:
        return (abs(dx) + abs(dy)) * profile.distance_weight
