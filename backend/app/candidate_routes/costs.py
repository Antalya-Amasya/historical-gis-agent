"""Composable deterministic edge costs for the synthetic candidate-route engine."""
from __future__ import annotations

from math import inf, sqrt

from .grid import GridCell
from .models import ArmyProfile, RouteCostBreakdown


class SyntheticCostModel:
    """Non-negative costs make the octile distance heuristic admissible for 8-neighbor A*."""

    def edge_cost(self, current: GridCell, neighbor: GridCell, profile: ArmyProfile) -> RouteCostBreakdown:
        if current.blocked or neighbor.blocked:
            return RouteCostBreakdown(
                distance_cost=0.0, slope_cost=0.0, terrain_cost=0.0,
                barrier_cost=inf, total_cost=inf,
            )
        is_diagonal = self.is_diagonal_step(current, neighbor)
        distance_cost = profile.distance_weight * (sqrt(2) if is_diagonal else 1.0)
        step_distance_m = self.step_distance_m(current, neighbor)
        slope_ratio = self.slope_ratio(current, neighbor)
        slope_cost = self.slope_cost(step_distance_m, slope_ratio, profile.slope_weight)
        terrain_cost = profile.terrain_weight * (neighbor.terrain_multiplier - 1.0)
        barrier_cost = 0.0
        total_cost = distance_cost + slope_cost + terrain_cost + barrier_cost
        return RouteCostBreakdown(
            distance_cost=distance_cost, slope_cost=slope_cost, terrain_cost=terrain_cost,
            barrier_cost=barrier_cost, total_cost=total_cost,
        )

    @staticmethod
    def is_diagonal_step(current: GridCell, neighbor: GridCell) -> bool:
        return abs(neighbor.point.x - current.point.x) == 1 and abs(neighbor.point.y - current.point.y) == 1

    @classmethod
    def step_distance_m(cls, current: GridCell, neighbor: GridCell) -> float:
        base_cell_size_m = current.cell_size_m if current.cell_size_m and current.cell_size_m > 0 else 1_000.0
        return base_cell_size_m * (sqrt(2) if cls.is_diagonal_step(current, neighbor) else 1.0)

    @classmethod
    def slope_ratio(cls, current: GridCell, neighbor: GridCell) -> float:
        return abs(neighbor.elevation_m - current.elevation_m) / cls.step_distance_m(current, neighbor)

    @staticmethod
    def slope_cost(step_distance_m: float, slope_ratio: float, slope_weight: float) -> float:
        if slope_ratio < 0.05:
            return 0.0
        multiplier = 1.5 if slope_ratio < 0.15 else 5.0
        return step_distance_m * slope_weight * multiplier

    @staticmethod
    def heuristic_distance_cost(dx: int, dy: int, profile: ArmyProfile) -> float:
        horizontal, vertical = abs(dx), abs(dy)
        return profile.distance_weight * (horizontal + vertical + (sqrt(2) - 2.0) * min(horizontal, vertical))
