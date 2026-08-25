"""Parameterized historical movement costs for already-supplied route data."""
from __future__ import annotations

from math import inf

from .costs import SyntheticCostModel
from .grid import GridCell
from .models import ArmyProfile, CandidateRoute, RouteCostBreakdown, RouteScore


class HistoricalCostModel(SyntheticCostModel):
    """Optional composition layer; A* receives an edge-cost provider, not history logic."""

    def calculate_movement_penalty(
        self,
        cell_from: GridCell,
        cell_to: GridCell,
        army_profile: ArmyProfile,
    ) -> float:
        """Penalty for a single edge based on change, terrain labels, and profile preferences."""
        if cell_from.blocked or cell_to.blocked:
            return inf
        elevation_delta = cell_to.elevation_m - cell_from.elevation_m
        uphill_penalty = max(elevation_delta, 0.0) * (1.0 - army_profile.mountain_tolerance)
        downhill_penalty = max(-elevation_delta, 0.0) * 0.05
        terrain_factor = max(cell_to.terrain_multiplier, self.terrain_factor(cell_from, cell_to))
        rough_penalty = (terrain_factor - 1.0) * army_profile.rough_terrain_penalty
        river_penalty = army_profile.river_crossing_penalty if cell_to.terrain == "river" else 0.0
        return uphill_penalty + downhill_penalty + rough_penalty + river_penalty

    def edge_cost(self, current: GridCell, neighbor: GridCell, profile: ArmyProfile) -> RouteCostBreakdown:
        base = super().edge_cost(current, neighbor, profile)
        if base.total_cost == inf:
            return base
        historical_cost = self.calculate_movement_penalty(current, neighbor, profile)
        return RouteCostBreakdown(
            distance_cost=base.distance_cost,
            slope_cost=base.slope_cost,
            terrain_cost=base.terrain_cost,
            barrier_cost=base.barrier_cost,
            historical_cost=historical_cost,
            total_cost=base.total_cost + historical_cost,
        )

    def score_route(self, route: CandidateRoute, profile: ArmyProfile) -> RouteScore:
        """Evaluate an already-built route without changing search, points, or provenance."""
        terrain_cost = route.cost_breakdown.slope_cost + route.cost_breakdown.terrain_cost
        ascent_cost = route.metrics.elevation_gain_m * (1.0 - profile.mountain_tolerance)
        descent_cost = route.metrics.elevation_loss_m * 0.05
        rough_cost = route.cost_breakdown.terrain_cost * profile.rough_terrain_penalty
        supply_excess_km = max(route.metrics.distance_km - profile.supply_range, 0.0)
        supply_cost = supply_excess_km
        historical_cost = ascent_cost + descent_cost + rough_cost + supply_cost
        explanation = [
            f"profile={profile.name}; movement_type={profile.movement_type}",
            f"uphill/elevation-change penalty={ascent_cost + descent_cost:.3f}",
            f"rough-terrain penalty={rough_cost:.3f}",
        ]
        if supply_excess_km:
            explanation.append(f"supply range exceeded by {supply_excess_km:.3f} km")
        else:
            explanation.append("route remains within configured supply range")
        return RouteScore(
            profile_name=profile.name,
            distance_cost=route.cost_breakdown.distance_cost,
            terrain_cost=terrain_cost,
            historical_cost=historical_cost,
            total_cost=route.cost_breakdown.distance_cost + terrain_cost + route.cost_breakdown.barrier_cost + historical_cost,
            explanation=explanation,
        )
