"""A deterministic 8-neighbor A* route engine over synthetic geography."""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import isinf

from backend.app.models import GeoJsonLineString

from .costs import SyntheticCostModel
from .grid import GridPoint, SyntheticGrid
from .models import ArmyProfile, CandidateRoute, CandidateRouteAnchor, RouteCostBreakdown, RouteMetrics


class NoPathError(ValueError):
    pass


@dataclass(frozen=True)
class CandidatePath:
    points: tuple[GridPoint, ...]
    cost_breakdown: RouteCostBreakdown
    elevation_gain_m: float
    elevation_loss_m: float


class CandidateRouteEngine:
    """Connects caller-supplied Evidence anchors; it never selects or invents anchors."""

    def __init__(self, cost_model: SyntheticCostModel | None = None):
        self.cost_model = cost_model or SyntheticCostModel()

    def find_path(self, grid: SyntheticGrid, start: GridPoint, goal: GridPoint, profile: ArmyProfile) -> CandidatePath:
        start_cell, goal_cell = grid.cell(start), grid.cell(goal)
        if start_cell.blocked or goal_cell.blocked:
            raise NoPathError("start and goal must be traversable")
        queue: list[tuple[float, float, float, int, int, int, GridPoint]] = []
        counter = 0
        initial_h = self.cost_model.heuristic_distance_cost(goal.x - start.x, goal.y - start.y, profile)
        heappush(queue, (initial_h, initial_h, 0.0, start.y, start.x, counter, start))
        came_from: dict[GridPoint, GridPoint] = {}
        g_score: dict[GridPoint, float] = {start: 0.0}
        edge_costs: dict[GridPoint, RouteCostBreakdown] = {}

        while queue:
            _, _, current_cost, _, _, _, current = heappop(queue)
            if current_cost != g_score.get(current):
                continue
            if current == goal:
                return self._reconstruct(grid, start, goal, came_from, edge_costs)
            for neighbor in grid.neighbors8(current):
                breakdown = self.cost_model.edge_cost(grid.cell(current), grid.cell(neighbor), profile)
                if isinf(breakdown.total_cost):
                    continue
                candidate_cost = current_cost + breakdown.total_cost
                previous = g_score.get(neighbor)
                if previous is not None and candidate_cost >= previous:
                    continue
                came_from[neighbor] = current
                edge_costs[neighbor] = breakdown
                g_score[neighbor] = candidate_cost
                heuristic = self.cost_model.heuristic_distance_cost(goal.x - neighbor.x, goal.y - neighbor.y, profile)
                counter += 1
                heappush(queue, (candidate_cost + heuristic, heuristic, candidate_cost, neighbor.y, neighbor.x, counter, neighbor))
        raise NoPathError("no traversable path exists between supplied anchors")

    def build_route(
        self,
        *,
        from_anchor: CandidateRouteAnchor,
        to_anchor: CandidateRouteAnchor,
        start: GridPoint,
        goal: GridPoint,
        grid: SyntheticGrid,
        profile: ArmyProfile,
    ) -> CandidateRoute:
        path = self.find_path(grid, start, goal, profile)
        refs = list(dict.fromkeys([*from_anchor.evidence_refs, *to_anchor.evidence_refs]))
        return CandidateRoute(
            id=f"candidate-{from_anchor.historical_place_id}-to-{to_anchor.historical_place_id}",
            from_anchor=from_anchor,
            to_anchor=to_anchor,
            geometry=GeoJsonLineString(coordinates=[(float(point.x), float(point.y)) for point in path.points]),
            metrics=RouteMetrics(
                distance_km=sum(
                    self.cost_model.step_distance_m(grid.cell(first), grid.cell(second))
                    for first, second in zip(path.points, path.points[1:])
                ) / 1_000.0,
                elevation_gain_m=path.elevation_gain_m,
                elevation_loss_m=path.elevation_loss_m,
                estimated_cost=path.cost_breakdown.total_cost,
                cell_count=len(path.points),
                segment_count=max(0, len(path.points) - 1),
                search_cost=path.cost_breakdown.total_cost,
                max_slope=max(
                    (self.cost_model.slope_ratio(grid.cell(first), grid.cell(second)) for first, second in zip(path.points, path.points[1:])),
                    default=0.0,
                ),
            ),
            cost_breakdown=path.cost_breakdown,
            confidence=0.0,
            assumptions=["A* search costs are normalized grid costs; physical distance_km is calculated from metre-valued grid edges.", "This is an algorithmic candidate connection, not an evidence-grounded historical track."],
            evidence_refs=refs,
        )

    @staticmethod
    def _reconstruct(
        grid: SyntheticGrid,
        start: GridPoint,
        goal: GridPoint,
        came_from: dict[GridPoint, GridPoint],
        edge_costs: dict[GridPoint, RouteCostBreakdown],
    ) -> CandidatePath:
        points = [goal]
        while points[-1] != start:
            points.append(came_from[points[-1]])
        points.reverse()
        components = [edge_costs[point] for point in points[1:]]
        breakdown = RouteCostBreakdown(
            distance_cost=sum(item.distance_cost for item in components),
            slope_cost=sum(item.slope_cost for item in components),
            terrain_cost=sum(item.terrain_cost for item in components),
            barrier_cost=sum(item.barrier_cost for item in components),
            historical_cost=sum(item.historical_cost for item in components),
            total_cost=sum(item.total_cost for item in components),
        )
        gain = loss = 0.0
        for current, neighbor in zip(points, points[1:]):
            delta = grid.cell(neighbor).elevation_m - grid.cell(current).elevation_m
            if delta >= 0:
                gain += delta
            else:
                loss += -delta
        return CandidatePath(tuple(points), breakdown, gain, loss)
