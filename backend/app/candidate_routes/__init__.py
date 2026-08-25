"""Deterministic, evidence-constrained candidate routing primitives."""
from .engine import CandidateRouteEngine, NoPathError
from .geographic import (
    GeographicCandidateRouteService, GeographicGridSpec, GridTooLargeError,
    LocalProjection, PointOutsideGridError, SyntheticGeographicTerrainProvider, TerrainGrid,
)
from .models import ArmyProfile, CandidateRoute, CandidateRouteAnchor, RouteCostBreakdown, RouteMetrics

__all__ = [
    "ArmyProfile", "CandidateRoute", "CandidateRouteAnchor", "CandidateRouteEngine",
    "GeographicCandidateRouteService", "GeographicGridSpec", "GridTooLargeError",
    "LocalProjection", "NoPathError", "PointOutsideGridError", "RouteCostBreakdown",
    "RouteMetrics", "SyntheticGeographicTerrainProvider", "TerrainGrid",
]
