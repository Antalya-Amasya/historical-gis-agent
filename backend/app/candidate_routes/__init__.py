"""Deterministic, evidence-constrained candidate routing primitives."""
from .engine import CandidateRouteEngine, NoPathError
from .corpus import HistoricalCorpus
from .generator import CandidateRouteGenerator
from .evaluation import (
    RouteEvaluationProvenance, RouteEvaluationResult, RouteExplanation, RouteExplanationFactor,
    build_route_explanation, evaluate_route, route_to_geojson, serialize_route_evaluation,
)
from .geographic import (
    GeographicCandidateRouteService, GeographicGridSpec, GridTooLargeError,
    LocalProjection, PointOutsideGridError, SyntheticGeographicTerrainProvider, TerrainGrid,
)
from .historical_costs import HistoricalCostModel
from .historical_adapter import (
    CoordinateResolutionError, HistoricalRouteAdapter, HistoricalRouteEvidenceError,
    HistoricalRoutePlanningService, MockCoordinateResolver, RouteCoordinateResolver,
)
from .ranking import RouteRankingModel
from .planner import HistoricalPlanningRequest, HistoricalRoutePlanner, PlanningConstraints, RoutePlanningResult
from .presentation import (
    HistoricalRouteExplanations, HistoricalRoutePresentation, HistoricalRoutePresentationService,
    HistoricalRouteResponse, HistoricalWaypointSegmentView, CoordinateAwareWaypointView,
    LocationAwarePresentationService,
)
from .location import LocationConfidence, ResolvedCoordinate
from .knowledge_panel import (
    EvidenceBackedSummary, EvidenceBackedSummaryProvider, HistoricalKnowledgePanel,
    HistoricalSummaryProvider, KnowledgePanelBuilder, SummaryEvidenceError, SummaryResult,
)
from .multi_segment import (
    MultiSegmentHistoricalRoutePlanner, MultiSegmentPlanningError, MultiSegmentPlanningRequest,
    MultiSegmentRoutePlanningResult,
)
from .models import (
    ArmyProfile, CandidateRoute, CandidateRouteAnchor, CandidateRouteConstraints, CandidateRouteSet,
    CandidateRouteSegmentLedger, RankedRoute, RankedRouteSet, RankingProfile, RouteCostBreakdown, RouteMetrics, RouteScore,
)
from .waypoint_graph import (
    HistoricalEventChain, HistoricalEventStep, HistoricalWaypoint, HistoricalWaypointBuilder,
    HistoricalWaypointEvidenceError, HistoricalWaypointGraph, HistoricalWaypointRole,
    HistoricalWaypointSegment, WaypointBuilder, WaypointGraphPlanningAdapter,
)
from .terrain import (
    DEMTerrainProvider, HgtRaster, MosaicDEMProvider, OfflineMockTerrainProvider, RealTerrainProvider,
    SyntheticTerrainProvider, TerrainDataUnavailableError, TerrainProvider, UnsupportedDemError,
)
from .historical_reconstruction import (
    HistoricalRouteReconstructor, OfflineMockTerrainGraphProvider, RealTerrainGraphProvider,
    ReconstructedHistoricalRoute, ReviewedHistoricalWaypoint, TerrainGraph,
)

__all__ = [
    "HistoricalCorpus", "ArmyProfile", "CandidateRoute", "CandidateRouteAnchor", "CandidateRouteConstraints", "CandidateRouteEngine",
    "CandidateRouteGenerator", "CandidateRouteSet", "HistoricalCostModel", "HistoricalRouteAdapter",
    "HistoricalRouteEvidenceError", "HistoricalRoutePlanningService", "MockCoordinateResolver",
    "RouteCoordinateResolver", "CoordinateResolutionError", "RankedRoute", "RankedRouteSet",
    "RankingProfile", "RouteRankingModel", "HistoricalPlanningRequest", "HistoricalRoutePlanner",
    "PlanningConstraints", "RoutePlanningResult", "HistoricalRouteExplanations", "HistoricalRoutePresentation",
    "HistoricalRoutePresentationService", "HistoricalRouteResponse", "HistoricalWaypointSegmentView",
    "CoordinateAwareWaypointView", "LocationAwarePresentationService",
    "MultiSegmentHistoricalRoutePlanner",
    "MultiSegmentPlanningError", "MultiSegmentPlanningRequest", "MultiSegmentRoutePlanningResult",
    "RouteEvaluationProvenance", "RouteEvaluationResult", "RouteExplanation", "RouteExplanationFactor",
    "build_route_explanation", "evaluate_route", "route_to_geojson", "serialize_route_evaluation",
    "GeographicCandidateRouteService", "GeographicGridSpec", "GridTooLargeError",
    "LocalProjection", "NoPathError", "PointOutsideGridError", "RouteCostBreakdown",
    "RouteMetrics", "RouteScore", "CandidateRouteSegmentLedger", "SyntheticGeographicTerrainProvider", "TerrainGrid",
    "DEMTerrainProvider", "HgtRaster", "MosaicDEMProvider", "OfflineMockTerrainProvider", "RealTerrainProvider",
    "SyntheticTerrainProvider", "TerrainDataUnavailableError", "TerrainProvider", "UnsupportedDemError",
    "HistoricalRouteReconstructor", "OfflineMockTerrainGraphProvider", "RealTerrainGraphProvider",
    "ReconstructedHistoricalRoute", "ReviewedHistoricalWaypoint", "TerrainGraph",
    "HistoricalEventChain", "HistoricalEventStep",
    "HistoricalWaypoint", "HistoricalWaypointBuilder", "HistoricalWaypointEvidenceError",
    "HistoricalWaypointGraph", "HistoricalWaypointRole", "HistoricalWaypointSegment", "WaypointBuilder",
    "WaypointGraphPlanningAdapter", "LocationConfidence", "ResolvedCoordinate",
    "HistoricalKnowledgePanel", "KnowledgePanelBuilder", "HistoricalSummaryProvider",
    "EvidenceBackedSummaryProvider", "SummaryResult", "EvidenceBackedSummary",
    "SummaryEvidenceError",
]
