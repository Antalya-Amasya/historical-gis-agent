"""Offline presentation DTOs for evidence-grounded route and waypoint display."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .evaluation import RouteEvaluationResult
from .knowledge_panel import HistoricalKnowledgePanel
from .location import CoordinateResolutionError, LocationConfidence, RouteCoordinateResolver
from .models import CandidateRoute, CandidateRouteAnchor, RouteScore
from .view_model import HistoricalWaypointViewModel, to_waypoint_view_model
from .waypoint_graph import HistoricalWaypointGraph


class HistoricalRouteExplanations(BaseModel):
    distance_reason: str
    terrain_reason: str
    historical_reason: str


class CoordinateAwareWaypointView(BaseModel):
    waypoint_id: str
    name: str
    coordinate: tuple[float, float] | None = None
    location_confidence: LocationConfidence
    location_notes: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    warning: str | None = None


class HistoricalWaypointSegmentView(BaseModel):
    from_waypoint_id: str
    to_waypoint_id: str
    evidence_refs: list[str] = Field(default_factory=list)


class HistoricalRoutePresentation(BaseModel):
    route_id: str
    route_name: str | None = None
    period: str | None = None
    waypoints: list[HistoricalWaypointViewModel] = Field(default_factory=list)
    segments: list[HistoricalWaypointSegmentView] = Field(default_factory=list)
    geojson: dict[str, object]
    route_geojson: dict[str, object] = Field(default_factory=dict)
    summary: str
    confidence: float = Field(ge=0, le=1)
    score: RouteScore
    explanations: HistoricalRouteExplanations
    location_warnings: list[str] = Field(default_factory=list)


class HistoricalDataSource(BaseModel):
    """Auditable metadata for a presentation input; never a claim of historical fact."""

    source_type: str
    dataset_name: str
    version: str | None = None
    license: str | None = None
    confidence: float = Field(ge=0, le=1)


class PresentationTimelineStep(BaseModel):
    """Display-only ordering copied from reviewed HistoricalRoute points."""

    order: int = Field(ge=1)
    title: str
    description: str | None = None
    period: str | None = None
    evidence_count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class PresentationSummary(BaseModel):
    """Structured, display-safe route text assembled only from reviewed presentation inputs."""

    title: str
    campaign_id: str | None = None
    campaign: str | None = None
    operation_id: str | None = None
    operation: str | None = None
    date: str | None = None
    historical_context: str
    route_method: str
    evidence_basis: list[str] = Field(default_factory=list)
    route_interpretation: str
    route_stages: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    geographic_constraints: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    timeline: list[PresentationTimelineStep] = Field(default_factory=list)


class RoutePresentationFragment(BaseModel):
    component_id: str
    status: Literal["COMPLETE", "FAILED", "SKIPPED"]
    evidence_refs: list[str] = Field(default_factory=list)
    waypoints: list[dict[str, object]] = Field(default_factory=list)
    route_geojson: dict[str, object] | None = None
    reason_code: str | None = None


class HistoricalRouteResponse(BaseModel):
    """HTTP-neutral response contract retained as plain data only in this phase."""

    route: HistoricalRoutePresentation
    waypoints: list[HistoricalWaypointViewModel] = Field(default_factory=list)
    geojson: dict[str, object]
    route_geojson: dict[str, object] = Field(default_factory=dict)
    explanations: HistoricalRouteExplanations
    location_warnings: list[str] = Field(default_factory=list)
    knowledge_panels: list[HistoricalKnowledgePanel] = Field(default_factory=list)
    presentation_summary: PresentationSummary | None = None
    fragments: list[RoutePresentationFragment] = Field(default_factory=list)


class HistoricalRoutePresentationService:
    """Combines supplied graph, route and evaluation data without route planning or enrichment."""

    def present(
        self,
        waypoint_graph: HistoricalWaypointGraph,
        candidate_route: CandidateRoute,
        evaluation: RouteEvaluationResult,
        *,
        route_name: str | None = None,
        period: str | None = None,
    ) -> HistoricalRoutePresentation:
        if evaluation.route_id != candidate_route.id:
            raise ValueError("evaluation route_id must match candidate route")
        waypoints = [to_waypoint_view_model(waypoint) for waypoint in waypoint_graph.waypoints]
        segments = [
            HistoricalWaypointSegmentView(
                from_waypoint_id=segment.from_waypoint.id,
                to_waypoint_id=segment.to_waypoint.id,
                evidence_refs=list(segment.evidence_refs),
            )
            for segment in waypoint_graph.segments
        ]
        explanations = self._explanations(waypoint_graph, evaluation)
        route_geojson = self._route_geojson(candidate_route, evaluation.score)
        geojson = self._geojson(route_geojson, waypoints)
        return HistoricalRoutePresentation(
            route_id=candidate_route.id,
            route_name=route_name,
            period=period,
            waypoints=waypoints,
            segments=segments,
            geojson=geojson,
            route_geojson=route_geojson,
            summary=f"Route connects {len(waypoints)} supplied historical waypoint(s).",
            confidence=candidate_route.confidence,
            score=evaluation.score,
            explanations=explanations,
        )

    @staticmethod
    def to_response(presentation: HistoricalRoutePresentation) -> HistoricalRouteResponse:
        return HistoricalRouteResponse(
            route=presentation,
            waypoints=presentation.waypoints,
            geojson=presentation.geojson,
            route_geojson=presentation.route_geojson,
            explanations=presentation.explanations,
        )

    @staticmethod
    def _explanations(graph: HistoricalWaypointGraph, evaluation: RouteEvaluationResult) -> HistoricalRouteExplanations:
        terrain_reason = (
            "Terrain cost is present in the supplied route evaluation."
            if evaluation.score.terrain_cost > 0
            else "No terrain cost is present in the supplied route evaluation."
        )
        evidence_count = len({reference for waypoint in graph.waypoints for reference in waypoint.evidence_refs})
        return HistoricalRouteExplanations(
            distance_reason=f"The route contains {len(graph.waypoints)} supplied historical waypoint(s).",
            terrain_reason=terrain_reason,
            historical_reason=f"The graph retains {evidence_count} supplied evidence reference(s) across its waypoints.",
        )

    @staticmethod
    def _route_geojson(route: CandidateRoute, score: RouteScore) -> dict[str, object]:
        return {
            "type": "Feature",
            "geometry": route.geometry.model_dump(mode="json"),
            "properties": {
                "route_id": route.id,
                "route_type": "schematic_historical_route",
                "confidence": route.confidence,
                "total_cost": score.total_cost,
                "explanation": "Line connects historically attested locations and does not represent an exact marching path.",
            },
        }

    @staticmethod
    def _geojson(
        route_feature: dict[str, object],
        waypoints: list[HistoricalWaypointViewModel],
    ) -> dict[str, object]:
        waypoint_features = [
            {
                "type": "Feature",
                # No point geometry is fabricated: graph input has no resolved coordinates.
                "geometry": None,
                "properties": {**view.model_dump(mode="json"), "knowledge_panel_id": view.id},
            }
            for view in waypoints
        ]
        return {"type": "FeatureCollection", "features": [route_feature, *waypoint_features]}


class LocationAwarePresentationService(HistoricalRoutePresentationService):
    """Adds resolver-supplied display coordinates only; routing modules never consume them."""

    def present(
        self,
        waypoint_graph: HistoricalWaypointGraph,
        candidate_route: CandidateRoute,
        evaluation: RouteEvaluationResult,
        coordinate_resolver: RouteCoordinateResolver,
        *,
        allow_disputed_locations: bool = False,
        route_name: str | None = None,
        period: str | None = None,
    ) -> HistoricalRoutePresentation:
        base = super().present(
            waypoint_graph, candidate_route, evaluation, route_name=route_name, period=period,
        )
        coordinate_views = [
            self._coordinate_view(waypoint, coordinate_resolver, allow_disputed_locations)
            for waypoint in waypoint_graph.waypoints
        ]
        waypoint_features = [
            {
                "type": "Feature",
                "geometry": (
                    {"type": "Point", "coordinates": list(view.coordinate)}
                    if view.coordinate is not None else None
                ),
                "properties": {
                    "waypoint_id": view.waypoint_id,
                    "name": view.name,
                    "location_confidence": view.location_confidence.value,
                    "location_notes": view.location_notes,
                    "evidence_refs": list(view.evidence_refs),
                    "warning": view.warning,
                    "knowledge_panel_id": view.waypoint_id,
                },
            }
            for view in coordinate_views
        ]
        geojson = dict(base.geojson)
        geojson["features"] = [base.geojson["features"][0], *waypoint_features]
        warnings = [view.warning for view in coordinate_views if view.warning]
        coordinate_by_id = {view.waypoint_id: view for view in coordinate_views}
        resolved_waypoints = [
            waypoint.model_copy(update={
                "location_confidence": coordinate_by_id[waypoint.id].location_confidence,
                "location_notes": coordinate_by_id[waypoint.id].location_notes,
            })
            for waypoint in base.waypoints
        ]
        return base.model_copy(update={
            "waypoints": resolved_waypoints,
            "geojson": geojson,
            "route_geojson": base.route_geojson,
            "location_warnings": warnings,
        })

    @staticmethod
    def _coordinate_view(waypoint, resolver: RouteCoordinateResolver, allow_disputed_locations: bool) -> CoordinateAwareWaypointView:
        view = to_waypoint_view_model(waypoint)
        try:
            resolved = resolver.resolve(CandidateRouteAnchor(
                historical_place_id=waypoint.id,
                canonical_name=waypoint.canonical_name,
                evidence_refs=list(waypoint.evidence_refs),
            ))
        except CoordinateResolutionError:
            return CoordinateAwareWaypointView(
                waypoint_id=waypoint.id, name=waypoint.canonical_name,
                location_confidence=LocationConfidence.UNKNOWN,
                location_notes=waypoint.location_notes, evidence_refs=list(waypoint.evidence_refs),
                warning="No resolved display coordinate is available.",
            )
        coordinate = resolved.display_coordinate
        confidence = resolved.confidence
        if confidence is LocationConfidence.UNKNOWN:
            coordinate = None
            warning = resolved.notes or "Location confidence is unknown; geometry withheld."
        elif confidence is LocationConfidence.DISPUTED and not allow_disputed_locations:
            coordinate = None
            warning = resolved.notes or "Location is disputed; geometry withheld."
        elif confidence is LocationConfidence.APPROXIMATE:
            warning = resolved.notes or "Location is approximate."
        elif confidence is LocationConfidence.DISPUTED:
            warning = resolved.notes or "Location is disputed and explicitly allowed."
        elif coordinate is None:
            warning = "No resolved display coordinate is available."
        else:
            warning = None
        return CoordinateAwareWaypointView(
            waypoint_id=view.id, name=view.name, coordinate=coordinate,
            location_confidence=confidence, location_notes=resolved.notes or view.location_notes,
            evidence_refs=list(view.evidence_refs), warning=warning,
        )
