"""Offline presentation DTOs for evidence-grounded route and waypoint display."""
from __future__ import annotations

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
    summary: str
    confidence: float = Field(ge=0, le=1)
    score: RouteScore
    explanations: HistoricalRouteExplanations
    location_warnings: list[str] = Field(default_factory=list)


class HistoricalRouteResponse(BaseModel):
    """HTTP-neutral response contract retained as plain data only in this phase."""

    route: HistoricalRoutePresentation
    waypoints: list[HistoricalWaypointViewModel] = Field(default_factory=list)
    geojson: dict[str, object]
    explanations: HistoricalRouteExplanations
    location_warnings: list[str] = Field(default_factory=list)
    knowledge_panels: list[HistoricalKnowledgePanel] = Field(default_factory=list)


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
        geojson = self._geojson(candidate_route, waypoints, evaluation.score)
        return HistoricalRoutePresentation(
            route_id=candidate_route.id,
            route_name=route_name,
            period=period,
            waypoints=waypoints,
            segments=segments,
            geojson=geojson,
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
    def _geojson(
        route: CandidateRoute,
        waypoints: list[HistoricalWaypointViewModel],
        score: RouteScore,
    ) -> dict[str, object]:
        route_feature = {
            "type": "Feature",
            "geometry": route.geometry.model_dump(mode="json"),
            "properties": {
                "route_id": route.id,
                "confidence": route.confidence,
                "total_cost": score.total_cost,
            },
        }
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
        return base.model_copy(update={"geojson": geojson, "location_warnings": warnings})

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
