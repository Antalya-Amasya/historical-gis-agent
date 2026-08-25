"""Read-only catalog for already-supplied historical route presentation data."""
from __future__ import annotations

from backend.app.candidate_routes.annotation import HistoricalEventType, HistoricalExternalReference, HistoricalExternalReferenceType
from backend.app.candidate_routes.knowledge_panel import HistoricalKnowledgePanel
from backend.app.candidate_routes.location import LocationConfidence
from backend.app.candidate_routes.models import RouteScore
from backend.app.candidate_routes.presentation import HistoricalRouteExplanations, HistoricalRoutePresentation, HistoricalRouteResponse
from backend.app.candidate_routes.view_model import HistoricalWaypointViewModel


class PresentationNotFoundError(KeyError):
    pass


class PresentationContractError(ValueError):
    pass


class HistoricalRoutePresentationCatalog:
    """Returns preconstructed presentation records; it performs no retrieval or route computation."""

    def __init__(self) -> None:
        self._responses = {"phase10-demo-route": self._phase10_demo()}

    def get(self, route_id: str) -> HistoricalRouteResponse:
        try:
            return self._responses[route_id]
        except KeyError as exc:
            raise PresentationNotFoundError(route_id) from exc

    @staticmethod
    def _phase10_demo() -> HistoricalRouteResponse:
        waypoints = [
            HistoricalWaypointViewModel(id="carthago-nova", name="Carthago Nova", event_type=HistoricalEventType.CITY, period="218 BCE", description="Supplied presentation metadata.", location_confidence=LocationConfidence.EXACT, evidence_refs=["polybius_iii"], source_references=["polybius_iii"]),
            HistoricalWaypointViewModel(id="rhodanus", name="Rhodanus", event_type=HistoricalEventType.CROSSING, period="218 BCE", description="Supplied presentation metadata.", location_confidence=LocationConfidence.APPROXIMATE, evidence_refs=["polybius_iii", "livy_xxi"], source_references=["polybius_iii", "livy_xxi"]),
            HistoricalWaypointViewModel(id="unresolved", name="Unresolved waypoint", event_type=HistoricalEventType.OTHER, period="218 BCE", description="No trusted display coordinate is supplied.", location_confidence=LocationConfidence.UNKNOWN, evidence_refs=["polybius_iii"], source_references=["polybius_iii"]),
        ]
        panels = [
            HistoricalKnowledgePanel(waypoint_id="carthago-nova", title="Carthago Nova", period="218 BCE", event_type=HistoricalEventType.CITY, summary="Explicitly supplied local summary.", evidence_refs=["polybius_iii"], source_references=["polybius_iii"], external_references=[HistoricalExternalReference(id="reading", reference_type=HistoricalExternalReferenceType.ENCYCLOPEDIA, title="Reference reading", url="https://example.invalid/reading")], confidence=LocationConfidence.EXACT),
            HistoricalKnowledgePanel(waypoint_id="rhodanus", title="Rhodanus", period="218 BCE", event_type=HistoricalEventType.CROSSING, summary=None, evidence_refs=["polybius_iii", "livy_xxi"], source_references=["polybius_iii", "livy_xxi"], external_references=[HistoricalExternalReference(id="modern", reference_type=HistoricalExternalReferenceType.GOOGLE_MAPS, title="Modern location link", url="https://example.invalid/modern")], confidence=LocationConfidence.APPROXIMATE),
            HistoricalKnowledgePanel(waypoint_id="unresolved", title="Unresolved waypoint", period="218 BCE", event_type=HistoricalEventType.OTHER, summary=None, evidence_refs=["polybius_iii"], source_references=["polybius_iii"], confidence=LocationConfidence.UNKNOWN),
        ]
        geojson = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[-0.4, 39.99], [4.83, 45.76], [9.19, 45.47]]}, "properties": {"route_id": "phase10-demo-route", "total_cost": 42.0, "confidence": 0.72}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-0.4, 39.99]}, "properties": {"waypoint_id": "carthago-nova", "knowledge_panel_id": "carthago-nova"}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [4.83, 45.76]}, "properties": {"waypoint_id": "rhodanus", "knowledge_panel_id": "rhodanus"}},
            {"type": "Feature", "geometry": None, "properties": {"waypoint_id": "unresolved", "knowledge_panel_id": "unresolved"}},
        ]}
        explanations = HistoricalRouteExplanations(distance_reason="The route contains 3 supplied historical waypoint(s).", terrain_reason="No terrain cost is present in the supplied route evaluation.", historical_reason="The graph retains supplied evidence references.")
        route = HistoricalRoutePresentation(route_id="phase10-demo-route", route_name="Evidence-backed presentation demo", period="218 BCE", waypoints=waypoints, segments=[], geojson=geojson, summary="Route connects supplied historical waypoints.", confidence=0.72, score=RouteScore(profile_name="baseline", distance_cost=42.0, terrain_cost=0.0, historical_cost=0.0, total_cost=42.0), explanations=explanations)
        return HistoricalRouteResponse(route=route, waypoints=waypoints, geojson=geojson, explanations=explanations, knowledge_panels=panels)


class HistoricalRoutePresentationReadService:
    def __init__(self, catalog: HistoricalRoutePresentationCatalog | None = None) -> None:
        self.catalog = catalog or HistoricalRoutePresentationCatalog()

    def get_presentation(self, route_id: str) -> HistoricalRouteResponse:
        try:
            return self.catalog.get(route_id)
        except PresentationNotFoundError:
            raise
        except (TypeError, ValueError) as exc:
            raise PresentationContractError("Stored presentation does not satisfy the response contract") from exc
