"""Presentation-only projection of Roman-road orchestration results."""
from __future__ import annotations

from pydantic import BaseModel

from backend.app.models import HistoricalRoute

from .roman_road_orchestration import RomanRoadRouteResult


class RomanRoadPresentation(BaseModel):
    route: dict[str, object]
    geojson: dict[str, object]
    route_geojson: None = None
    road_network: dict[str, object]
    knowledge_panels: list[dict[str, object]]
    presentation_summary: dict[str, object]


class RomanRoadPresentationService:
    """Creates role-labelled map data without altering route computation."""

    def present(self, historical_route: HistoricalRoute, result: RomanRoadRouteResult) -> RomanRoadPresentation:
        anchor_features = [self._anchor_feature(point) for point in historical_route.ordered_points]
        segment_features = []
        for segment in result.geometry_segments:
            geometry = {"type": "LineString", "coordinates": [list(item) for item in segment.coordinates]} if segment.coordinates else None
            segment_features.append({"type": "Feature", "geometry": geometry, "properties": {
                "layer_type": "roman_road_segment", "segment_role": segment.segment_type,
                "leg_index": segment.leg_index, "source_anchor_id": segment.source_anchor_id,
                "destination_anchor_id": segment.destination_anchor_id, "failure_status": segment.failure_status,
            }})
        road_network = {
            "source": "Itiner-e — The Digital Atlas of Ancient Roads",
            "route_status": result.status.value,
            "aggregate": result.aggregate.model_dump(mode="json"),
            "legs": [leg.model_dump(mode="json") for leg in result.legs],
            "limitations": list(result.limitations),
        }
        return RomanRoadPresentation(
            route={"route_id": historical_route.id, "route_name": historical_route.name, "period": historical_route.period, "confidence": historical_route.historical_confidence, "generation_method": result.generation_method, "route_status": result.status.value},
            geojson={"type": "FeatureCollection", "features": [*anchor_features, *segment_features]},
            road_network=road_network, knowledge_panels=[self._anchor_panel(point) for point in historical_route.ordered_points],
            presentation_summary={
                "title": historical_route.name, "route_method": "ROMAN_ROAD_NETWORK", "route_status": result.status.value,
                "route_interpretation": "Partial Roman-road candidate reconstruction where available; it is not proof of an exact historical track.",
                "limitations": list(result.limitations),
            },
        )

    @staticmethod
    def _anchor_feature(point) -> dict[str, object]:
        place = point.historical_place
        return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [place.longitude, place.latitude]}, "properties": {
            "layer_type": "historical_anchor", "waypoint_id": place.id, "knowledge_panel_id": place.id,
            "name": place.canonical_name, "modern_name": place.modern_name, "coordinate_role": place.coordinate_role,
            "spatial_semantics": place.spatial_semantics.value, "evidence_refs": list(point.evidence_refs),
            "confidence": point.confidence, "uncertain": place.uncertain, "source": place.source,
            "source_id": place.source_id, "source_url": place.source_url,
        }}

    @staticmethod
    def _anchor_panel(point) -> dict[str, object]:
        place = point.historical_place
        return {"waypoint_id": place.id, "title": place.canonical_name, "period": point.date_or_period,
                "event_type": None, "summary": point.event_summary, "evidence_refs": list(point.evidence_refs),
                "source_references": [place.source], "external_references": [], "confidence": "approximate" if place.uncertain else "exact"}
