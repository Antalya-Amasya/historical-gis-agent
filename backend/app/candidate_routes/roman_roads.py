"""Fail-closed Roman-road candidate connections between supplied historical anchors.

This module never creates historical anchors, claims, or HistoricalRoute points.
It only turns two caller-supplied, evidence-grounded points into an auditable
Itiner-e infrastructure candidate when both can safely access the same road
component.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field

from backend.app.models import HistoricalRoutePoint, PlaceSpatialSemantics
from backend.app.roads.itiner_e import Coordinate, RomanRoadEdge, RomanRoadGraph

from .models import CandidateRouteAnchor


class RoadAccessStatus(str, Enum):
    AVAILABLE = "POINT_ACCESS_OK"
    TOO_FAR = "ROAD_ACCESS_TOO_FAR"
    UNAVAILABLE = "ROAD_ACCESS_UNAVAILABLE"
    NON_POINT_PLACE = "NON_POINT_PLACE"
    RIVER_GEOMETRY_UNAVAILABLE = "RIVER_GEOMETRY_UNAVAILABLE"
    REGION_GEOMETRY_UNAVAILABLE = "REGION_GEOMETRY_UNAVAILABLE"
    UNKNOWN_PLACE_SEMANTICS = "UNKNOWN_PLACE_SEMANTICS"


class RomanRoadCandidateStatus(str, Enum):
    AVAILABLE = "ROMAN_ROAD_CANDIDATE_AVAILABLE"
    DISCONNECTED = "ROMAN_ROAD_DISCONNECTED"
    SOURCE_ACCESS_FAILED = "ROMAN_ROAD_SOURCE_ACCESS_FAILED"
    DESTINATION_ACCESS_FAILED = "ROMAN_ROAD_DESTINATION_ACCESS_FAILED"


class RomanRoadAccess(BaseModel):
    """A non-mutating relationship between one historical anchor and one road node."""

    anchor: CandidateRouteAnchor
    historical_anchor_coordinate: tuple[float, float]
    road_node_id: str | None = None
    roman_road_access_coordinate: tuple[float, float] | None = None
    access_distance_m: float | None = Field(default=None, ge=0)
    component_id: str | None = None
    nearest_road_type: str | None = None
    nearest_segment_status: str | None = None
    spatial_semantics: PlaceSpatialSemantics
    coordinate_role: str
    status: RoadAccessStatus
    max_access_distance_m: float = Field(gt=0)
    provenance: str = "itiner_e_nearest_endpoint_access"


class RomanRoadGeometrySegment(BaseModel):
    segment_type: str
    coordinates: list[tuple[float, float]]


class RomanRoadCandidateRoute(BaseModel):
    """Infrastructure candidate, explicitly not a historical movement claim."""

    id: str
    generation_method: str = "ROMAN_ROAD_NETWORK"
    source_access: RomanRoadAccess
    destination_access: RomanRoadAccess
    geometry_segments: list[RomanRoadGeometrySegment]
    network_geometry: list[tuple[float, float]]
    ordered_road_edge_ids: list[str]
    itiner_e_record_ids: list[str]
    network_distance_m: float = Field(ge=0)
    access_connector_distance_m: float = Field(ge=0)
    road_type_counts: dict[str, int]
    segment_status_counts: dict[str, int]
    segment_status_distance_m: dict[str, float]
    chronology_counts: dict[str, int]
    citations: list[str]
    bibliographies: list[str]
    provenance: str = "itiner_e_roman_road_infrastructure_candidate"
    limitations: list[str]


class RomanRoadCandidateResult(BaseModel):
    status: RomanRoadCandidateStatus
    source_access: RomanRoadAccess
    destination_access: RomanRoadAccess
    candidate: RomanRoadCandidateRoute | None = None
    limitation: str | None = None


class RomanRoadCandidateService:
    """Distance-only candidate builder over a prebuilt, immutable RomanRoadGraph."""

    def __init__(self, graph: RomanRoadGraph, *, max_access_distance_m: float = 2_000.0) -> None:
        if max_access_distance_m <= 0:
            raise ValueError("max_access_distance_m must be positive")
        self.graph = graph
        self.max_access_distance_m = max_access_distance_m

    def resolve_access(self, point: HistoricalRoutePoint) -> RomanRoadAccess:
        anchor = CandidateRouteAnchor.from_historical_point(point)
        coordinate = (point.historical_place.longitude, point.historical_place.latitude)
        place = point.historical_place
        blocked = self._semantic_block(anchor, coordinate, place.spatial_semantics, place.coordinate_role)
        if blocked is not None:
            return blocked
        nearest = self.graph.nearest_node(coordinate)
        if nearest is None:
            return RomanRoadAccess(anchor=anchor, historical_anchor_coordinate=coordinate, spatial_semantics=place.spatial_semantics, coordinate_role=place.coordinate_role, status=RoadAccessStatus.UNAVAILABLE, max_access_distance_m=self.max_access_distance_m)
        node, distance_m = nearest
        edge = self._representative_edge(node.id)
        status = RoadAccessStatus.AVAILABLE if distance_m <= self.max_access_distance_m else RoadAccessStatus.TOO_FAR
        return RomanRoadAccess(
            anchor=anchor, historical_anchor_coordinate=coordinate,
            road_node_id=node.id, roman_road_access_coordinate=node.coordinate,
            access_distance_m=distance_m, component_id=self.graph.component_id(node.id),
            nearest_road_type=edge.segment.road_type if edge else None,
            nearest_segment_status=edge.segment.segment_status if edge else None,
            spatial_semantics=place.spatial_semantics, coordinate_role=place.coordinate_role,
            status=status, max_access_distance_m=self.max_access_distance_m,
        )

    def _semantic_block(self, anchor: CandidateRouteAnchor, coordinate: Coordinate, semantics: PlaceSpatialSemantics, coordinate_role: str) -> RomanRoadAccess | None:
        if semantics is PlaceSpatialSemantics.RIVER:
            status = RoadAccessStatus.RIVER_GEOMETRY_UNAVAILABLE
        elif semantics is PlaceSpatialSemantics.MOUNTAIN_REGION:
            status = RoadAccessStatus.REGION_GEOMETRY_UNAVAILABLE
        elif semantics is PlaceSpatialSemantics.UNKNOWN:
            status = RoadAccessStatus.UNKNOWN_PLACE_SEMANTICS
        elif coordinate_role != "exact_site":
            status = RoadAccessStatus.NON_POINT_PLACE
        else:
            return None
        return RomanRoadAccess(anchor=anchor, historical_anchor_coordinate=coordinate, spatial_semantics=semantics, coordinate_role=coordinate_role, status=status, max_access_distance_m=self.max_access_distance_m, provenance="place_semantics_fail_closed")

    def build(self, source: HistoricalRoutePoint, destination: HistoricalRoutePoint) -> RomanRoadCandidateResult:
        source_access = self.resolve_access(source)
        destination_access = self.resolve_access(destination)
        if source_access.status is not RoadAccessStatus.AVAILABLE:
            return RomanRoadCandidateResult(status=RomanRoadCandidateStatus.SOURCE_ACCESS_FAILED, source_access=source_access, destination_access=destination_access, limitation="Source historical anchor is not safely accessible from the configured Roman-road network threshold.")
        if destination_access.status is not RoadAccessStatus.AVAILABLE:
            return RomanRoadCandidateResult(status=RomanRoadCandidateStatus.DESTINATION_ACCESS_FAILED, source_access=source_access, destination_access=destination_access, limitation="Destination historical anchor is not safely accessible from the configured Roman-road network threshold.")
        assert source_access.road_node_id and destination_access.road_node_id
        if source_access.component_id != destination_access.component_id:
            return RomanRoadCandidateResult(status=RomanRoadCandidateStatus.DISCONNECTED, source_access=source_access, destination_access=destination_access, limitation="Roman-road access nodes are in disconnected topology components; no off-road or terrain fallback was attempted.")
        path = self.graph.shortest_path(source_access.road_node_id, destination_access.road_node_id)
        if path is None:  # defensive: component ids and path must agree
            return RomanRoadCandidateResult(status=RomanRoadCandidateStatus.DISCONNECTED, source_access=source_access, destination_access=destination_access, limitation="Roman-road topology did not produce a path; no fallback was attempted.")
        candidate = self._candidate(source_access, destination_access, path.edge_ids, path.geometry, path.distance_m)
        return RomanRoadCandidateResult(status=RomanRoadCandidateStatus.AVAILABLE, source_access=source_access, destination_access=destination_access, candidate=candidate)

    def _representative_edge(self, node_id: str) -> RomanRoadEdge | None:
        edge_ids = self.graph.adjacency.get(node_id, ())
        return self.graph.edges[edge_ids[0]] if edge_ids else None

    def _candidate(self, source: RomanRoadAccess, destination: RomanRoadAccess, edge_ids: tuple[str, ...], network_geometry: tuple[Coordinate, ...], network_distance_m: float) -> RomanRoadCandidateRoute:
        edges = [self.graph.edges[edge_id] for edge_id in edge_ids]
        status_counts = Counter((edge.segment.segment_status or "Unknown") for edge in edges)
        status_distance: dict[str, float] = {}
        for edge in edges:
            status = edge.segment.segment_status or "Unknown"
            status_distance[status] = status_distance.get(status, 0.0) + edge.length_m
        chronology_counts = Counter(edge.segment.chronology.status for edge in edges)
        geometry_segments: list[RomanRoadGeometrySegment] = []
        if source.access_distance_m:
            geometry_segments.append(RomanRoadGeometrySegment(segment_type="access_connector", coordinates=[source.historical_anchor_coordinate, source.roman_road_access_coordinate]))
        geometry_segments.append(RomanRoadGeometrySegment(segment_type="roman_road", coordinates=list(network_geometry)))
        if destination.access_distance_m:
            geometry_segments.append(RomanRoadGeometrySegment(segment_type="access_connector", coordinates=[destination.roman_road_access_coordinate, destination.historical_anchor_coordinate]))
        return RomanRoadCandidateRoute(
            id=f"roman-road-{source.anchor.historical_place_id}-to-{destination.anchor.historical_place_id}",
            source_access=source, destination_access=destination, geometry_segments=geometry_segments,
            network_geometry=list(network_geometry), ordered_road_edge_ids=list(edge_ids),
            itiner_e_record_ids=[edge.segment.record_id for edge in edges], network_distance_m=network_distance_m,
            access_connector_distance_m=(source.access_distance_m or 0) + (destination.access_distance_m or 0),
            road_type_counts=dict(Counter(edge.segment.road_type or "Unknown" for edge in edges)),
            segment_status_counts=dict(status_counts), segment_status_distance_m=status_distance,
            chronology_counts=dict(chronology_counts),
            citations=list(dict.fromkeys(edge.segment.citation for edge in edges if edge.segment.citation)),
            bibliographies=list(dict.fromkeys(edge.segment.bibliography for edge in edges if edge.segment.bibliography)),
            limitations=[
                "This is an Itiner-e Roman-road infrastructure candidate, not evidence that the historical actor used these roads.",
                "Access connectors are geometric access relationships and are not represented as Roman roads or historical movement evidence.",
                "Unknown chronology is retained; road availability and use during the event are not inferred.",
            ],
        )
