"""Deterministic, partial Roman-road orchestration over an existing HistoricalRoute.

The input route supplies all historical anchors and their order.  This module
does not call terrain routing, resolve places, or manufacture historical facts.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field

from backend.app.models import HistoricalRoute, HistoricalRoutePoint

from .roman_roads import (
    RoadAccessStatus,
    RomanRoadCandidateResult,
    RomanRoadCandidateRoute,
    RomanRoadCandidateService,
    RomanRoadCandidateStatus,
)


class RomanRoadRouteStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class RomanRoadRouteGeometrySegment(BaseModel):
    """Display-ready, non-continuous geometry roles for a road-route audit."""

    segment_type: str
    leg_index: int
    coordinates: list[tuple[float, float]] = Field(default_factory=list)
    source_anchor_id: str
    destination_anchor_id: str
    failure_status: str | None = None


class RomanRoadRouteLeg(BaseModel):
    leg_index: int = Field(ge=1)
    source_anchor_id: str
    destination_anchor_id: str
    source_evidence_refs: list[str]
    destination_evidence_refs: list[str]
    status: RomanRoadCandidateStatus
    failure_status: str | None = None
    candidate: RomanRoadCandidateRoute | None = None
    limitation: str | None = None


class RomanRoadRouteAggregate(BaseModel):
    successful_leg_count: int = Field(ge=0)
    failed_leg_count: int = Field(ge=0)
    total_network_distance_m: float = Field(ge=0)
    total_access_connector_distance_m: float = Field(ge=0)
    road_type_counts: dict[str, int]
    segment_status_counts: dict[str, int]
    chronology_counts: dict[str, int]


class RomanRoadRouteResult(BaseModel):
    historical_route_id: str
    generation_method: str = "ROMAN_ROAD_NETWORK"
    status: RomanRoadRouteStatus
    legs: list[RomanRoadRouteLeg]
    geometry_segments: list[RomanRoadRouteGeometrySegment]
    aggregate: RomanRoadRouteAggregate
    limitations: list[str]


class RomanRoadRouteOrchestrator:
    """Build road candidates for adjacent supplied anchors, never skipping gaps."""

    def __init__(self, candidate_service: RomanRoadCandidateService) -> None:
        self.candidate_service = candidate_service

    def build_roman_road_candidates(self, historical_route: HistoricalRoute) -> RomanRoadRouteResult:
        points = historical_route.ordered_points
        if len(points) < 2:
            return RomanRoadRouteResult(
                historical_route_id=historical_route.id, status=RomanRoadRouteStatus.UNAVAILABLE,
                legs=[], geometry_segments=[], aggregate=self._aggregate([]),
                limitations=["A Roman-road candidate requires at least two pre-existing evidence-backed historical anchors."],
            )
        legs: list[RomanRoadRouteLeg] = []
        geometry_segments: list[RomanRoadRouteGeometrySegment] = []
        for index, (source, destination) in enumerate(zip(points, points[1:]), start=1):
            result = self.candidate_service.build(source, destination)
            leg = self._leg(index, source, destination, result)
            legs.append(leg)
            geometry_segments.extend(self._geometry(index, source, destination, leg))
        successes = sum(leg.candidate is not None for leg in legs)
        status = RomanRoadRouteStatus.COMPLETE if successes == len(legs) else RomanRoadRouteStatus.PARTIAL if successes else RomanRoadRouteStatus.UNAVAILABLE
        return RomanRoadRouteResult(
            historical_route_id=historical_route.id, status=status, legs=legs, geometry_segments=geometry_segments,
            aggregate=self._aggregate(legs), limitations=[
                "HistoricalRoute anchors and order come from supplied evidence-backed route data; Roman-road paths are infrastructure candidates only.",
                "Failed legs remain explicit gaps. No terrain, OSM, straight-line, or cross-leg fallback was attempted.",
                "Road chronology and certainty are preserved as source metadata and are not converted into historical movement claims.",
            ],
        )

    @staticmethod
    def _leg(index: int, source: HistoricalRoutePoint, destination: HistoricalRoutePoint, result: RomanRoadCandidateResult) -> RomanRoadRouteLeg:
        failure_status = None
        if result.candidate is None:
            if result.status is RomanRoadCandidateStatus.SOURCE_ACCESS_FAILED:
                failure_status = result.source_access.status.value
            elif result.status is RomanRoadCandidateStatus.DESTINATION_ACCESS_FAILED:
                failure_status = result.destination_access.status.value
            else:
                failure_status = result.status.value
        return RomanRoadRouteLeg(
            leg_index=index, source_anchor_id=source.historical_place.id, destination_anchor_id=destination.historical_place.id,
            source_evidence_refs=list(source.evidence_refs), destination_evidence_refs=list(destination.evidence_refs),
            status=result.status, failure_status=failure_status, candidate=result.candidate, limitation=result.limitation,
        )

    @staticmethod
    def _geometry(index: int, source: HistoricalRoutePoint, destination: HistoricalRoutePoint, leg: RomanRoadRouteLeg) -> list[RomanRoadRouteGeometrySegment]:
        if leg.candidate is None:
            # Empty coordinates deliberately avoid rendering a fabricated line
            # across an unresolved historical gap.
            return [RomanRoadRouteGeometrySegment(
                segment_type="failed_gap", leg_index=index, coordinates=[], source_anchor_id=source.historical_place.id,
                destination_anchor_id=destination.historical_place.id, failure_status=leg.failure_status,
            )]
        return [RomanRoadRouteGeometrySegment(
            segment_type=item.segment_type, leg_index=index, coordinates=list(item.coordinates),
            source_anchor_id=source.historical_place.id, destination_anchor_id=destination.historical_place.id,
        ) for item in leg.candidate.geometry_segments]

    @staticmethod
    def _aggregate(legs: list[RomanRoadRouteLeg]) -> RomanRoadRouteAggregate:
        candidates = [leg.candidate for leg in legs if leg.candidate is not None]
        road_types: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        chronology: Counter[str] = Counter()
        for candidate in candidates:
            road_types.update(candidate.road_type_counts)
            statuses.update(candidate.segment_status_counts)
            chronology.update(candidate.chronology_counts)
        return RomanRoadRouteAggregate(
            successful_leg_count=len(candidates), failed_leg_count=len(legs) - len(candidates),
            total_network_distance_m=sum(item.network_distance_m for item in candidates),
            total_access_connector_distance_m=sum(item.access_connector_distance_m for item in candidates),
            road_type_counts=dict(road_types), segment_status_counts=dict(statuses), chronology_counts=dict(chronology),
        )
