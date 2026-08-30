"""Roman-road-preferred reconstruction over an existing HistoricalRoute.

The input route supplies all historical anchors and their order.  This module
does not resolve places or manufacture historical facts. An explicitly
configured terrain service may reconstruct only an unavailable adjacent leg.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field

from backend.app.models import GeoJsonLineString, HistoricalRoute, HistoricalRoutePoint

from .barrier_crossings import (
    BarrierCrossingService,
    BarrierCrossingStatus,
    CrossingCandidate,
    is_broad_mountain_constraint,
)
from .geographic import GeographicCandidateRouteService
from .models import ArmyProfile, CandidateRoute
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
    terrain_candidate: CandidateRoute | None = None
    crossing_candidate: CrossingCandidate | None = None
    barrier_anchor_id: str | None = None
    barrier_evidence_refs: list[str] = Field(default_factory=list)
    reconstruction_method: str = "UNAVAILABLE"
    ordering_provenance: list[dict[str, object]] = Field(default_factory=list)
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
    """Build adjacent candidates in supplied order, preferring Roman roads."""

    def __init__(
        self,
        candidate_service: RomanRoadCandidateService,
        *,
        terrain_route_service: GeographicCandidateRouteService | None = None,
        terrain_profile: ArmyProfile | None = None,
    ) -> None:
        self.candidate_service = candidate_service
        self.terrain_route_service = terrain_route_service
        self.terrain_profile = terrain_profile or ArmyProfile(name="terrain_fallback")
        self.barrier_crossing_service = BarrierCrossingService(
            candidate_service,
            terrain_route_service=terrain_route_service,
            terrain_profile=self.terrain_profile,
        )

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
        point_index = 0
        leg_index = 1
        while point_index < len(points) - 1:
            source, destination = points[point_index], points[point_index + 1]
            if is_broad_mountain_constraint(destination):
                if point_index + 2 >= len(points):
                    leg = self._missing_exit_leg(leg_index, source, destination, historical_route)
                    legs.append(leg)
                    geometry_segments.extend(self._geometry(leg_index, source, destination, leg))
                    break
                exit_point = points[point_index + 2]
                crossing = self.barrier_crossing_service.build(source, destination, exit_point)
                leg = self._barrier_leg(leg_index, source, destination, exit_point, crossing, historical_route)
                legs.append(leg)
                geometry_segments.extend(self._geometry(leg_index, source, exit_point, leg))
                point_index += 2
                leg_index += 1
                continue
            if is_broad_mountain_constraint(source):
                leg = self._missing_approach_leg(leg_index, source, destination, historical_route)
                legs.append(leg)
                geometry_segments.extend(self._geometry(leg_index, source, destination, leg))
                point_index += 1
                leg_index += 1
                continue
            result = self.candidate_service.build(source, destination)
            leg = self._leg(leg_index, source, destination, result, historical_route)
            if leg.candidate is None and self.terrain_route_service is not None:
                leg = self._terrain_fallback(leg, source, destination)
            legs.append(leg)
            geometry_segments.extend(self._geometry(leg_index, source, destination, leg))
            point_index += 1
            leg_index += 1
        successes = sum(leg.candidate is not None or leg.terrain_candidate is not None for leg in legs)
        status = RomanRoadRouteStatus.COMPLETE if successes == len(legs) else RomanRoadRouteStatus.PARTIAL if successes else RomanRoadRouteStatus.UNAVAILABLE
        return RomanRoadRouteResult(
            historical_route_id=historical_route.id,
            generation_method="ROMAN_ROAD_PREFERRED_TERRAIN_FALLBACK" if self.terrain_route_service is not None else "ROMAN_ROAD_NETWORK",
            status=status, legs=legs, geometry_segments=geometry_segments,
            aggregate=self._aggregate(legs), limitations=[
                "HistoricalRoute anchors and order come from supplied evidence-backed route data; Roman-road paths are infrastructure candidates only.",
                "Failed legs remain explicit gaps. Terrain fallback, when configured, is attempted only for the same supplied adjacent anchors; no OSM, straight-line, or cross-leg fallback is used.",
                "Road chronology and certainty are preserved as source metadata and are not converted into historical movement claims.",
            ],
        )

    @staticmethod
    def _barrier_leg(index, source, barrier, exit_point, result, historical_route) -> RomanRoadRouteLeg:
        available = result.status is BarrierCrossingStatus.AVAILABLE
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=source.historical_place.id,
            destination_anchor_id=exit_point.historical_place.id,
            source_evidence_refs=list(source.evidence_refs),
            destination_evidence_refs=list(exit_point.evidence_refs),
            status=RomanRoadCandidateStatus.AVAILABLE if available else RomanRoadCandidateStatus.DISCONNECTED,
            failure_status=None if available else result.failure_reason,
            candidate=result.road_candidate,
            terrain_candidate=result.terrain_candidate,
            crossing_candidate=result.crossing,
            barrier_anchor_id=barrier.historical_place.id,
            barrier_evidence_refs=list(barrier.evidence_refs),
            reconstruction_method=(
                "ANCIENT_ROAD_BARRIER_CROSSING" if result.road_candidate is not None
                else "TERRAIN_BARRIER_CROSSING" if result.terrain_candidate is not None
                else "UNAVAILABLE"
            ),
            ordering_provenance=[
                *RomanRoadRouteOrchestrator._ordering_provenance(source, barrier, historical_route),
                *RomanRoadRouteOrchestrator._ordering_provenance(barrier, exit_point, historical_route),
            ],
            limitation=(
                "Broad mountain-region coordinate is a search reference only; the selected crossing is an algorithmic GIS artifact."
                if available else "No defensible crossing was generated; the historical mountain constraint remains an explicit gap."
            ),
        )

    @staticmethod
    def _missing_exit_leg(index, source, barrier, historical_route) -> RomanRoadRouteLeg:
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=source.historical_place.id,
            destination_anchor_id=barrier.historical_place.id,
            source_evidence_refs=list(source.evidence_refs),
            destination_evidence_refs=list(barrier.evidence_refs),
            status=RomanRoadCandidateStatus.DESTINATION_ACCESS_FAILED,
            failure_status="BARRIER_EXIT_CONSTRAINT_UNAVAILABLE",
            barrier_anchor_id=barrier.historical_place.id,
            barrier_evidence_refs=list(barrier.evidence_refs),
            ordering_provenance=RomanRoadRouteOrchestrator._ordering_provenance(source, barrier, historical_route),
            limitation="Broad mountain constraint has no trusted onward waypoint; its representative coordinate is not used as an exact route destination.",
        )

    @staticmethod
    def _missing_approach_leg(index, barrier, exit_point, historical_route) -> RomanRoadRouteLeg:
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=barrier.historical_place.id,
            destination_anchor_id=exit_point.historical_place.id,
            source_evidence_refs=list(barrier.evidence_refs),
            destination_evidence_refs=list(exit_point.evidence_refs),
            status=RomanRoadCandidateStatus.SOURCE_ACCESS_FAILED,
            failure_status="BARRIER_APPROACH_CONSTRAINT_UNAVAILABLE",
            barrier_anchor_id=barrier.historical_place.id,
            barrier_evidence_refs=list(barrier.evidence_refs),
            ordering_provenance=RomanRoadRouteOrchestrator._ordering_provenance(barrier, exit_point, historical_route),
            limitation="Broad mountain constraint has no trusted approach waypoint; its representative coordinate is not used as an exact route origin.",
        )

    @staticmethod
    def _leg(index: int, source: HistoricalRoutePoint, destination: HistoricalRoutePoint, result: RomanRoadCandidateResult, historical_route: HistoricalRoute) -> RomanRoadRouteLeg:
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
            status=result.status, failure_status=failure_status, candidate=result.candidate,
            reconstruction_method="ROMAN_ROAD_NETWORK" if result.candidate is not None else "UNAVAILABLE",
            ordering_provenance=RomanRoadRouteOrchestrator._ordering_provenance(source, destination, historical_route),
            limitation=result.limitation,
        )

    def _terrain_fallback(self, leg: RomanRoadRouteLeg, source: HistoricalRoutePoint, destination: HistoricalRoutePoint) -> RomanRoadRouteLeg:
        assert self.terrain_route_service is not None
        try:
            candidate = self.terrain_route_service.build_between(source, destination, self.terrain_profile)
        except (ValueError, KeyError) as exc:
            return leg.model_copy(update={
                "failure_status": "TERRAIN_RECONSTRUCTION_FAILED",
                "limitation": f"{leg.limitation or 'Roman-road candidate unavailable.'} Terrain fallback failed: {type(exc).__name__}.",
            })
        coordinates = list(candidate.geometry.coordinates)
        coordinates[0] = (source.historical_place.longitude, source.historical_place.latitude)
        coordinates[-1] = (destination.historical_place.longitude, destination.historical_place.latitude)
        return leg.model_copy(update={
            "terrain_candidate": candidate.model_copy(update={"geometry": GeoJsonLineString(coordinates=coordinates)}),
            "reconstruction_method": "TERRAIN_ASTAR_FALLBACK",
            "limitation": f"{leg.limitation or 'Roman-road candidate unavailable.'} Terrain A* supplied an algorithmic fallback for this same waypoint pair.",
        })

    @staticmethod
    def _geometry(index: int, source: HistoricalRoutePoint, destination: HistoricalRoutePoint, leg: RomanRoadRouteLeg) -> list[RomanRoadRouteGeometrySegment]:
        if leg.candidate is None and leg.terrain_candidate is None:
            # Empty coordinates deliberately avoid rendering a fabricated line
            # across an unresolved historical gap.
            return [RomanRoadRouteGeometrySegment(
                segment_type="failed_gap", leg_index=index, coordinates=[], source_anchor_id=source.historical_place.id,
                destination_anchor_id=destination.historical_place.id, failure_status=leg.failure_status,
            )]
        if leg.terrain_candidate is not None:
            return [RomanRoadRouteGeometrySegment(
                segment_type="terrain_candidate", leg_index=index, coordinates=list(leg.terrain_candidate.geometry.coordinates),
                source_anchor_id=source.historical_place.id, destination_anchor_id=destination.historical_place.id,
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
            successful_leg_count=sum(leg.candidate is not None or leg.terrain_candidate is not None for leg in legs),
            failed_leg_count=sum(leg.candidate is None and leg.terrain_candidate is None for leg in legs),
            total_network_distance_m=sum(item.network_distance_m for item in candidates),
            total_access_connector_distance_m=sum(item.access_connector_distance_m for item in candidates),
            road_type_counts=dict(road_types), segment_status_counts=dict(statuses), chronology_counts=dict(chronology),
        )

    @staticmethod
    def _ordering_provenance(source: HistoricalRoutePoint, destination: HistoricalRoutePoint, historical_route: HistoricalRoute) -> list[dict[str, object]]:
        claim_ids = set(source.claim_ids) & set(destination.claim_ids)
        return [
            {
                "claim_id": claim.id,
                "claim_type": claim.claim_type,
                "movement_relation": claim.movement_relation,
                "sequence_status": claim.sequence_status,
            }
            for claim in historical_route.claims
            if claim.id in claim_ids
        ]
