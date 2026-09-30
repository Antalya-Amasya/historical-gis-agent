"""Roman-road-preferred reconstruction over an existing HistoricalRoute.

The input route supplies all historical anchors and their order.  This module
does not resolve places or manufacture historical facts. An explicitly
configured terrain service may reconstruct only an unavailable adjacent leg.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field

from backend.app.gis.coastal_access import find_coastal_accesses, select_simulation_mode, shortest_valid_water_pair
from backend.app.gis.sea import _distance
from backend.app.models import GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, HistoricalTravelMode, PlaceSpatialSemantics

from .barrier_crossings import (
    BarrierCrossingService,
    BarrierCrossingStatus,
    CrossingCandidate,
    is_broad_mountain_constraint,
)
from .geographic import GeographicCandidateRouteService
from .models import ArmyProfile, CandidateRoute, CandidateRouteAnchor, RouteCostBreakdown, RouteMetrics
from backend.app.gis.sea import plan_direct_water_edge
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
    travel_mode: HistoricalTravelMode = HistoricalTravelMode.UNKNOWN
    simulation_route_mode: str = "LAND"
    simulation_segments: list[RomanRoadRouteGeometrySegment] = Field(default_factory=list)


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
        candidate_service: RomanRoadCandidateService | None,
        *,
        terrain_route_service: GeographicCandidateRouteService | None = None,
        terrain_profile: ArmyProfile | None = None,
        maritime_surface=None,
    ) -> None:
        self.candidate_service = candidate_service
        self.terrain_route_service = terrain_route_service
        self.terrain_profile = terrain_profile or ArmyProfile(name="terrain_fallback")
        self.maritime_surface = maritime_surface
        self.barrier_crossing_service = (
            BarrierCrossingService(
                candidate_service,
                terrain_route_service=terrain_route_service,
                terrain_profile=self.terrain_profile,
            ) if candidate_service is not None else None
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
            travel_mode = self._travel_mode(source, destination, historical_route)
            if travel_mode is HistoricalTravelMode.SEA:
                leg = self._maritime_leg(leg_index, source, destination, historical_route)
                legs.append(leg)
                geometry_segments.extend(self._geometry(leg_index, source, destination, leg))
                point_index += 1
                leg_index += 1
                continue
            if is_broad_mountain_constraint(destination):
                if point_index + 2 >= len(points):
                    leg = self._missing_exit_leg(leg_index, source, destination, historical_route)
                    legs.append(leg)
                    geometry_segments.extend(self._geometry(leg_index, source, destination, leg))
                    break
                exit_point = points[point_index + 2]
                if self.barrier_crossing_service is None:
                    leg = self._land_unavailable_leg(leg_index, source, exit_point, historical_route)
                    legs.append(leg)
                    geometry_segments.extend(self._geometry(leg_index, source, exit_point, leg))
                    point_index += 2
                    leg_index += 1
                    continue
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
            leg = (
                self._leg(leg_index, source, destination, self.candidate_service.build(source, destination), historical_route)
                if self.candidate_service is not None
                else self._land_unavailable_leg(leg_index, source, destination, historical_route)
            )
            if leg.candidate is None and self.terrain_route_service is not None:
                leg = self._terrain_fallback(leg, source, destination)
            if leg.travel_mode is HistoricalTravelMode.UNKNOWN and leg.failure_status not in {
                "RIVER_GEOMETRY_UNAVAILABLE", "REGION_GEOMETRY_UNAVAILABLE", "UNKNOWN_PLACE_SEMANTICS", "NON_POINT_PLACE",
            }:
                mixed = self._mixed_simulation(leg_index, source, destination, historical_route, leg)
                if mixed is not None:
                    leg = mixed
            legs.append(leg)
            geometry_segments.extend(self._geometry(leg_index, source, destination, leg))
            point_index += 1
            leg_index += 1
        successes = sum(self._leg_succeeded(leg) for leg in legs)
        status = RomanRoadRouteStatus.COMPLETE if successes == len(legs) else RomanRoadRouteStatus.PARTIAL if successes else RomanRoadRouteStatus.UNAVAILABLE
        return RomanRoadRouteResult(
            historical_route_id=historical_route.id,
            generation_method=(
                "GIS_RECONSTRUCTION_NO_ROAD_NETWORK" if self.candidate_service is None
                else "ROMAN_ROAD_PREFERRED_TERRAIN_FALLBACK" if self.terrain_route_service is not None
                else "ROMAN_ROAD_NETWORK"
            ),
            status=status, legs=legs, geometry_segments=geometry_segments,
            aggregate=self._aggregate(legs), limitations=[
                "HistoricalRoute anchors and order come from supplied evidence-backed route data; Roman-road paths are infrastructure candidates only.",
                "Failed legs remain explicit gaps. Terrain fallback, when configured, is attempted only for the same supplied adjacent anchors; no OSM, straight-line, or cross-leg fallback is used.",
                "Road chronology and certainty are preserved as source metadata and are not converted into historical movement claims.",
                *(
                    ["史料支持两端地点之间的移动。显示路线是根据古代道路、地形和海陆拓扑生成的合理模拟，并非史料记录的精确行程。"]
                    if any(leg.simulation_route_mode == "LAND_SEA_MIXED" for leg in legs) else []
                ),
            ],
        )

    @staticmethod
    def _travel_mode(
        source: HistoricalRoutePoint,
        destination: HistoricalRoutePoint,
        historical_route: HistoricalRoute,
    ) -> HistoricalTravelMode:
        shared_claim_ids = set(source.claim_ids) & set(destination.claim_ids)
        modes = {
            claim.travel_mode
            for claim in historical_route.claims
            if claim.id in shared_claim_ids
        }
        return next(iter(modes)) if len(modes) == 1 else HistoricalTravelMode.UNKNOWN

    def _maritime_leg(self, index, source, destination, historical_route) -> RomanRoadRouteLeg:
        if self.maritime_surface is None:
            return self._maritime_gap_leg(index, source, destination, historical_route)
        plan = plan_direct_water_edge(
            source.historical_place.latitude,
            source.historical_place.longitude,
            destination.historical_place.latitude,
            destination.historical_place.longitude,
            self.maritime_surface,
        )
        if not plan.available and plan.reason == "endpoint_not_ocean_interior":
            coastal = self._explicit_sea_with_coastal_access(index, source, destination, historical_route)
            if coastal is not None:
                return coastal
        if not plan.available:
            status = (
                "MARITIME_PLANNER_UNAVAILABLE"
                if plan.reason == "maritime_surface_unavailable"
                else "MARITIME_GEOMETRY_UNAVAILABLE"
            )
            return RomanRoadRouteLeg(
                leg_index=index,
                source_anchor_id=source.historical_place.id,
                destination_anchor_id=destination.historical_place.id,
                source_evidence_refs=list(source.evidence_refs),
                destination_evidence_refs=list(destination.evidence_refs),
                status=RomanRoadCandidateStatus.DISCONNECTED,
                failure_status=status,
                reconstruction_method=status,
                ordering_provenance=self._ordering_provenance(source, destination, historical_route),
                limitation="; ".join(plan.limitations),
                travel_mode=HistoricalTravelMode.SEA,
                simulation_route_mode="SEA",
            )
        distance_km = plan.physical_distance_m / 1000
        candidate = CandidateRoute(
            id=f"{source.historical_place.id}-{destination.historical_place.id}-direct-water",
            from_anchor=CandidateRouteAnchor.from_historical_point(source),
            to_anchor=CandidateRouteAnchor.from_historical_point(destination),
            geometry=GeoJsonLineString(coordinates=list(plan.coordinates)),
            metrics=RouteMetrics(
                distance_km=distance_km, elevation_gain_m=0, elevation_loss_m=0,
                estimated_cost=plan.physical_distance_m, cell_count=max(1, len(plan.coordinates)),
                segment_count=max(0, len(plan.coordinates) - 1),
            ),
            cost_breakdown=RouteCostBreakdown(
                distance_cost=plan.physical_distance_m, slope_cost=0, terrain_cost=0, barrier_cost=0,
                total_cost=plan.physical_distance_m,
            ),
            confidence=1.0,
            assumptions=[
                *plan.limitations,
                "direct_water_validated=true",
                "detour_used=false",
                "algorithmic_transitions=",
                *[f"{key}={value}" for key, value in plan.provenance],
            ],
            evidence_refs=[],
            provenance="gis_reconstruction",
            coordinate_system="WGS84",
            terrain_source="natural_earth_10m",
            generation_method="DIRECT_WATER_EDGE",
        )
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=source.historical_place.id,
            destination_anchor_id=destination.historical_place.id,
            source_evidence_refs=list(source.evidence_refs),
            destination_evidence_refs=list(destination.evidence_refs),
            status=RomanRoadCandidateStatus.AVAILABLE,
            terrain_candidate=candidate,
            reconstruction_method="DIRECT_WATER_EDGE",
            ordering_provenance=self._ordering_provenance(source, destination, historical_route),
            limitation="; ".join(plan.limitations),
            travel_mode=HistoricalTravelMode.SEA,
            simulation_route_mode="SEA",
        )

    @staticmethod
    def _maritime_gap_leg(index, source, destination, historical_route) -> RomanRoadRouteLeg:
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=source.historical_place.id,
            destination_anchor_id=destination.historical_place.id,
            source_evidence_refs=list(source.evidence_refs),
            destination_evidence_refs=list(destination.evidence_refs),
            status=RomanRoadCandidateStatus.DISCONNECTED,
            failure_status="MARITIME_PLANNER_UNAVAILABLE",
            reconstruction_method="MARITIME_PLANNER_UNAVAILABLE",
            ordering_provenance=RomanRoadRouteOrchestrator._ordering_provenance(source, destination, historical_route),
            limitation="Explicit sea travel is not eligible for Roman-road or land-terrain planning; no maritime planner is configured.",
            travel_mode=HistoricalTravelMode.SEA,
            simulation_route_mode="SEA",
        )

    def _explicit_sea_with_coastal_access(self, index, source, destination, historical_route) -> RomanRoadRouteLeg | None:
        """Keep historical SEA authority while embarkation points stay simulated."""
        pair = shortest_valid_water_pair(
            find_coastal_accesses(source.historical_place.latitude, source.historical_place.longitude, self.maritime_surface),
            find_coastal_accesses(destination.historical_place.latitude, destination.historical_place.longitude, self.maritime_surface),
            self.maritime_surface,
        )
        if pair is None:
            return None
        left, right, plan = pair
        segments = [
            self._coastal_segment(index, source, destination, left),
            self._water_segment(index, source, destination, plan),
            self._coastal_segment(index, source, destination, right),
        ]
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=source.historical_place.id,
            destination_anchor_id=destination.historical_place.id,
            source_evidence_refs=list(source.evidence_refs),
            destination_evidence_refs=list(destination.evidence_refs),
            status=RomanRoadCandidateStatus.AVAILABLE,
            reconstruction_method="DIRECT_WATER_EDGE",
            ordering_provenance=self._ordering_provenance(source, destination, historical_route),
            limitation="Historical sea travel is attested. Coastal access points are simulated and are not historical embarkation sites.",
            travel_mode=HistoricalTravelMode.SEA,
            simulation_route_mode="SEA",
            simulation_segments=segments,
        )

    def _mixed_simulation(self, index, source, destination, historical_route, land_leg: RomanRoadRouteLeg) -> RomanRoadRouteLeg | None:
        if self.maritime_surface is None:
            return None
        pair = shortest_valid_water_pair(
            find_coastal_accesses(source.historical_place.latitude, source.historical_place.longitude, self.maritime_surface),
            find_coastal_accesses(destination.historical_place.latitude, destination.historical_place.longitude, self.maritime_surface),
            self.maritime_surface,
        )
        if pair is None:
            return None
        left, right, plan = pair
        origin_link = self._land_link(source, left.land_longitude, left.land_latitude, "origin")
        destination_link = self._land_link(destination, right.land_longitude, right.land_latitude, "destination")
        if origin_link is None or destination_link is None:
            return None
        mixed_distance = origin_link[1] + plan.physical_distance_m + destination_link[1]
        land_distance = self._land_distance_m(land_leg)
        if select_simulation_mode(historical_mode="UNKNOWN", land_distance_m=land_distance, mixed_distance_m=mixed_distance) != "LAND_SEA_MIXED":
            return None
        segments = [
            *origin_link[0],
            self._coastal_segment(index, source, destination, left),
            self._water_segment(index, source, destination, plan),
            self._coastal_segment(index, source, destination, right),
            *destination_link[0],
        ]
        for item in segments:
            item.leg_index = index
            item.source_anchor_id = source.historical_place.id
            item.destination_anchor_id = destination.historical_place.id
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=source.historical_place.id,
            destination_anchor_id=destination.historical_place.id,
            source_evidence_refs=list(source.evidence_refs),
            destination_evidence_refs=list(destination.evidence_refs),
            status=RomanRoadCandidateStatus.AVAILABLE,
            reconstruction_method="LAND_SEA_MIXED",
            ordering_provenance=self._ordering_provenance(source, destination, historical_route),
            limitation="Simulated land and sea geometry. Coastal access points are not historical waypoints, and the selected mode is not a historical travel claim.",
            travel_mode=HistoricalTravelMode.UNKNOWN,
            simulation_route_mode="LAND_SEA_MIXED",
            simulation_segments=segments,
        )

    def _land_link(self, historical_point: HistoricalRoutePoint, longitude: float, latitude: float, side: str):
        target = self._simulated_access_point(f"sim-coast-{side}-{historical_point.historical_place.id}", longitude, latitude)
        if _distance(
            type("P", (), {"latitude": historical_point.historical_place.latitude, "longitude": historical_point.historical_place.longitude})(),
            type("P", (), {"latitude": latitude, "longitude": longitude})(),
        ) < 1.0:
            return [], 0.0
        if self.candidate_service is not None:
            built = self.candidate_service.build(historical_point, target)
            if built.candidate is not None:
                segments = [
                    RomanRoadRouteGeometrySegment(
                        segment_type=item.segment_type, leg_index=1, coordinates=list(item.coordinates),
                        source_anchor_id=historical_point.historical_place.id, destination_anchor_id=target.historical_place.id,
                    )
                    for item in built.candidate.geometry_segments
                ]
                return segments, built.candidate.network_distance_m + built.candidate.access_connector_distance_m
        if self.terrain_route_service is None:
            return None
        try:
            candidate = self.terrain_route_service.build_between(historical_point, target, self.terrain_profile)
        except (ValueError, KeyError):
            return None
        return [RomanRoadRouteGeometrySegment(
            segment_type="terrain_candidate", leg_index=1, coordinates=list(candidate.geometry.coordinates),
            source_anchor_id=historical_point.historical_place.id, destination_anchor_id=target.historical_place.id,
        )], candidate.metrics.distance_km * 1000

    @staticmethod
    def _simulated_access_point(identifier: str, longitude: float, latitude: float) -> HistoricalRoutePoint:
        return HistoricalRoutePoint(
            sequence=1,
            historical_place=HistoricalPlace(
                id=identifier, canonical_name="simulated coastal access", longitude=longitude, latitude=latitude,
                source="simulated_coastal_access", confidence=0.0, spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
                coordinate_role="representative_point",
            ),
            event_summary="Simulated coastal access point; not a historical waypoint.",
            evidence_refs=[], confidence=0.0, coordinate_role="representative_point",
        )

    @staticmethod
    def _coastal_segment(index, source, destination, access) -> RomanRoadRouteGeometrySegment:
        return RomanRoadRouteGeometrySegment(
            segment_type="simulated_coastal_access", leg_index=index,
            coordinates=[(access.land_longitude, access.land_latitude), (access.ocean_longitude, access.ocean_latitude)],
            source_anchor_id=source.historical_place.id, destination_anchor_id=destination.historical_place.id,
        )

    @staticmethod
    def _water_segment(index, source, destination, plan) -> RomanRoadRouteGeometrySegment:
        return RomanRoadRouteGeometrySegment(
            segment_type="direct_water_edge", leg_index=index, coordinates=list(plan.coordinates),
            source_anchor_id=source.historical_place.id, destination_anchor_id=destination.historical_place.id,
        )

    @staticmethod
    def _land_distance_m(leg: RomanRoadRouteLeg) -> float | None:
        if leg.candidate is not None:
            return leg.candidate.network_distance_m + leg.candidate.access_connector_distance_m
        if leg.terrain_candidate is not None and leg.reconstruction_method != "DIRECT_WATER_EDGE":
            return leg.terrain_candidate.metrics.distance_km * 1000
        return None

    @staticmethod
    def _land_unavailable_leg(index, source, destination, historical_route) -> RomanRoadRouteLeg:
        return RomanRoadRouteLeg(
            leg_index=index,
            source_anchor_id=source.historical_place.id,
            destination_anchor_id=destination.historical_place.id,
            source_evidence_refs=list(source.evidence_refs),
            destination_evidence_refs=list(destination.evidence_refs),
            status=RomanRoadCandidateStatus.DISCONNECTED,
            failure_status="LAND_NETWORK_UNAVAILABLE",
            reconstruction_method="UNAVAILABLE",
            ordering_provenance=RomanRoadRouteOrchestrator._ordering_provenance(source, destination, historical_route),
            limitation="No Roman-road network is configured for this land leg.",
            travel_mode=RomanRoadRouteOrchestrator._travel_mode(source, destination, historical_route),
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
            travel_mode=RomanRoadRouteOrchestrator._travel_mode(source, exit_point, historical_route),
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
            travel_mode=RomanRoadRouteOrchestrator._travel_mode(source, barrier, historical_route),
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
            travel_mode=RomanRoadRouteOrchestrator._travel_mode(barrier, exit_point, historical_route),
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
            travel_mode=RomanRoadRouteOrchestrator._travel_mode(source, destination, historical_route),
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
    def _leg_succeeded(leg: RomanRoadRouteLeg) -> bool:
        return leg.candidate is not None or leg.terrain_candidate is not None or any(item.coordinates for item in leg.simulation_segments)

    @staticmethod
    def _geometry(index: int, source: HistoricalRoutePoint, destination: HistoricalRoutePoint, leg: RomanRoadRouteLeg) -> list[RomanRoadRouteGeometrySegment]:
        if leg.simulation_segments:
            return list(leg.simulation_segments)
        if leg.candidate is None and leg.terrain_candidate is None:
            # Empty coordinates deliberately avoid rendering a fabricated line
            # across an unresolved historical gap.
            return [RomanRoadRouteGeometrySegment(
                segment_type="failed_gap", leg_index=index, coordinates=[], source_anchor_id=source.historical_place.id,
                destination_anchor_id=destination.historical_place.id, failure_status=leg.failure_status,
            )]
        if leg.terrain_candidate is not None:
            return [RomanRoadRouteGeometrySegment(
                segment_type="direct_water_edge" if leg.reconstruction_method == "DIRECT_WATER_EDGE" else "terrain_candidate",
                leg_index=index, coordinates=list(leg.terrain_candidate.geometry.coordinates),
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
            successful_leg_count=sum(RomanRoadRouteOrchestrator._leg_succeeded(leg) for leg in legs),
            failed_leg_count=sum(not RomanRoadRouteOrchestrator._leg_succeeded(leg) for leg in legs),
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
