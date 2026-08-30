"""Algorithmic mountain-barrier crossings between ordered historical constraints.

The barrier reference point narrows a search over an existing road or terrain
path. It is never used as a historical destination or promoted to evidence.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.app.models import HistoricalRoutePoint, PlaceSpatialSemantics
from backend.app.roads.itiner_e import Coordinate, haversine_m

from .geographic import GeographicCandidateRouteService, LocalProjection
from .models import ArmyProfile, CandidateRoute
from .roman_roads import RomanRoadCandidateRoute, RomanRoadCandidateService
from .terrain import TerrainDataUnavailableError


class CrossingCandidateSource(str, Enum):
    ANCIENT_ROAD_BARRIER_CANDIDATE = "ANCIENT_ROAD_BARRIER_CANDIDATE"
    TERRAIN_DERIVED_CROSSING = "TERRAIN_DERIVED_CROSSING"


class CrossingCandidate(BaseModel):
    """A ranked GIS artifact, deliberately separate from HistoricalRoutePoint."""

    coordinate: tuple[float, float]
    barrier_id: str
    barrier_name: str
    approach_anchor_id: str
    exit_anchor_id: str
    source: CrossingCandidateSource
    road_support: bool
    terrain_support: bool
    reconstruction_cost: float = Field(ge=0)
    distance_to_barrier_reference_km: float = Field(ge=0)
    elevation_m: float | None = None
    road_edge_ids: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    historical_evidence_refs: list[str] = Field(default_factory=list)
    authority: str = "algorithmic_gis_candidate"


class BarrierCrossingStatus(str, Enum):
    AVAILABLE = "BARRIER_CROSSING_AVAILABLE"
    UNAVAILABLE = "BARRIER_CROSSING_UNAVAILABLE"


class BarrierCrossingResult(BaseModel):
    status: BarrierCrossingStatus
    crossing: CrossingCandidate | None = None
    road_candidate: RomanRoadCandidateRoute | None = None
    terrain_candidate: CandidateRoute | None = None
    failure_reason: str | None = None


def is_broad_mountain_constraint(point: HistoricalRoutePoint) -> bool:
    place = point.historical_place
    return (
        place.spatial_semantics is PlaceSpatialSemantics.MOUNTAIN_REGION
        and place.coordinate_role in {"regional_centroid", "representative_point"}
        and bool(place.spatial_semantics_provenance)
    )


def barrier_reference_is_between(
    approach: HistoricalRoutePoint,
    barrier: HistoricalRoutePoint,
    exit_point: HistoricalRoutePoint,
) -> bool:
    reference = (barrier.historical_place.longitude, barrier.historical_place.latitude)
    projection = LocalProjection(*reference)
    approach_xy = projection.to_local(
        approach.historical_place.longitude, approach.historical_place.latitude
    )
    exit_xy = projection.to_local(
        exit_point.historical_place.longitude, exit_point.historical_place.latitude
    )
    direction = (exit_xy[0] - approach_xy[0], exit_xy[1] - approach_xy[1])
    denominator = direction[0] ** 2 + direction[1] ** 2
    if denominator == 0:
        return False
    reference_fraction = (
        (-approach_xy[0]) * direction[0] + (-approach_xy[1]) * direction[1]
    ) / denominator
    return 0.0 <= reference_fraction <= 1.0


def select_crossing_candidate(
    coordinates: list[Coordinate],
    approach: HistoricalRoutePoint,
    barrier: HistoricalRoutePoint,
    exit_point: HistoricalRoutePoint,
    *,
    source: CrossingCandidateSource,
    road_edge_ids: list[str] | None = None,
    elevation_provider=None,
    search_radius_m: float = 250_000.0,
    elevation_weight: float = 0.25,
) -> CrossingCandidate | None:
    """Rank coordinates on one constrained path; never creates route anchors."""
    if not coordinates:
        return None
    reference = (barrier.historical_place.longitude, barrier.historical_place.latitude)
    if not barrier_reference_is_between(approach, barrier, exit_point):
        return None
    candidates = coordinates[1:-1] if len(coordinates) > 2 else coordinates
    scored: list[tuple[float, float, Coordinate, float | None]] = []
    for coordinate in candidates:
        distance_m = haversine_m(coordinate, reference)
        if distance_m > search_radius_m:
            continue
        try:
            elevation = elevation_provider(coordinate) if elevation_provider is not None else None
        except (TerrainDataUnavailableError, ValueError, KeyError):
            elevation = None
        terrain_penalty = max(elevation or 0.0, 0.0) / 1_000.0 * elevation_weight
        scored.append((distance_m / 1_000.0 + terrain_penalty, distance_m, coordinate, elevation))
    if not scored:
        return None
    cost, distance_m, coordinate, elevation = min(scored, key=lambda item: (item[0], item[1], item[2]))
    road_supported = source is CrossingCandidateSource.ANCIENT_ROAD_BARRIER_CANDIDATE
    return CrossingCandidate(
        coordinate=coordinate,
        barrier_id=barrier.historical_place.id,
        barrier_name=barrier.historical_place.canonical_name,
        approach_anchor_id=approach.historical_place.id,
        exit_anchor_id=exit_point.historical_place.id,
        source=source,
        road_support=road_supported,
        terrain_support=elevation is not None or not road_supported,
        reconstruction_cost=cost,
        distance_to_barrier_reference_km=distance_m / 1_000.0,
        elevation_m=elevation,
        road_edge_ids=list(road_edge_ids or []),
        provenance=[
            "HistoricalRoute supplied approach, mountain constraint, exit, and order.",
            "Itiner-e ancient/Roman transport-network prior constrained the path."
            if road_supported else
            "Offline terrain A* supplied the path after road reconstruction was unavailable.",
            "The audited regional representative point was used only as a bounded search reference, not as a destination.",
        ],
        limitations=[
            "This crossing is algorithmically inferred and is not directly attested by historical evidence.",
            "No authoritative mountain boundary, ridge, or watershed geometry is loaded.",
            "Ancient-road infrastructure does not prove contemporaneous use by the historical actor."
            if road_supported else
            "Terrain plausibility does not establish historical route usage.",
        ],
        historical_evidence_refs=list(dict.fromkeys(
            [*approach.evidence_refs, *barrier.evidence_refs, *exit_point.evidence_refs]
        )),
    )


class BarrierCrossingService:
    """Selects a path-supported crossing near an audited broad mountain region."""

    def __init__(
        self,
        road_service: RomanRoadCandidateService,
        *,
        terrain_route_service: GeographicCandidateRouteService | None = None,
        terrain_profile: ArmyProfile | None = None,
        search_radius_m: float = 250_000.0,
        elevation_weight: float = 0.25,
    ) -> None:
        if search_radius_m <= 0 or elevation_weight < 0:
            raise ValueError("barrier crossing search parameters must be non-negative")
        self.road_service = road_service
        self.terrain_route_service = terrain_route_service
        self.terrain_profile = terrain_profile or ArmyProfile(name="barrier_crossing")
        self.search_radius_m = search_radius_m
        self.elevation_weight = elevation_weight

    def build(
        self,
        approach: HistoricalRoutePoint,
        barrier: HistoricalRoutePoint,
        exit_point: HistoricalRoutePoint,
    ) -> BarrierCrossingResult:
        if not is_broad_mountain_constraint(barrier):
            return BarrierCrossingResult(
                status=BarrierCrossingStatus.UNAVAILABLE,
                failure_reason="PLACE_IS_NOT_AUDITED_BROAD_MOUNTAIN_CONSTRAINT",
            )

        road_result = self.road_service.build(approach, exit_point)
        if road_result.candidate is not None:
            crossing = self._select(
                road_result.candidate.network_geometry,
                approach,
                barrier,
                exit_point,
                source=CrossingCandidateSource.ANCIENT_ROAD_BARRIER_CANDIDATE,
                road_edge_ids=road_result.candidate.ordered_road_edge_ids,
            )
            if crossing is not None:
                return BarrierCrossingResult(
                    status=BarrierCrossingStatus.AVAILABLE,
                    crossing=crossing,
                    road_candidate=road_result.candidate,
                )

        if self.terrain_route_service is not None:
            try:
                terrain_candidate = self.terrain_route_service.build_between(
                    approach, exit_point, self.terrain_profile,
                )
            except (KeyError, ValueError):
                terrain_candidate = None
            if terrain_candidate is not None:
                crossing = self._select(
                    terrain_candidate.geometry.coordinates,
                    approach,
                    barrier,
                    exit_point,
                    source=CrossingCandidateSource.TERRAIN_DERIVED_CROSSING,
                )
                if crossing is not None:
                    return BarrierCrossingResult(
                        status=BarrierCrossingStatus.AVAILABLE,
                        crossing=crossing,
                        terrain_candidate=terrain_candidate,
                    )

        return BarrierCrossingResult(
            status=BarrierCrossingStatus.UNAVAILABLE,
            failure_reason="NO_PATH_SUPPORTED_CROSSING_WITHIN_BARRIER_SEARCH_AREA",
        )

    def _select(
        self,
        coordinates: list[Coordinate],
        approach: HistoricalRoutePoint,
        barrier: HistoricalRoutePoint,
        exit_point: HistoricalRoutePoint,
        *,
        source: CrossingCandidateSource,
        road_edge_ids: list[str] | None = None,
    ) -> CrossingCandidate | None:
        return select_crossing_candidate(
            coordinates,
            approach,
            barrier,
            exit_point,
            source=source,
            road_edge_ids=road_edge_ids,
            elevation_provider=self._elevation,
            search_radius_m=self.search_radius_m,
            elevation_weight=self.elevation_weight,
        )

    def _elevation(self, coordinate: Coordinate) -> float | None:
        if self.terrain_route_service is None:
            return None
        try:
            return self.terrain_route_service.terrain_provider.get_elevation(*coordinate)
        except (TerrainDataUnavailableError, ValueError):
            return None
