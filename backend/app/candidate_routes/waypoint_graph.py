"""Evidence-preserving waypoint graph construction from an existing HistoricalRoute."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.app.models import HistoricalRoute

from .grid import SyntheticGrid
from .annotation import HistoricalAnnotation
from .corpus import HistoricalCorpus
from .location import LocationConfidence, RouteCoordinateResolver, validate_resolved_coordinate
from .models import ArmyProfile, CandidateRouteAnchor, RankingProfile
from .view_model import to_waypoint_view_model
from .multi_segment import MultiSegmentPlanningRequest
from .planner import HistoricalPlanningRequest, PlanningConstraints


class HistoricalEventStep(BaseModel):
    id: str
    name: str
    order: int = Field(ge=1)
    description: str
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class HistoricalEventChain(BaseModel):
    steps: list[HistoricalEventStep] = Field(default_factory=list, min_length=1)
    corpus_id: str | None = None
    event_id: str | None = None
    period: str | None = None
    region: str | None = None


class HistoricalWaypointRole(str, Enum):
    START = "START"
    WAYPOINT = "WAYPOINT"
    CROSSING = "CROSSING"
    BATTLE_SITE = "BATTLE_SITE"
    END = "END"


class HistoricalWaypoint(BaseModel):
    id: str
    canonical_name: str
    order: int = Field(ge=1)
    role: HistoricalWaypointRole
    evidence_refs: list[str] = Field(default_factory=list)
    location_confidence: LocationConfidence = LocationConfidence.UNKNOWN
    location_notes: str | None = None
    annotation: HistoricalAnnotation | None = None


class HistoricalWaypointSegment(BaseModel):
    from_waypoint: HistoricalWaypoint
    to_waypoint: HistoricalWaypoint
    evidence_refs: list[str] = Field(default_factory=list)


class HistoricalWaypointGraph(BaseModel):
    waypoints: list[HistoricalWaypoint] = Field(default_factory=list)
    segments: list[HistoricalWaypointSegment] = Field(default_factory=list)


class HistoricalWaypointEvidenceError(ValueError):
    pass


class CorpusEvidenceError(HistoricalWaypointEvidenceError):
    pass


class WaypointBuilder:
    """Builds only ordered, evidence-supported graph nodes already present in HistoricalRoute."""

    def build(self, historical_route: HistoricalRoute) -> HistoricalWaypointGraph:
        if not historical_route.evidence_refs:
            raise HistoricalWaypointEvidenceError("historical route has no route-level evidence references")
        waypoints: list[HistoricalWaypoint] = []
        point_count = len(historical_route.ordered_points)
        for index, point in enumerate(historical_route.ordered_points):
            if not point.evidence_refs:
                raise HistoricalWaypointEvidenceError(f"historical route point {point.sequence} has no evidence references")
            role = (
                HistoricalWaypointRole.START if index == 0 else
                HistoricalWaypointRole.END if index == point_count - 1 else
                HistoricalWaypointRole.WAYPOINT
            )
            waypoints.append(HistoricalWaypoint(
                id=point.historical_place.id,
                canonical_name=point.historical_place.canonical_name,
                order=point.sequence,
                role=role,
                evidence_refs=list(point.evidence_refs),
            ))
        segments = [
            HistoricalWaypointSegment(
                from_waypoint=first,
                to_waypoint=second,
                evidence_refs=list(dict.fromkeys([*first.evidence_refs, *second.evidence_refs])),
            )
            for first, second in zip(waypoints, waypoints[1:])
        ]
        return HistoricalWaypointGraph(waypoints=waypoints, segments=segments)

    @staticmethod
    def from_event_chain(event_chain: HistoricalEventChain, corpus: HistoricalCorpus | None = None) -> HistoricalWaypointGraph:
        """Compatibility entry point for corpus-aware event-chain expansion."""
        return HistoricalWaypointBuilder.from_event_chain(event_chain, corpus)


class HistoricalWaypointBuilder:
    """Builds a graph from explicitly supplied, evidence-backed historical event steps."""

    @staticmethod
    def from_event_chain(
        event_chain: HistoricalEventChain,
        corpus: HistoricalCorpus | None = None,
    ) -> HistoricalWaypointGraph:
        HistoricalWaypointBuilder._validate_corpus(event_chain, corpus)
        merged: dict[str, HistoricalWaypoint] = {}
        ordered_ids: list[str] = []
        for step in sorted(event_chain.steps, key=lambda item: (item.order, item.id)):
            if not step.evidence_refs:
                raise HistoricalWaypointEvidenceError(f"event step {step.id} has no evidence references")
            existing = merged.get(step.id)
            if existing is None:
                merged[step.id] = HistoricalWaypoint(
                    id=step.id,
                    canonical_name=step.name,
                    order=step.order,
                    role=HistoricalWaypointRole.WAYPOINT,
                    evidence_refs=list(dict.fromkeys(step.evidence_refs)),
                )
                ordered_ids.append(step.id)
            else:
                merged[step.id] = existing.model_copy(update={
                    "evidence_refs": list(dict.fromkeys([*existing.evidence_refs, *step.evidence_refs])),
                })
        waypoints = [merged[identifier] for identifier in ordered_ids]
        if waypoints:
            waypoints[0] = waypoints[0].model_copy(update={"role": HistoricalWaypointRole.START})
            if len(waypoints) > 1:
                waypoints[-1] = waypoints[-1].model_copy(update={"role": HistoricalWaypointRole.END})
        segments = [
            HistoricalWaypointSegment(
                from_waypoint=first,
                to_waypoint=second,
                evidence_refs=list(dict.fromkeys([*first.evidence_refs, *second.evidence_refs])),
            )
            for first, second in zip(waypoints, waypoints[1:])
        ]
        return HistoricalWaypointGraph(waypoints=waypoints, segments=segments)

    @staticmethod
    def _validate_corpus(event_chain: HistoricalEventChain, corpus: HistoricalCorpus | None) -> None:
        if corpus is None:
            if event_chain.corpus_id is not None:
                raise CorpusEvidenceError("event chain declares a corpus_id but no corpus contract was supplied")
            return
        if event_chain.corpus_id != corpus.id:
            raise CorpusEvidenceError("event chain corpus_id does not match supplied corpus")
        allowed_refs = set(corpus.source_refs)
        for step in event_chain.steps:
            if not set(step.evidence_refs).issubset(allowed_refs):
                raise CorpusEvidenceError(f"event step {step.id} contains evidence outside corpus {corpus.id}")

    @staticmethod
    def attach_annotations(
        waypoints: list[HistoricalWaypoint],
        annotations: list[HistoricalAnnotation],
    ) -> list[HistoricalWaypoint]:
        """Bind only input-supplied annotations whose IDs and evidence references match a waypoint."""
        by_id = {waypoint.id: waypoint for waypoint in waypoints}
        if len(by_id) != len(waypoints):
            raise HistoricalWaypointEvidenceError("waypoint ids must be unique before annotation binding")
        bound: dict[str, HistoricalAnnotation] = {}
        for annotation in annotations:
            if not annotation.source_refs:
                raise HistoricalWaypointEvidenceError(f"annotation {annotation.id} has no source references")
            waypoint = by_id.get(annotation.id)
            if waypoint is None:
                raise HistoricalWaypointEvidenceError(f"annotation {annotation.id} has no matching waypoint")
            if annotation.id in bound:
                raise HistoricalWaypointEvidenceError(f"multiple annotations target waypoint {annotation.id}")
            if not set(annotation.source_refs).issubset(waypoint.evidence_refs):
                raise HistoricalWaypointEvidenceError(f"annotation {annotation.id} source references are not waypoint evidence")
            bound[annotation.id] = annotation
        return [waypoint.model_copy(update={"annotation": bound.get(waypoint.id)}) for waypoint in waypoints]


def waypoints_to_geojson(waypoints: list[HistoricalWaypoint]) -> dict[str, object]:
    """A map-SDK-free metadata carrier; geometry is intentionally absent until a resolver supplies it."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "id": view.id,
                    "name": view.name,
                    "event_type": view.event_type.value if view.event_type else None,
                    "period": view.period,
                    "description": view.description,
                    "location_confidence": view.location_confidence.value,
                    "location_notes": view.location_notes,
                    "evidence_refs": list(view.evidence_refs),
                    "external_references": [reference.model_dump(mode="json") for reference in view.external_references],
                    "order": waypoint.order,
                    "role": waypoint.role.value,
                    "annotation": waypoint.annotation.model_dump(mode="json") if waypoint.annotation else None,
                    "waypoint_name": view.name,
                    "annotation_title": waypoint.annotation.title if waypoint.annotation else None,
                    "annotation_description": view.description,
                    "knowledge_panel_id": view.id,
                },
            }
            for waypoint in waypoints
            for view in [to_waypoint_view_model(waypoint)]
        ],
    }


class WaypointGraphPlanningAdapter:
    """Converts graph edges to existing single-segment requests without changing planner behavior."""

    def __init__(self, coordinate_resolver: RouteCoordinateResolver):
        self.coordinate_resolver = coordinate_resolver

    def to_multi_segment_request(
        self,
        graph: HistoricalWaypointGraph,
        *,
        grid: SyntheticGrid,
        army_profile: ArmyProfile,
        ranking_profile: RankingProfile | None = None,
        constraints: PlanningConstraints | None = None,
    ) -> MultiSegmentPlanningRequest:
        requests: list[HistoricalPlanningRequest] = []
        for segment in graph.segments:
            start = CandidateRouteAnchor(
                historical_place_id=segment.from_waypoint.id,
                canonical_name=segment.from_waypoint.canonical_name,
                evidence_refs=list(segment.from_waypoint.evidence_refs),
            )
            end = CandidateRouteAnchor(
                historical_place_id=segment.to_waypoint.id,
                canonical_name=segment.to_waypoint.canonical_name,
                evidence_refs=list(segment.to_waypoint.evidence_refs),
            )
            active_constraints = constraints or PlanningConstraints()
            start_resolution = self.coordinate_resolver.resolve(start)
            end_resolution = self.coordinate_resolver.resolve(end)
            location_warnings = [
                *validate_resolved_coordinate(start, start_resolution, allow_disputed_locations=active_constraints.allow_disputed_locations),
                *validate_resolved_coordinate(end, end_resolution, allow_disputed_locations=active_constraints.allow_disputed_locations),
            ]
            requests.append(HistoricalPlanningRequest(
                start_anchor=start,
                end_anchor=end,
                start_grid_point=start_resolution.point,
                end_grid_point=end_resolution.point,
                grid=grid,
                army_profile=army_profile,
                ranking_profile=ranking_profile or RankingProfile(),
                constraints=active_constraints,
                location_warnings=location_warnings,
            ))
        return MultiSegmentPlanningRequest(segments=requests)
