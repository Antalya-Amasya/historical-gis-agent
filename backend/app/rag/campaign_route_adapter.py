"""Adapters from reviewed RAG campaign chains to evidence-preserving waypoint graphs."""
from __future__ import annotations

from backend.app.candidate_routes.annotation import HistoricalAnnotation
from backend.app.candidate_routes.location import (
    CoordinateResolutionError,
    LocationConfidence,
    ResolvedCoordinate,
    RouteCoordinateResolver,
)
from backend.app.candidate_routes.models import CandidateRouteAnchor
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointBuilder,
    HistoricalWaypointEvidenceError,
    HistoricalWaypointGraph,
    HistoricalWaypointRole,
    HistoricalWaypointSegment,
)

from .event_chain_builder import HistoricalCampaignChain


class CampaignChainWaypointAdapter:
    """Copies reviewed campaign steps in supplied order; it never enriches locations."""

    def to_waypoint_graph(
        self,
        campaign: HistoricalCampaignChain,
        *,
        annotations: list[HistoricalAnnotation] | None = None,
    ) -> HistoricalWaypointGraph:
        if not campaign.steps:
            raise HistoricalWaypointEvidenceError("campaign chain has no steps")
        waypoints: list[HistoricalWaypoint] = []
        event_ids: set[str] = set()
        for step in campaign.steps:
            if step.event_id in event_ids:
                raise HistoricalWaypointEvidenceError(f"campaign chain repeats event {step.event_id}")
            if not step.evidence_refs:
                raise HistoricalWaypointEvidenceError(f"campaign event {step.event_id} has no evidence references")
            event_ids.add(step.event_id)
            waypoints.append(HistoricalWaypoint(
                id=step.event_id,
                canonical_name=step.title,
                order=step.order,
                role=HistoricalWaypointRole.WAYPOINT,
                evidence_refs=list(step.evidence_refs),
                involved_places=list(step.involved_places),
                period=step.period,
                description=None,
                source_book=step.source_book,
                source_chapter=step.source_chapter,
                historical_confidence=step.confidence,
            ))
        waypoints[0] = waypoints[0].model_copy(update={"role": HistoricalWaypointRole.START})
        if len(waypoints) > 1:
            waypoints[-1] = waypoints[-1].model_copy(update={"role": HistoricalWaypointRole.END})
        if annotations:
            waypoints = HistoricalWaypointBuilder.attach_annotations(waypoints, annotations)
        segments = [
            HistoricalWaypointSegment(
                from_waypoint=first,
                to_waypoint=second,
                evidence_refs=list(dict.fromkeys([*first.evidence_refs, *second.evidence_refs])),
            )
            for first, second in zip(waypoints, waypoints[1:])
        ]
        return HistoricalWaypointGraph(waypoints=waypoints, segments=segments)


class MockHistoricalCoordinateResolver(RouteCoordinateResolver):
    """Strict, offline event-ID mapping for audited demos; no name fallback or geocoding."""

    def __init__(self, mapping: dict[str, ResolvedCoordinate]):
        self.mapping = dict(mapping)

    def resolve(self, anchor: CandidateRouteAnchor) -> ResolvedCoordinate:
        try:
            value = self.mapping[anchor.historical_place_id]
        except KeyError as exc:
            raise CoordinateResolutionError(
                f"no configured coordinate for campaign event {anchor.historical_place_id}"
            ) from exc
        if value.display_coordinate is None:
            raise CoordinateResolutionError(
                f"campaign event {anchor.historical_place_id} has no configured display coordinate"
            )
        if value.confidence is LocationConfidence.UNKNOWN:
            raise CoordinateResolutionError(
                f"campaign event {anchor.historical_place_id} has unknown location confidence"
            )
        return value
