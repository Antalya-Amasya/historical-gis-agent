"""Evidence-closed HistoricalEvent to geographic-anchor projection.

This deliberately projects no edges and performs no route/GIS inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from backend.app.models import (
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalPlace,
    TemporalGroundingStatus,
)


_STRONG_ROLES = {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION, EventPlaceRole.EVENT_SITE}


@dataclass(frozen=True)
class EventAnchor:
    event_id: str
    event_type: str
    canonical_name: str
    role: EventPlaceRole
    latitude: float
    longitude: float
    evidence_refs: tuple[str, ...]
    resolver_provenance: str | None
    coordinate_role: str
    limitations: tuple[str, ...]
    period: str | None
    place: HistoricalPlace
    admission_type: str


def _contextual_admission_reason(event: HistoricalEvent, binding, visible: set[str], enabled: bool) -> str | None:
    """Return a rejection code unless a related place is a bounded V1 map constraint."""
    if not enabled:
        return "CONTEXTUAL_ROUTE_INTENT_REQUIRED"
    if event.grounding_status is not EventGroundingStatus.EVIDENCE_GROUNDED:
        return "CONTEXTUAL_EVENT_ASSOCIATION_INSUFFICIENT"
    if not event.source_statements:
        return "CONTEXTUAL_EVENT_ASSOCIATION_INSUFFICIENT"
    refs = set(binding.evidence_refs)
    if not refs or not refs.issubset(visible) or not refs & set(event.evidence_refs):
        return "CONTEXTUAL_EVIDENCE_PROVENANCE_INVALID"
    if event.temporal_grounding.status is TemporalGroundingStatus.CONFLICT:
        return "CONTEXTUAL_TEMPORAL_CONFLICT"
    return None


def project_event_anchors(
    events: list[HistoricalEvent], evidence: list, *, allow_contextual_related_places: bool = False,
) -> tuple[list[EventAnchor], list[str]]:
    visible = {str(item.id) for item in evidence}
    anchors: list[EventAnchor] = []
    diagnostics: list[str] = []
    for event in events:
        event_refs = set(event.evidence_refs)
        if not event_refs or not event_refs.issubset(visible):
            diagnostics.append(f"INVALID_EVIDENCE_PROVENANCE:{event.id}")
            continue
        for binding in event.place_bindings:
            strong = binding.role in _STRONG_ROLES
            contextual_reason = None
            if not strong and binding.role is EventPlaceRole.RELATED_PLACE:
                contextual_reason = _contextual_admission_reason(event, binding, visible, allow_contextual_related_places)
            contextual = not strong and binding.role is EventPlaceRole.RELATED_PLACE and contextual_reason is None
            if not strong and not contextual:
                if allow_contextual_related_places:
                    diagnostics.append(f"{contextual_reason or 'ROLE_NOT_ANCHOR_ELIGIBLE'}:{event.id}")
                continue
            if binding.resolution_status is EventPlaceResolutionStatus.AMBIGUOUS:
                diagnostics.append(f"AMBIGUOUS_PLACE:{event.id}"); continue
            if binding.resolution_status is not EventPlaceResolutionStatus.RESOLVED or not binding.place:
                diagnostics.append(f"UNRESOLVED_PLACE:{event.id}"); continue
            if binding.place.latitude is None or binding.place.longitude is None:
                diagnostics.append(f"MISSING_COORDINATE:{event.id}"); continue
            refs = set(binding.evidence_refs) or event_refs
            if not refs.issubset(visible):
                diagnostics.append(f"INVALID_EVIDENCE_PROVENANCE:{event.id}"); continue
            admission_type = (
                "MOVEMENT_WAYPOINT" if binding.role in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION}
                else "EVENT_SITE_WAYPOINT" if binding.role is EventPlaceRole.EVENT_SITE
                else "CONTEXTUAL_WAYPOINT"
            )
            anchors.append(EventAnchor(event.id, event.event_type.value, binding.place.canonical_name, binding.role, binding.place.latitude, binding.place.longitude, tuple(sorted(refs)), binding.resolver_provenance, binding.place.coordinate_role, tuple(binding.limitations), event.period, binding.place, admission_type))
    return anchors, diagnostics or ([] if anchors else ["NO_ELIGIBLE_EVENT_PLACES"])
