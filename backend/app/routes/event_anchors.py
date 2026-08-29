"""Evidence-closed HistoricalEvent to geographic-anchor projection.

This deliberately projects no edges and performs no route/GIS inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from backend.app.models import EventPlaceResolutionStatus, EventPlaceRole, HistoricalEvent


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


def project_event_anchors(events: list[HistoricalEvent], evidence: list) -> tuple[list[EventAnchor], list[str]]:
    visible = {str(item.id) for item in evidence}
    anchors: list[EventAnchor] = []
    diagnostics: list[str] = []
    for event in events:
        event_refs = set(event.evidence_refs)
        if not event_refs or not event_refs.issubset(visible):
            diagnostics.append(f"INVALID_EVIDENCE_PROVENANCE:{event.id}")
            continue
        for binding in event.place_bindings:
            if binding.role not in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION, EventPlaceRole.EVENT_SITE}:
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
            anchors.append(EventAnchor(event.id, event.event_type.value, binding.place.canonical_name, binding.role, binding.place.latitude, binding.place.longitude, tuple(sorted(refs)), binding.resolver_provenance, binding.place.coordinate_role, tuple(binding.limitations), event.period))
    return anchors, diagnostics or ([] if anchors else ["NO_ELIGIBLE_EVENT_PLACES"])
