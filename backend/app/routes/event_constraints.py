"""Evidence-closed HistoricalEvent to transition-constraint projection.

This deliberately projects no route geometry and performs no crossing-point selection.
"""
from __future__ import annotations

import hashlib

from backend.app.geography.feature_semantics import place_limitations, traversal_eligible
from backend.app.models import (
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    GeographicFeatureKind,
    HistoricalEvent,
    HistoricalEventType,
    PlaceSpatialSemantics,
    TemporalGroundingStatus,
    TransitionAction,
    TransitionConstraint,
)


def feature_kind_for_place(place) -> GeographicFeatureKind:
    mapping = {
        PlaceSpatialSemantics.RIVER: GeographicFeatureKind.RIVER,
        PlaceSpatialSemantics.MOUNTAIN_REGION: GeographicFeatureKind.MOUNTAIN_REGION,
        PlaceSpatialSemantics.SEA: GeographicFeatureKind.SEA,
        PlaceSpatialSemantics.STRAIT: GeographicFeatureKind.STRAIT,
        PlaceSpatialSemantics.PASS: GeographicFeatureKind.PASS,
    }
    return mapping.get(place.spatial_semantics, GeographicFeatureKind.UNKNOWN)


def project_transition_constraints(
    events: list[HistoricalEvent],
    evidence: list,
) -> tuple[list[TransitionConstraint], list[str]]:
    visible = {str(item.id) for item in evidence}
    constraints: list[TransitionConstraint] = []
    diagnostics: list[str] = []
    seen: set[tuple[str, str]] = set()

    for event in events:
        if event.event_type is not HistoricalEventType.MOVEMENT:
            continue
        event_refs = set(event.evidence_refs)
        if not event_refs or not event_refs.issubset(visible):
            diagnostics.append(f"INVALID_EVIDENCE_PROVENANCE:{event.id}")
            continue
        if event.grounding_status is not EventGroundingStatus.EVIDENCE_GROUNDED:
            diagnostics.append(f"INSUFFICIENT_EVENT_GROUNDING:{event.id}")
            continue
        if not event.source_statements:
            diagnostics.append(f"MISSING_SOURCE_STATEMENT:{event.id}")
            continue
        if event.temporal_grounding.status is TemporalGroundingStatus.CONFLICT:
            diagnostics.append(f"TEMPORAL_CONFLICT:{event.id}")
            continue

        for binding in event.place_bindings:
            if binding.role is not EventPlaceRole.RELATED_PLACE:
                continue
            refs = set(binding.evidence_refs) or event_refs
            if not refs or not refs.issubset(visible) or not refs & event_refs:
                diagnostics.append(f"INVALID_BINDING_PROVENANCE:{event.id}")
                continue
            if binding.resolution_status is EventPlaceResolutionStatus.AMBIGUOUS:
                diagnostics.append(f"AMBIGUOUS_FEATURE:{event.id}")
                continue
            if binding.resolution_status is not EventPlaceResolutionStatus.RESOLVED or not binding.place:
                diagnostics.append(f"UNRESOLVED_FEATURE:{event.id}")
                continue
            if not traversal_eligible(binding.place):
                continue

            feature_key = (binding.place.canonical_name or binding.mention.raw_text).casefold()
            dedupe_key = (event.id, feature_key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            digest = hashlib.sha256(f"{event.id}:{feature_key}:{TransitionAction.CROSS.value}".encode()).hexdigest()[:12]
            limitations = list(binding.limitations) or place_limitations(binding.place)
            constraints.append(TransitionConstraint(
                id=f"transition-{digest}",
                event_id=event.id,
                action=TransitionAction.CROSS,
                feature_surface=binding.mention.raw_text,
                feature_canonical=binding.place.canonical_name,
                feature_kind=feature_kind_for_place(binding.place),
                actor_text=event.actor.actor_text,
                actor_status=event.actor.actor_status,
                evidence_refs=sorted(refs),
                source_statement=event.source_statements[0],
                period=event.period,
                feature_coordinate_role=binding.place.coordinate_role,
                resolver_provenance=binding.resolver_provenance,
                limitations=limitations,
            ))

    return constraints, diagnostics or ([] if constraints else ["NO_TRANSITION_CONSTRAINTS"])
