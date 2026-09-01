"""Conservative event-place grounding through the existing Geography resolver."""
from __future__ import annotations

from typing import Protocol

from backend.app.models import (
    EventPlaceResolutionStatus,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalPlace,
    PlaceMentionValidationClass,
    PlaceSpatialSemantics,
)


class GeographyResolver(Protocol):
    def call(self, tool: str, arguments: dict) -> dict: ...


class HistoricalEventPlaceResolver:
    """Resolve text mentions independently of routes, claims, terrain, and roads."""

    def __init__(self, geography_client: GeographyResolver) -> None:
        self.geography_client = geography_client

    @staticmethod
    def _provenance(place: HistoricalPlace) -> str:
        identifier = f" / {place.source_id}" if place.source_id else ""
        return f"Geography resolver: {place.source}{identifier}"

    @staticmethod
    def _limitations(place: HistoricalPlace) -> list[str]:
        if place.spatial_semantics is PlaceSpatialSemantics.RIVER:
            return ["Resolved river location is a representative entity point, not an exact event site."]
        if place.spatial_semantics is PlaceSpatialSemantics.MOUNTAIN_REGION:
            return ["Resolved mountain-region centroid identifies a region, not an exact event site or pass."]
        if place.coordinate_role != "exact_site":
            return [f"Resolved coordinate role is {place.coordinate_role}; it is not asserted as an exact event site."]
        return []

    def resolve(self, events: list[HistoricalEvent]) -> tuple[list[HistoricalEvent], dict[str, object]]:
        cache: dict[str, dict] = {}
        diagnostics = {
            "place_mention_count": 0, "resolved_place_count": 0, "unresolved_place_count": 0,
            "unlocated_place_count": 0, "unavailable_place_count": 0,
            "exact_site_count": 0, "representative_point_count": 0, "regional_count": 0,
            "ambiguous_count": 0, "non_place_validated_count": 0, "reason_codes": [],
        }
        resolved_events: list[HistoricalEvent] = []
        for event in events:
            bindings: list[HistoricalEventPlaceBinding] = []
            seen: dict[tuple[str, str], HistoricalEventPlaceBinding] = {}
            for mention in event.place_mentions:
                diagnostics["place_mention_count"] += 1
                if mention.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE:
                    diagnostics["non_place_validated_count"] += 1
                    binding = HistoricalEventPlaceBinding(
                        mention=mention,
                        place=None,
                        role=mention.role,
                        resolution_status=EventPlaceResolutionStatus.TEXT_ONLY,
                        evidence_refs=list(mention.evidence_refs),
                        limitations=[
                            f"Broad mention retained for audit; geography skipped ({mention.validation_reason or 'high-confidence non-place'}).",
                        ],
                    )
                    marker = ((mention.canonical_hint or mention.raw_text).casefold(), mention.role.value)
                    existing = seen.get(marker)
                    if existing is None:
                        bindings.append(binding)
                        seen[marker] = binding
                    else:
                        existing.evidence_refs = list(dict.fromkeys([*existing.evidence_refs, *binding.evidence_refs]))
                    continue
                lookup = mention.canonical_hint or mention.raw_text
                if lookup not in cache:
                    try:
                        cache[lookup] = self.geography_client.call(
                            "resolve_ancient_place", {"name": lookup, "period": event.period},
                        )
                    except Exception:
                        cache[lookup] = {"found": False, "resolver_error": True}
                result = cache[lookup]
                place: HistoricalPlace | None = None
                status = EventPlaceResolutionStatus.UNRESOLVED
                limitations: list[str] = []
                provenance: str | None = None
                if result.get("found"):
                    place = HistoricalPlace.model_validate({key: value for key, value in result.items() if key != "found"})
                    status = EventPlaceResolutionStatus.RESOLVED
                    provenance, limitations = self._provenance(place), self._limitations(place)
                    diagnostics["resolved_place_count"] += 1
                    if place.spatial_semantics is PlaceSpatialSemantics.SETTLEMENT and place.coordinate_role == "exact_site":
                        diagnostics["exact_site_count"] += 1
                    elif place.coordinate_role == "representative_point":
                        diagnostics["representative_point_count"] += 1
                    elif place.coordinate_role == "regional_centroid":
                        diagnostics["regional_count"] += 1
                elif result.get("status") == "UNLOCATED":
                    status = EventPlaceResolutionStatus.UNLOCATED
                    limitations = ["Authority candidate exists but no safe coordinate is available."]
                    diagnostics["unlocated_place_count"] += 1
                elif result.get("status") == "UNAVAILABLE":
                    status = EventPlaceResolutionStatus.UNAVAILABLE
                    limitations = [result.get("reason") or "Geographic authority infrastructure is unavailable."]
                    diagnostics["unavailable_place_count"] += 1
                elif result.get("ambiguous") or result.get("alternatives") or result.get("status") == "AMBIGUOUS":
                    status = EventPlaceResolutionStatus.AMBIGUOUS
                    limitations = ["Resolver returned no single safe historical-place identity."]
                    diagnostics["ambiguous_count"] += 1
                else:
                    limitations = ["No trusted geographic resolver entry was available; no coordinate was invented."]
                    diagnostics["unresolved_place_count"] += 1
                binding = HistoricalEventPlaceBinding(
                    mention=mention, place=place, role=mention.role, resolution_status=status,
                    confidence=place.confidence if place else None, evidence_refs=list(mention.evidence_refs),
                    resolver_provenance=provenance, limitations=limitations,
                )
                marker = ((place.id if place else mention.canonical_hint or mention.raw_text).casefold(), mention.role.value)
                existing = seen.get(marker)
                if existing is None:
                    bindings.append(binding)
                    seen[marker] = binding
                else:
                    existing.evidence_refs = list(dict.fromkeys([*existing.evidence_refs, *binding.evidence_refs]))
            places = list({binding.place.id: binding.place for binding in bindings if binding.place}.values())
            resolved_events.append(event.model_copy(update={"place_bindings": bindings, "places": places}))
        codes: list[str] = []
        if diagnostics["resolved_place_count"]:
            codes.append("PLACE_RESOLVED")
        if diagnostics["unresolved_place_count"]:
            codes.append("PLACE_UNRESOLVED")
        if diagnostics["unlocated_place_count"]:
            codes.append("PLACE_UNLOCATED")
        if diagnostics["unavailable_place_count"]:
            codes.append("PLACE_UNAVAILABLE")
        if diagnostics["ambiguous_count"]:
            codes.append("PLACE_AMBIGUOUS")
        if diagnostics["non_place_validated_count"]:
            codes.append("PLACE_NON_GEOGRAPHIC_VALIDATED")
        if diagnostics["representative_point_count"] or diagnostics["regional_count"]:
            codes.append("NON_EXACT_SPATIAL_SEMANTICS")
        if any(binding.place and binding.place.spatial_semantics is PlaceSpatialSemantics.UNKNOWN for event in resolved_events for binding in event.place_bindings):
            codes.append("UNKNOWN_PLACE_SEMANTICS")
        diagnostics["reason_codes"] = codes or ["NO_EVENT_PLACE_MENTIONS"]
        return resolved_events, diagnostics
