"""Observational provenance ledger for HistoricalRoute construction.

The ledger records decisions already made by event-first and legacy builders.
It has no resolver, ordering, or route-building authority of its own.
"""
from __future__ import annotations

from backend.app.models import Evidence, HistoricalEvent, HistoricalRoute
from backend.app.routes.event_route_orchestration import EventRouteOutcome
from backend.app.routes.extractor import RouteBuildOutcome


def _source(evidence_by_id: dict[str, Evidence], refs: list[str]) -> list[dict[str, str]]:
    return [
        {"evidence_id": ref, "author": item.author, "work": item.work, "locator": item.locator}
        for ref in refs
        if (item := evidence_by_id.get(ref)) is not None
    ]


class HistoricalRouteTraceBuilder:
    """Create a JSON-safe, evidence-closed explanation of existing decisions."""

    @staticmethod
    def build(
        events: list[HistoricalEvent], evidence: list[Evidence], event_first: EventRouteOutcome,
        legacy: RouteBuildOutcome | None, final_route: HistoricalRoute | None, route_source: str,
    ) -> dict[str, object]:
        evidence_by_id = {item.id: item for item in evidence}
        contextual_keys = set(event_first.diagnostics.get("contextual_anchor_keys", []))
        places: list[dict[str, object]] = []
        for event in events:
            for binding in event.place_bindings:
                place = binding.place
                contextual = f"{event.id}|{binding.mention.canonical_hint or binding.mention.raw_text}" in contextual_keys
                eligible = (
                    (binding.role.value in {"ORIGIN", "DESTINATION", "EVENT_SITE"} or contextual)
                    and binding.resolution_status.value == "RESOLVED"
                    and place is not None and place.latitude is not None and place.longitude is not None
                )
                if eligible:
                    rejection = None
                elif binding.role.value == "RELATED_PLACE":
                    rejection = "ROLE_NOT_ANCHOR_ELIGIBLE"
                elif binding.resolution_status.value != "RESOLVED":
                    rejection = "GEOGRAPHY_UNRESOLVED"
                else:
                    rejection = "MISSING_COORDINATE"
                places.append({
                    "normalized_name": binding.mention.canonical_hint or binding.mention.raw_text,
                    "raw_mention": binding.mention.raw_text,
                    "event_id": event.id,
                    "role": binding.role.value,
                    "evidence_ids": list(binding.evidence_refs),
                    "sources": _source(evidence_by_id, list(binding.evidence_refs)),
                    "resolution_status": binding.resolution_status.value,
                    "resolved_place_id": place.id if place else None,
                    "spatial_semantics": place.spatial_semantics.value if place else None,
                    "anchor_eligible": eligible,
                    "waypoint_authority": "CONTEXTUAL_WAYPOINT" if contextual else None,
                    "rejection_reason": rejection,
                })

        final_edges = {
            (claim.source_place, claim.destination_place): claim
            for claim in (final_route.claims if final_route else [])
            if claim.source_place and claim.destination_place
        }
        legacy_claims: list[dict[str, object]] = []
        if legacy and legacy.route:
            for claim in legacy.route.claims:
                accepted = (claim.source_place, claim.destination_place) in final_edges
                legacy_claims.append({
                    "claim_id": claim.id,
                    "origin": claim.source_place,
                    "destination": claim.destination_place,
                    "traversed_place": claim.traversed_place,
                    "relation": claim.movement_relation,
                    "sequence_status": claim.sequence_status,
                    "evidence_ids": list(claim.supporting_evidence_ids),
                    "sources": _source(evidence_by_id, list(claim.supporting_evidence_ids)),
                    "accepted": accepted,
                    "rejection_reason": None if accepted else "NOT_IN_FINAL_ROUTE_CHAIN",
                })
        edges = [
            {
                "from": source,
                "to": destination,
                "route_source": route_source,
                "claim_id": claim.id,
                "evidence_ids": list(claim.supporting_evidence_ids),
                "sources": _source(evidence_by_id, list(claim.supporting_evidence_ids)),
                "ordering_authority": claim.movement_relation or "NONE",
            }
            for (source, destination), claim in final_edges.items()
        ]
        return {
            "route_source": route_source,
            "event_first": {
                "attempted": True,
                "event_count": len(events),
                "anchor_count": event_first.diagnostics.get("anchor_count", 0),
                "ordering_relation_count": event_first.diagnostics.get("ordering_relation_count", 0),
                "rejection_reasons": list(event_first.diagnostics.get("reason_codes", [])),
                "route_created": event_first.route is not None,
            },
            "places": places,
            "legacy": {
                "activated": route_source == "legacy_movement_claims",
                "claims": legacy_claims,
                "reason_codes": list(legacy.diagnostics.get("reason_codes", [])) if legacy else [],
            },
            "final_edges": edges,
        }
