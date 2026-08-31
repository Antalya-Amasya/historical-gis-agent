"""EventAnchor ordering validation and ordered HistoricalRoute assembly.

Historical order must be proven by an approved ordering authority.  Retrieval
rank, list order, narrative order, coordinates, roads, and terrain never
establish it.  The resulting ordered waypoints constrain later GIS
reconstruction; their schematic connections are not documentary path proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.app.models import (
    Evidence,
    EventPlaceRole,
    GeoJsonLineString,
    HistoricalClaim,
    HistoricalEvent,
    HistoricalRoute,
    HistoricalRoutePoint,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_anchors import EventAnchor, project_event_anchors
from backend.app.routes.extractor import evidence_structural_key


class OrderingRule(str, Enum):
    SAME_MOVEMENT_EVENT = "SAME_MOVEMENT_EVENT"
    TEMPORAL_ORDER = "TEMPORAL_ORDER"
    SOURCE_STRUCTURAL_ORDER = "SOURCE_STRUCTURAL_ORDER"


_RULE_PRIORITY = {
    OrderingRule.SAME_MOVEMENT_EVENT: 0,
    OrderingRule.TEMPORAL_ORDER: 1,
    OrderingRule.SOURCE_STRUCTURAL_ORDER: 2,
}
_RULE_CONFIDENCE = {
    OrderingRule.SAME_MOVEMENT_EVENT: 0.9,
    OrderingRule.TEMPORAL_ORDER: 0.8,
    OrderingRule.SOURCE_STRUCTURAL_ORDER: 0.7,
}
_RESOLUTION_FAILURES = {"UNRESOLVED_PLACE", "AMBIGUOUS_PLACE", "MISSING_COORDINATE"}
_COMPARABLE_PRECISION = {TemporalPrecision.DAY, TemporalPrecision.MONTH, TemporalPrecision.YEAR, TemporalPrecision.YEAR_RANGE}


@dataclass(frozen=True)
class AnchorOrderingRelation:
    """One proven ordering step between two canonical historical places."""

    earlier: str
    later: str
    rule: OrderingRule
    event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def as_provenance(self) -> dict[str, object]:
        direct_movement = self.rule is OrderingRule.SAME_MOVEMENT_EVENT
        return {
            "earlier": self.earlier,
            "later": self.later,
            "rule": self.rule.value,
            "event_ids": list(self.event_ids),
            "evidence_refs": list(self.evidence_refs),
            "historical_authority": "ATTESTED_MOVEMENT_ORDERING" if direct_movement else "EVIDENCE_GROUNDED_WAYPOINT_ORDERING",
            "connection_semantics": "ALGORITHMIC_GIS_RECONSTRUCTION_REQUIRED",
        }


@dataclass(frozen=True)
class EventRouteOutcome:
    route: HistoricalRoute | None
    relations: tuple[AnchorOrderingRelation, ...]
    diagnostics: dict[str, object]


def _sole(anchors: list[EventAnchor], role: EventPlaceRole) -> EventAnchor | None:
    matches = [anchor for anchor in anchors if anchor.role is role]
    return matches[0] if len(matches) == 1 else None


def _initial(anchors: list[EventAnchor]) -> EventAnchor | None:
    return _sole(anchors, EventPlaceRole.ORIGIN) or _sole(anchors, EventPlaceRole.EVENT_SITE)


def _terminal(anchors: list[EventAnchor]) -> EventAnchor | None:
    return _sole(anchors, EventPlaceRole.DESTINATION) or _sole(anchors, EventPlaceRole.EVENT_SITE)


def _temporal_interval(event: HistoricalEvent | None) -> tuple[int, int] | None:
    """Evidence-grounded, comparable, sufficiently precise historical years only."""
    if event is None:
        return None
    grounding = event.temporal_grounding
    if grounding.status is not TemporalGroundingStatus.EVIDENCE_GROUNDED or grounding.precision not in _COMPARABLE_PRECISION:
        return None
    try:
        start, end = int(grounding.normalized_start), int(grounding.normalized_end)
    except (TypeError, ValueError):
        return None
    return (start, end) if start <= end else (end, start)


def _structural_span(anchors: list[EventAnchor], evidence_by_id: dict[str, Evidence]) -> tuple[tuple[str, int, int], tuple[str, int, int]] | None:
    keys: list[tuple[str, int, int]] = []
    for anchor in anchors:
        for ref in anchor.evidence_refs:
            item = evidence_by_id.get(ref)
            key = evidence_structural_key(item) if item is not None else None
            if key is None:
                return None
            keys.append(key)
    if not keys or len({key[0] for key in keys}) != 1:
        return None
    return min(keys), max(keys)


class EventAnchorRouteBuilder:
    """Build a HistoricalRoute only from anchors whose order is independently proven."""

    def build_with_diagnostics(self, events: list[HistoricalEvent], evidence: list[Evidence], *, event_id: str, name: str, period: str, allow_contextual_related_places: bool = False) -> EventRouteOutcome:
        anchors, projection = project_event_anchors(
            events, evidence, allow_contextual_related_places=allow_contextual_related_places,
        )
        diagnostics: dict[str, object] = {
            "route_source": "event_anchor",
            "anchor_count": len(anchors),
            "distinct_place_count": 0,
            "ordering_relation_count": 0,
            "ordered_place_count": 0,
            "route_point_count": 0,
            "strong_anchor_count": sum(anchor.admission_type != "CONTEXTUAL_WAYPOINT" for anchor in anchors),
            "contextual_anchor_count": sum(anchor.admission_type == "CONTEXTUAL_WAYPOINT" for anchor in anchors),
            "contextual_anchor_keys": [f"{anchor.event_id}|{anchor.canonical_name}" for anchor in anchors if anchor.admission_type == "CONTEXTUAL_WAYPOINT"],
            "projection_diagnostics": list(projection),
            "ordering_provenance": [],
            "reason_codes": [],
        }
        if not anchors:
            unresolved = any(code.split(":")[0] in _RESOLUTION_FAILURES for code in projection)
            diagnostics["reason_codes"] = ["PLACE_RESOLUTION_FAILED"] if unresolved else ["NO_MOVEMENT_EVENTS"]
            return EventRouteOutcome(None, (), diagnostics)
        places: dict[str, list[EventAnchor]] = {}
        for anchor in anchors:
            places.setdefault(anchor.canonical_name, []).append(anchor)
        if any(len({(anchor.latitude, anchor.longitude) for anchor in group}) > 1 for group in places.values()):
            diagnostics["reason_codes"] = ["PLACE_RESOLUTION_FAILED"]
            return EventRouteOutcome(None, (), diagnostics)
        diagnostics["distinct_place_count"] = len(places)
        if len(places) < 2:
            diagnostics["reason_codes"] = ["INSUFFICIENT_PLACES"]
            return EventRouteOutcome(None, (), diagnostics)
        relations = self._relations(events, anchors, {item.id: item for item in evidence})
        diagnostics["ordering_relation_count"] = len(relations)
        chain, edges = self._chain(relations)
        if len(chain) < 2:
            diagnostics["reason_codes"] = ["INSUFFICIENT_ORDERING"]
            return EventRouteOutcome(None, tuple(relations), diagnostics)
        used = tuple(edges[(chain[index], chain[index + 1])] for index in range(len(chain) - 1))
        diagnostics["ordering_provenance"] = [relation.as_provenance() for relation in used]
        diagnostics["ordered_place_count"] = len(chain)
        diagnostics["route_point_count"] = len(chain)
        if len(chain) < len(places):
            diagnostics["reason_codes"] = ["PARTIAL_ROUTE"]
        route = self._route(chain, places, used, {event.id: event for event in events}, {item.id: item for item in evidence}, event_id=event_id, name=name, period=period)
        return EventRouteOutcome(route, used, diagnostics)

    def _relations(self, events: list[HistoricalEvent], anchors: list[EventAnchor], evidence_by_id: dict[str, Evidence]) -> list[AnchorOrderingRelation]:
        events_by_id = {event.id: event for event in events}
        by_event: dict[str, list[EventAnchor]] = {}
        for anchor in anchors:
            by_event.setdefault(anchor.event_id, []).append(anchor)
        found: list[AnchorOrderingRelation] = []
        for identifier, group in by_event.items():
            origin, destination = _sole(group, EventPlaceRole.ORIGIN), _sole(group, EventPlaceRole.DESTINATION)
            if origin is not None and destination is not None and origin.canonical_name != destination.canonical_name:
                found.append(AnchorOrderingRelation(origin.canonical_name, destination.canonical_name, OrderingRule.SAME_MOVEMENT_EVENT, (identifier,), tuple(sorted(set(origin.evidence_refs) | set(destination.evidence_refs)))))
        identifiers = list(by_event)
        for position, first in enumerate(identifiers):
            for second in identifiers[position + 1:]:
                ordered = self._inter_event_order(first, second, by_event, events_by_id, evidence_by_id)
                if ordered is None:
                    continue
                earlier_id, later_id, rule = ordered
                tail, head = _terminal(by_event[earlier_id]), _initial(by_event[later_id])
                if tail is None or head is None or tail.canonical_name == head.canonical_name:
                    continue
                found.append(AnchorOrderingRelation(tail.canonical_name, head.canonical_name, rule, (earlier_id, later_id), tuple(sorted(set(tail.evidence_refs) | set(head.evidence_refs)))))
        return found

    @staticmethod
    def _inter_event_order(first: str, second: str, by_event: dict[str, list[EventAnchor]], events_by_id: dict[str, HistoricalEvent], evidence_by_id: dict[str, Evidence]) -> tuple[str, str, OrderingRule] | None:
        first_time, second_time = _temporal_interval(events_by_id.get(first)), _temporal_interval(events_by_id.get(second))
        if first_time is not None and second_time is not None:
            if first_time[1] < second_time[0]:
                return first, second, OrderingRule.TEMPORAL_ORDER
            if second_time[1] < first_time[0]:
                return second, first, OrderingRule.TEMPORAL_ORDER
            return None  # comparable evidence-grounded values that overlap actively fail to separate the events
        first_span, second_span = _structural_span(by_event[first], evidence_by_id), _structural_span(by_event[second], evidence_by_id)
        if first_span is None or second_span is None or first_span[0][0] != second_span[0][0]:
            return None
        if first_span[1] < second_span[0]:
            return first, second, OrderingRule.SOURCE_STRUCTURAL_ORDER
        if second_span[1] < first_span[0]:
            return second, first, OrderingRule.SOURCE_STRUCTURAL_ORDER
        return None

    @staticmethod
    def _chain(relations: list[AnchorOrderingRelation]) -> tuple[list[str], dict[tuple[str, str], AnchorOrderingRelation]]:
        """Reduce proven relations to one unambiguous continuous chain, or nothing."""
        best: dict[tuple[str, str], AnchorOrderingRelation] = {}
        for relation in relations:
            key = (relation.earlier, relation.later)
            if key not in best or _RULE_PRIORITY[relation.rule] < _RULE_PRIORITY[best[key].rule]:
                best[key] = relation
        contradictory = {pair for pair in best if (pair[1], pair[0]) in best}
        usable = {pair: relation for pair, relation in best.items() if pair not in contradictory}
        outgoing: dict[str, set[str]] = {}
        incoming: dict[str, set[str]] = {}
        for earlier, later in usable:
            outgoing.setdefault(earlier, set()).add(later)
            incoming.setdefault(later, set()).add(earlier)
        edges = {(earlier, later): relation for (earlier, later), relation in usable.items() if len(outgoing[earlier]) == 1 and len(incoming[later]) == 1}
        successor = {earlier: later for earlier, later in edges}
        targets = set(successor.values())
        chains: list[list[str]] = []
        for start in (node for node in successor if node not in targets):
            chain, seen = [start], {start}
            while chain[-1] in successor and successor[chain[-1]] not in seen:
                chain.append(successor[chain[-1]])
                seen.add(chain[-1])
            chains.append(chain)
        if not chains:
            return [], edges
        longest = max(len(chain) for chain in chains)
        candidates = [chain for chain in chains if len(chain) == longest]
        return (candidates[0] if len(candidates) == 1 else []), edges

    @staticmethod
    def _route(chain: list[str], places: dict[str, list[EventAnchor]], used: tuple[AnchorOrderingRelation, ...], events_by_id: dict[str, HistoricalEvent], evidence_by_id: dict[str, Evidence], *, event_id: str, name: str, period: str) -> HistoricalRoute:
        claims: list[HistoricalClaim] = []
        for index, relation in enumerate(used, start=1):
            direct_movement = relation.rule is OrderingRule.SAME_MOVEMENT_EVENT
            text = (
                f"{relation.earlier} precedes {relation.later} within the same attested movement event."
                if direct_movement
                else f"{relation.earlier} is an evidence-grounded waypoint before {relation.later} by {relation.rule.value}; no direct movement is asserted."
            )
            claims.append(HistoricalClaim(
                id=f"{event_id}-ordering-{index}",
                claim_type="ORDERING" if direct_movement else "WAYPOINT_ORDERING",
                text=text,
                textual_basis=relation.rule.value,
                source_place=relation.earlier,
                destination_place=relation.later,
                movement_relation=relation.rule.value if direct_movement else None,
                sequence_status="explicit",
                supporting_evidence_ids=list(relation.evidence_refs),
                source_documents=sorted({str(evidence_by_id[ref].metadata.get("document_id") or evidence_by_id[ref].source_file or evidence_by_id[ref].author) for ref in relation.evidence_refs if ref in evidence_by_id}),
                confidence=_RULE_CONFIDENCE[relation.rule],
            ))
        points: list[HistoricalRoutePoint] = []
        limitations = {"Historical reconstruction only; not an exact march track or road route.", "Geometry is a schematic connection between ordered historical anchors, not path evidence."}
        for position, place_name in enumerate(chain, start=1):
            group = places[place_name]
            anchor = group[0]
            refs = sorted({ref for item in group for ref in item.evidence_refs})
            source_events = [events_by_id[item.event_id] for item in group if item.event_id in events_by_id]
            limitations.update(limitation for item in group for limitation in item.limitations)
            points.append(HistoricalRoutePoint(
                sequence=position, historical_place=anchor.place,
                event_summary=source_events[0].summary if source_events else place_name,
                date_or_period=anchor.period or period, evidence_refs=refs,
                confidence=anchor.place.confidence, coordinate_role=anchor.coordinate_role,
                source_support=sorted({evidence_by_id[ref].author for ref in refs if ref in evidence_by_id}),
                claim_ids=[claim.id for claim in claims if place_name in (claim.source_place, claim.destination_place)],
            ))
        return HistoricalRoute(
            id=f"{event_id}-event-anchor-route", event_id=event_id, name=name, period=period,
            ordered_points=points, geometry=GeoJsonLineString(coordinates=[(point.historical_place.longitude, point.historical_place.latitude) for point in points]),
            evidence_refs=sorted({ref for point in points for ref in point.evidence_refs}),
            assumptions=[
                "Anchor order is taken only from proven historical ordering relations, never from geography or retrieval order.",
                "Connections between consecutive waypoints are inputs to later algorithmic GIS reconstruction and do not by themselves assert direct historical movement.",
            ],
            limitations=sorted(limitations),
            historical_confidence=round(sum(point.confidence for point in points) / len(points), 2),
            claims=claims,
        )
