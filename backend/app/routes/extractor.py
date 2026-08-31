"""Evidence -> explicit movement claims -> MCP-resolved schematic routes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from backend.app.models import Evidence, ExtractedHistoricalPlaceMention, GeoJsonLineString, HistoricalClaim, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint
from backend.app.routes.place_aliases import HISTORICAL_PLACE_ALIASES, HistoricalPlaceAlias


class GeographyResolver(Protocol):
    def call(self, tool: str, arguments: dict) -> dict: ...


@dataclass(frozen=True)
class RouteBuildOutcome:
    route: HistoricalRoute | None
    diagnostics: dict[str, object]


def evidence_structural_key(item: Evidence) -> tuple[str, int, int] | None:
    """Only structure-derived source positions may order a connected chain."""
    document_id = str(item.metadata.get("document_id") or item.source_file or "")
    spine_index, start_offset = item.metadata.get("spine_index"), item.metadata.get("start_offset")
    if not document_id or not isinstance(spine_index, int) or not isinstance(start_offset, int):
        return None
    return document_id, spine_index, start_offset


class HistoricalPlaceMentionExtractor:
    """Literal place mentions for display/audit; mentions alone never create routes."""

    def __init__(self, aliases: tuple[HistoricalPlaceAlias, ...] = HISTORICAL_PLACE_ALIASES):
        self.aliases = aliases

    @staticmethod
    def _text(item: Evidence) -> str:
        return " ".join(dict.fromkeys(value for value in (item.text, item.excerpt, item.topic) if value))

    def aliases_in(self, text: str) -> list[tuple[int, HistoricalPlaceAlias, str]]:
        lower = text.lower()
        found: list[tuple[int, HistoricalPlaceAlias, str]] = []
        for place in self.aliases:
            hits = []
            for alias in place.aliases:
                match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lower)
                if match:
                    hits.append((match.start(), alias))
            if hits:
                position, alias = min(hits)
                found.append((position, place, alias))
        return sorted(found, key=lambda value: value[0])

    def extract(self, evidence: list[Evidence]) -> list[ExtractedHistoricalPlaceMention]:
        grouped: dict[str, ExtractedHistoricalPlaceMention] = {}
        for evidence_index, item in enumerate(evidence):
            text = self._text(item)
            for position, place, alias in self.aliases_in(text):
                existing = grouped.get(place.canonical_name)
                if existing is None:
                    start, end = max(0, position - 80), min(len(text), position + len(alias) + 160)
                    grouped[place.canonical_name] = ExtractedHistoricalPlaceMention(raw_name=text[position:position + len(alias)], normalized_name=place.canonical_name, sequence_hint=evidence_index * 100000 + position, date_or_period=item.period, evidence_refs=[item.id], context_excerpt=text[start:end], confidence=0.8, alias_provenance=place.provenance)
                elif item.id not in existing.evidence_refs:
                    existing.evidence_refs.append(item.id)
        return sorted(grouped.values(), key=lambda mention: mention.sequence_hint)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]

    @staticmethod
    def _place_after(start: int, aliases: list[tuple[int, HistoricalPlaceAlias, str]], *, before: int | None = None) -> HistoricalPlaceAlias | None:
        candidates = [place for position, place, _ in aliases if position >= start and (before is None or position < before)]
        return candidates[0] if candidates else None

    def movement_claims(self, evidence: list[Evidence], *, event_id: str) -> list[HistoricalClaim]:
        """Extract only explicit same-statement movement claims.

        Traversal-only statements retain evidence provenance but cannot form a route edge.
        """
        claims: list[HistoricalClaim] = []
        claim_number = 0
        for item in evidence:
            document = str(item.metadata.get("document_id") or item.source_file or item.author)
            for sentence in self._sentences(self._text(item)):
                aliases = self.aliases_in(sentence)
                lower = sentence.lower()
                relation: tuple[str, HistoricalPlaceAlias, HistoricalPlaceAlias] | None = None
                movement_from = re.search(r"\b(?:marched|advanced|proceeded|moved|travelled|traveled|returned|withdrew|retreated|hastened|led(?:\s+(?:his|the)\s+army)?)\b.{0,180}?\bfrom\s+", lower)
                if movement_from:
                    from_match = re.search(r"\bfrom\s+", lower[movement_from.start():])
                    from_start = movement_from.start() + from_match.start() if from_match else None
                    from_end = movement_from.start() + from_match.end() if from_match else None
                    to_match = re.search(r"\b(?:to|into|toward|towards)\s+", lower[from_end:] if from_end is not None else "")
                    to_start = from_end + to_match.start() if to_match and from_end is not None else None
                    to_end = from_end + to_match.end() if to_match and from_end is not None else None
                else:
                    from_start = from_end = to_start = to_end = None
                if from_start is not None and from_end is not None and to_start is not None and to_end is not None:
                    source = self._place_after(from_end, aliases, before=to_start)
                    destination = self._place_after(to_end, aliases)
                    if source and destination and source != destination:
                        relation = ("from_to", source, destination)
                if relation is None:
                    leave_match = re.search(r"\b(?:left|leaving|departed(?:\s+from)?)\s+", lower)
                    arrive_match = re.search(r"\b(?:reached|arriv(?:ed|ing)\s+(?:at|in)|came\s+to|entered|passed\s+into)\s+", lower)
                    if leave_match and arrive_match and leave_match.start() < arrive_match.start():
                        source = self._place_after(leave_match.end(), aliases, before=arrive_match.start())
                        destination = self._place_after(arrive_match.end(), aliases)
                        if source and destination and source != destination:
                            relation = ("departure_arrival", source, destination)
                if relation is None:
                    cross_match = re.search(r"\b(?:crossed|crossing|traversed|traversing)\s+(?:the\s+)?", lower)
                    arrive_match = re.search(r"\b(?:came\s+to|arriv(?:ed|ing)\s+(?:at|in)|entered|passed\s+into)\s+", lower)
                    if cross_match and arrive_match and cross_match.start() < arrive_match.start():
                        crossed = self._place_after(cross_match.end(), aliases, before=arrive_match.start())
                        destination = self._place_after(arrive_match.end(), aliases)
                        if crossed and destination and crossed != destination:
                            relation = ("crossing_arrival", crossed, destination)
                if relation is None:
                    reached_match = re.search(r"\b(?:reached|arrived\s+(?:at|in)|came\s+to)\s+", lower)
                    lead_match = re.search(r"\b(?:led|conducted)\b.{0,180}?\b(?:to|into)\s+", lower)
                    if reached_match and lead_match and reached_match.start() < lead_match.start():
                        source, destination = self._place_after(reached_match.end(), aliases), self._place_after(lead_match.end(), aliases)
                        if source and destination and source != destination:
                            relation = ("arrival_then_lead", source, destination)
                claim_number += 1
                if relation is not None:
                    movement_relation, source, destination = relation
                    claims.append(HistoricalClaim(id=f"{event_id}-movement-{claim_number}", claim_type="MOVEMENT", text=sentence, textual_basis=sentence, source_place=source.canonical_name, destination_place=destination.canonical_name, movement_relation=movement_relation, sequence_status="explicit", supporting_evidence_ids=[item.id], source_documents=[document], confidence=0.9))
                    continue
                traverse_match = re.search(r"\b(?:crossed|crossing|traversed|traversing|entered|entering|ascent\s+of|descent\s+of)\s+(?:the\s+)?", lower)
                if traverse_match:
                    traversed = self._place_after(traverse_match.end(), aliases)
                    if traversed:
                        claims.append(HistoricalClaim(id=f"{event_id}-movement-{claim_number}", claim_type="MOVEMENT", text=sentence, textual_basis=sentence, traversed_place=traversed.canonical_name, movement_relation="traversal", sequence_status="unordered", supporting_evidence_ids=[item.id], source_documents=[document], confidence=0.8))
        return claims


class HistoricalRouteExtractor:
    """Build a schematic route only from explicit movement-claim edges."""

    def __init__(self, geography_client: GeographyResolver, mention_extractor: HistoricalPlaceMentionExtractor | None = None):
        self.geography_client = geography_client
        self.mention_extractor = mention_extractor or HistoricalPlaceMentionExtractor()

    _evidence_order_key = staticmethod(evidence_structural_key)

    def build(self, evidence: list[Evidence], *, event_id: str, name: str, period: str) -> HistoricalRoute | None:
        return self.build_with_diagnostics(evidence, event_id=event_id, name=name, period=period).route

    def build_with_diagnostics(self, evidence: list[Evidence], *, event_id: str, name: str, period: str) -> RouteBuildOutcome:
        mentions = self.mention_extractor.extract(evidence)
        claims = self.mention_extractor.movement_claims(evidence, event_id=event_id)
        ordered = [claim for claim in claims if claim.sequence_status == "explicit" and claim.source_place and claim.destination_place]
        diagnostics: dict[str, object] = {
            "evidence_count": len(evidence),
            "recognized_place_mentions": len(mentions),
            "movement_claim_count": len(claims),
            "explicit_edge_count": len(ordered),
            "connected_edge_count": 0,
            "unresolved_anchor_count": 0,
            "route_point_count": 0,
            "reason_codes": [],
        }
        if not ordered:
            diagnostics["reason_codes"] = ["NO_MOVEMENT_CLAIMS"]
            return RouteBuildOutcome(None, diagnostics)
        evidence_by_id = {item.id: item for item in evidence}
        claim_keys = {
            claim.id: self._evidence_order_key(evidence_by_id[claim.supporting_evidence_ids[0]])
            for claim in ordered
            if claim.supporting_evidence_ids and claim.supporting_evidence_ids[0] in evidence_by_id
        }
        can_chain = (
            len(claim_keys) == len(ordered)
            and len({key[0] for key in claim_keys.values() if key}) == 1
            and len(set(claim_keys.values())) == len(claim_keys)
            and all(claim_keys.values())
        )
        if can_chain:
            ordered = sorted(ordered, key=lambda claim: claim_keys[claim.id])
        elif len(ordered) > 1:
            diagnostics["reason_codes"] = ["DISCONNECTED_EVIDENCE"]
        anchor_names: list[str] = []
        anchor_claims: dict[str, list[HistoricalClaim]] = {}
        for claim in ordered:
            if not anchor_names:
                anchor_names.extend([claim.source_place, claim.destination_place])
                diagnostics["connected_edge_count"] = 1
            elif can_chain and anchor_names[-1] == claim.source_place:
                anchor_names.append(claim.destination_place)
                diagnostics["connected_edge_count"] = int(diagnostics["connected_edge_count"]) + 1
            else:
                continue  # disconnected evidence cannot be silently fused into a route
            for place_name in (claim.source_place, claim.destination_place):
                anchor_claims.setdefault(place_name, []).append(claim)
        if len(anchor_names) < 2:
            diagnostics["reason_codes"] = ["NO_EXPLICIT_EDGES"]
            return RouteBuildOutcome(None, diagnostics)
        points: list[HistoricalRoutePoint] = []
        for place_name in anchor_names:
            supporting_claims = anchor_claims[place_name]
            resolved = self.geography_client.call("resolve_ancient_place", {"name": place_name, "period": period})
            if not resolved.get("found"):
                diagnostics["unresolved_anchor_count"] = int(diagnostics["unresolved_anchor_count"]) + 1
                diagnostics["reason_codes"] = ["UNRESOLVED_ANCHOR"]
                return RouteBuildOutcome(None, diagnostics)  # required evidence-grounded anchor cannot receive an invented coordinate
            place = HistoricalPlace.model_validate({key: value for key, value in resolved.items() if key != "found"})
            refs = list(dict.fromkeys(ref for claim in supporting_claims for ref in claim.supporting_evidence_ids))
            sources = [item.author for item in evidence if item.id in refs]
            points.append(HistoricalRoutePoint(sequence=len(points) + 1, historical_place=place, event_summary=supporting_claims[0].textual_basis or supporting_claims[0].text, date_or_period=period, evidence_refs=refs, confidence=min(min(claim.confidence for claim in supporting_claims), place.confidence), coordinate_role=place.coordinate_role, source_support=list(dict.fromkeys(sources)), claim_ids=[claim.id for claim in supporting_claims]))
        coordinates = [(point.historical_place.longitude, point.historical_place.latitude) for point in points]
        diagnostics["route_point_count"] = len(points)
        return RouteBuildOutcome(HistoricalRoute(id=f"{event_id}-evidence-route", event_id=event_id, name=name, period=period, ordered_points=points, geometry=GeoJsonLineString(coordinates=coordinates), evidence_refs=list(dict.fromkeys(ref for claim in ordered for ref in claim.supporting_evidence_ids)), assumptions=["Evidence-grounded regional anchors are connected with schematic straight segments."], limitations=["Historical reconstruction only; not an exact march track or road route.", "Geometry is a model-derived connection between evidence-grounded anchors, not historical track evidence.", "Representative river, mountain, or regional coordinates are not exact passage locations."], historical_confidence=round(sum(point.confidence for point in points) / len(points), 2), claims=claims), diagnostics)
