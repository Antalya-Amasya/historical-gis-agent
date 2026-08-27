"""Evidence -> explicit movement claims -> MCP-resolved schematic routes."""

from __future__ import annotations

import re
from typing import Protocol

from backend.app.models import Evidence, ExtractedHistoricalPlaceMention, GeoJsonLineString, HistoricalClaim, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint
from backend.app.routes.place_aliases import HISTORICAL_PLACE_ALIASES, HistoricalPlaceAlias


class GeographyResolver(Protocol):
    def call(self, tool: str, arguments: dict) -> dict: ...


class HistoricalPlaceMentionExtractor:
    """Literal place mentions for display/audit; mentions alone never create routes."""

    def __init__(self, aliases: tuple[HistoricalPlaceAlias, ...] = HISTORICAL_PLACE_ALIASES):
        self.aliases = aliases

    @staticmethod
    def _text(item: Evidence) -> str:
        return " ".join(filter(None, (item.text, item.excerpt, item.topic)))

    def aliases_in(self, text: str) -> list[tuple[int, HistoricalPlaceAlias, str]]:
        lower = text.lower()
        found: list[tuple[int, HistoricalPlaceAlias, str]] = []
        for place in self.aliases:
            hits = [(lower.find(alias), alias) for alias in place.aliases if lower.find(alias) >= 0]
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
                    grouped[place.canonical_name] = ExtractedHistoricalPlaceMention(raw_name=text[position:position + len(alias)], normalized_name=place.canonical_name, sequence_hint=evidence_index * 100000 + position, date_or_period=item.period, evidence_refs=[item.id], context_excerpt=text[start:end], confidence=0.8)
                elif item.id not in existing.evidence_refs:
                    existing.evidence_refs.append(item.id)
        return sorted(grouped.values(), key=lambda mention: mention.sequence_hint)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]

    @staticmethod
    def _place_after(start: int, aliases: list[tuple[int, HistoricalPlaceAlias, str]]) -> HistoricalPlaceAlias | None:
        candidates = [place for position, place, _ in aliases if position >= start]
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
                from_match = re.search(r"\bfrom\s+", lower)
                to_match = re.search(r"\b(?:marched|advanced|proceeded|moved|travelled|traveled)\b.{0,180}?\b(?:to|into)\s+", lower)
                if from_match and to_match:
                    source, destination = self._place_after(from_match.end(), aliases), self._place_after(to_match.end(), aliases)
                    if source and destination and source != destination:
                        relation = ("from_to", source, destination)
                if relation is None:
                    leave_match = re.search(r"\b(?:left|departed(?:\s+from)?)\s+", lower)
                    arrive_match = re.search(r"\b(?:reached|arrived\s+(?:at|in)|came\s+to|entered|passed\s+into)\s+", lower)
                    if leave_match and arrive_match:
                        source, destination = self._place_after(leave_match.end(), aliases), self._place_after(arrive_match.end(), aliases)
                        if source and destination and source != destination:
                            relation = ("departure_arrival", source, destination)
                if relation is None:
                    cross_match = re.search(r"\b(?:crossed|crossing|traversed|traversing)\s+(?:the\s+)?", lower)
                    arrive_match = re.search(r"\b(?:came\s+to|arrived\s+(?:at|in)|entered|passed\s+into)\s+", lower)
                    if cross_match and arrive_match:
                        crossed, destination = self._place_after(cross_match.end(), aliases), self._place_after(arrive_match.end(), aliases)
                        if crossed and destination and crossed != destination:
                            relation = ("crossing_arrival", crossed, destination)
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

    def build(self, evidence: list[Evidence], *, event_id: str, name: str, period: str) -> HistoricalRoute | None:
        claims = self.mention_extractor.movement_claims(evidence, event_id=event_id)
        ordered = [claim for claim in claims if claim.sequence_status == "explicit" and claim.source_place and claim.destination_place]
        if not ordered:
            return None
        anchor_names: list[str] = []
        anchor_claims: dict[str, list[HistoricalClaim]] = {}
        for claim in ordered:
            if not anchor_names:
                anchor_names.extend([claim.source_place, claim.destination_place])
            elif anchor_names[-1] == claim.source_place:
                anchor_names.append(claim.destination_place)
            else:
                continue  # disconnected evidence cannot be silently fused into a route
            for place_name in (claim.source_place, claim.destination_place):
                anchor_claims.setdefault(place_name, []).append(claim)
        if len(anchor_names) < 2:
            return None
        points: list[HistoricalRoutePoint] = []
        for place_name in anchor_names:
            supporting_claims = anchor_claims[place_name]
            resolved = self.geography_client.call("resolve_ancient_place", {"name": place_name, "period": period})
            if not resolved.get("found"):
                return None  # required evidence-grounded anchor cannot receive an invented coordinate
            place = HistoricalPlace.model_validate({key: value for key, value in resolved.items() if key != "found"})
            refs = list(dict.fromkeys(ref for claim in supporting_claims for ref in claim.supporting_evidence_ids))
            sources = [item.author for item in evidence if item.id in refs]
            points.append(HistoricalRoutePoint(sequence=len(points) + 1, historical_place=place, event_summary=supporting_claims[0].textual_basis or supporting_claims[0].text, date_or_period=period, evidence_refs=refs, confidence=min(min(claim.confidence for claim in supporting_claims), place.confidence), coordinate_role=place.coordinate_role, source_support=list(dict.fromkeys(sources)), claim_ids=[claim.id for claim in supporting_claims]))
        coordinates = [(point.historical_place.longitude, point.historical_place.latitude) for point in points]
        return HistoricalRoute(id=f"{event_id}-evidence-route", event_id=event_id, name=name, period=period, ordered_points=points, geometry=GeoJsonLineString(coordinates=coordinates), evidence_refs=list(dict.fromkeys(ref for claim in ordered for ref in claim.supporting_evidence_ids)), assumptions=["Evidence-grounded regional anchors are connected with schematic straight segments."], limitations=["Historical reconstruction only; not an exact march track or road route.", "Geometry is a model-derived connection between evidence-grounded anchors, not historical track evidence.", "Representative river, mountain, or regional coordinates are not exact passage locations."], historical_confidence=round(sum(point.confidence for point in points) / len(points), 2), claims=claims)
