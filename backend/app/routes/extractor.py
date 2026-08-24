"""Rule-based Evidence -> place mentions -> MCP-resolved schematic routes."""

from collections import defaultdict
from pathlib import Path
import re
from typing import Protocol

from backend.app.models import Evidence, ExtractedHistoricalPlaceMention, GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint
from backend.app.routes.place_aliases import HISTORICAL_PLACE_ALIASES, HistoricalPlaceAlias


class GeographyResolver(Protocol):
    def call(self, tool: str, arguments: dict) -> dict: ...


class HistoricalPlaceMentionExtractor:
    """Extract only aliases literally attested in Evidence, never coordinates or order rules."""

    def __init__(self, aliases: tuple[HistoricalPlaceAlias, ...] = HISTORICAL_PLACE_ALIASES):
        self.aliases = aliases

    @staticmethod
    def _source_key(item: Evidence) -> str:
        return f"{item.source_file or item.author + ':' + item.work}#book:{item.book or 'unknown'}"

    @staticmethod
    def _evidence_key(item: Evidence) -> tuple[str, str, int, str]:
        page = item.page_start if item.page_start is not None else 10**9
        return (HistoricalPlaceMentionExtractor._source_key(item), str(item.book or ""), page, "")

    @staticmethod
    def _narrative_phase(text: str, position: int) -> int:
        """Generic textual cue: origins precede, destinations follow otherwise page order."""
        before = text[max(0, position - 120):position].lower()
        if re.search(r"\b(from|leaving|departed from|started from)\s*$", before):
            return 0
        if re.search(r"\b(into|reached|arrived at|arrived in|bring him into)\b[^.]{0,90}$", before):
            return 2
        return 1

    def _matches(self, item: Evidence) -> list[tuple[HistoricalPlaceAlias, int, str, str]]:
        text = " ".join(filter(None, (item.text, item.excerpt, item.topic)))
        lower = text.lower()
        found: list[tuple[HistoricalPlaceAlias, int, str, str]] = []
        for place in self.aliases:
            hits = [(lower.find(alias), alias) for alias in place.aliases if lower.find(alias) >= 0]
            if hits:
                position, alias = min(hits)
                found.append((place, position, alias, text))
        return found

    def ordering_source(self, evidence: list[Evidence]) -> str | None:
        support: dict[str, set[str]] = defaultdict(set)
        phases: dict[str, set[int]] = defaultdict(set)
        for item in evidence:
            source = self._source_key(item)
            for place, position, _, text in self._matches(item):
                support[source].add(place.canonical_name)
                phases[source].add(self._narrative_phase(text, position))
        if not support:
            return None
        # Prefer a source/book that itself attests both an origin and a destination;
        # only then use breadth of attested anchors. This avoids mixing unrelated books.
        return min(support, key=lambda source: (-(0 in phases[source] and 2 in phases[source]), -len(support[source]), source))

    def extract(self, evidence: list[Evidence]) -> list[ExtractedHistoricalPlaceMention]:
        selected_source = self.ordering_source(evidence)
        ordered_evidence = sorted(
            (item for item in evidence if selected_source is None or self._source_key(item) == selected_source),
            key=self._evidence_key,
        )
        grouped: dict[str, ExtractedHistoricalPlaceMention] = {}
        for evidence_index, item in enumerate(ordered_evidence):
            for place, position, alias, text in self._matches(item):
                existing = grouped.get(place.canonical_name)
                if existing is None:
                    start, end = max(0, position - 80), min(len(text), position + len(alias) + 160)
                    grouped[place.canonical_name] = ExtractedHistoricalPlaceMention(
                        raw_name=text[position:position + len(alias)], normalized_name=place.canonical_name,
                        sequence_hint=self._narrative_phase(text, position) * 10**9 + evidence_index * 100000 + position, date_or_period=item.period,
                        evidence_refs=[item.id], context_excerpt=text[start:end], confidence=0.8,
                    )
                elif item.id not in existing.evidence_refs:
                    existing.evidence_refs.append(item.id)
        return sorted(grouped.values(), key=lambda mention: mention.sequence_hint)


class HistoricalRouteExtractor:
    """General deterministic route builder. It never infers a place or coordinate."""

    def __init__(self, geography_client: GeographyResolver, mention_extractor: HistoricalPlaceMentionExtractor | None = None):
        self.geography_client = geography_client
        self.mention_extractor = mention_extractor or HistoricalPlaceMentionExtractor()

    def build(self, evidence: list[Evidence], *, event_id: str, name: str, period: str) -> HistoricalRoute | None:
        mentions = self.mention_extractor.extract(evidence)
        points: list[HistoricalRoutePoint] = []
        unresolved: list[ExtractedHistoricalPlaceMention] = []
        for mention in mentions:
            resolved = self.geography_client.call("resolve_ancient_place", {"name": mention.normalized_name, "period": period})
            if not resolved.get("found"):
                mention.unresolved_reason = "Geography MCP could not resolve this evidence-mentioned place"
                unresolved.append(mention)
                continue
            place = HistoricalPlace.model_validate({key: value for key, value in resolved.items() if key != "found"})
            supporting = [item.author for item in evidence if item.id in mention.evidence_refs]
            points.append(HistoricalRoutePoint(
                sequence=len(points) + 1, historical_place=place, event_summary=mention.context_excerpt,
                date_or_period=mention.date_or_period or period, evidence_refs=mention.evidence_refs,
                confidence=min(mention.confidence, place.confidence), coordinate_role=place.coordinate_role,
                source_support=list(dict.fromkeys(supporting)),
            ))
        if len(points) < 2:
            return None
        ordering_source = self.mention_extractor.ordering_source(evidence)
        distinct_sources = {item.source_file or item.author for item in evidence}
        disagreements = []
        if len(distinct_sources) > 1 and ordering_source:
            disagreements.append(f"Multiple source/book sequences mention anchors; ordering follows narrative/page order in {Path(ordering_source.split('#book:')[0]).name}, book {ordering_source.rsplit('#book:', 1)[-1]}.")
        coordinates = [(point.historical_place.longitude, point.historical_place.latitude) for point in points]
        return HistoricalRoute(
            id=f"{event_id}-evidence-route", event_id=event_id, name=name, period=period,
            ordered_points=points, geometry=GeoJsonLineString(coordinates=coordinates),
            evidence_refs=list(dict.fromkeys(ref for point in points for ref in point.evidence_refs)),
            assumptions=["Evidence-supported anchors are connected with schematic straight segments."],
            limitations=["Historical reconstruction only; not an exact march track or road route.", "Unresolved mentions are retained without coordinates.", "Representative river, mountain, or regional coordinates are not exact passage locations."],
            historical_confidence=round(sum(point.confidence for point in points) / len(points), 2),
            unresolved_mentions=unresolved, source_disagreements=disagreements,
        )
