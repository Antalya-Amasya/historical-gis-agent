"""Deterministic, evidence-gated historical route reconstruction."""

from dataclasses import dataclass
from typing import Protocol

from backend.app.models import Evidence, GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint


class GeographyResolver(Protocol):
    def call(self, tool: str, arguments: dict) -> dict: ...


@dataclass(frozen=True)
class PlaceRule:
    canonical_name: str
    aliases: tuple[str, ...]
    sequence: int
    summary: str


HANNIBAL_218_RULES = (
    PlaceRule("Carthago Nova", ("carthago nova", "new carthage", "cartagena"), 1, "Departure context attested in the retrieved source material."),
    PlaceRule("Rhodanus", ("rhodanus", "rhone", "rhône"), 2, "The retrieved material identifies the Rhône/Rhodanus stage."),
    PlaceRule("Padus", ("padus", "po valley", "river po", "po, italy"), 3, "The retrieved material identifies the Po/Padus stage in northern Italy."),
)


class HistoricalRouteExtractor:
    """Creates only an evidence-supported schematic line; it never invents points."""

    def __init__(self, geography_client: GeographyResolver):
        self.geography_client = geography_client

    @staticmethod
    def _matching_evidence(rule: PlaceRule, evidence: list[Evidence]) -> list[Evidence]:
        matches = []
        for item in evidence:
            haystack = " ".join(filter(None, (item.excerpt, item.text, item.topic))).lower()
            if any(alias in haystack for alias in rule.aliases):
                matches.append(item)
        return matches

    def extract_hannibal_218(self, evidence: list[Evidence]) -> HistoricalRoute | None:
        route_points: list[HistoricalRoutePoint] = []
        all_refs: list[str] = []
        for rule in HANNIBAL_218_RULES:
            supporting = self._matching_evidence(rule, evidence)
            if not supporting:
                continue
            resolved = self.geography_client.call("resolve_ancient_place", {"name": rule.canonical_name, "period": "218 BCE"})
            if not resolved.get("found"):
                continue
            place = HistoricalPlace.model_validate({key: value for key, value in resolved.items() if key != "found"})
            refs = [item.id for item in supporting]
            all_refs.extend(refs)
            route_points.append(HistoricalRoutePoint(sequence=rule.sequence, historical_place=place, event_summary=rule.summary, date_or_period="218 BCE", evidence_refs=refs, confidence=min(place.confidence, 0.85)))
        route_points.sort(key=lambda point: point.sequence)
        if len(route_points) < 2:
            return None
        coordinates = [(point.historical_place.longitude, point.historical_place.latitude) for point in route_points]
        return HistoricalRoute(
            id="hannibal-218-evidence-route",
            event_id="hannibal-alps-218-bc",
            name="Hannibal, 218 BCE: evidence-supported stages",
            period="218 BCE",
            ordered_points=route_points,
            geometry=GeoJsonLineString(coordinates=coordinates),
            evidence_refs=list(dict.fromkeys(all_refs)),
            assumptions=["Connections are schematic straight segments between resolved reference points."],
            limitations=["This is a historical reconstruction, not an exact march track or road route.", "No Alpine pass is asserted unless separately supported and resolved.", "River reference coordinates are representative points, not documented crossing locations."],
            historical_confidence=round(sum(point.confidence for point in route_points) / len(route_points), 2),
        )
