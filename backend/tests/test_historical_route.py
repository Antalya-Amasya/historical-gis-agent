from backend.app.models import Evidence
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor, HistoricalRouteExtractor


class FakeGeographyClient:
    def __init__(self, unresolved: set[str] | None = None):
        self.unresolved = unresolved or set()
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool: str, arguments: dict) -> dict:
        self.calls.append((tool, arguments))
        name = arguments["name"]
        if name in self.unresolved:
            return {"found": False}
        places = {
            "Carthago Nova": ("pleiades-265849", 37.599896, -0.98452, "265849", "exact_site"),
            "Rhodanus": ("pleiades-148168", 43.33167, 4.84861, "148168", "representative_point"),
            "Massalia": ("pleiades-148127", 43.296854, 5.382499, "148127", "exact_site"),
            "Alpes": ("pleiades-783", 43.74465275, 7.40183905, "783", "regional_centroid"),
            "Padus": ("pleiades-393469", 44.952389, 12.432028, "393469", "representative_point"),
        }
        place_id, latitude, longitude, source_id, coordinate_role = places[name]
        return {"found": True, "id": place_id, "canonical_name": name, "latitude": latitude, "longitude": longitude, "source": "Pleiades: A Gazetteer of Past Places", "source_id": source_id, "source_url": f"https://pleiades.stoa.org/places/{source_id}", "confidence": 0.9, "uncertain": coordinate_role != "exact_site", "coordinate_role": coordinate_role}


def evidence(identifier: str, text: str, author: str = "Polybius", page: int = 1) -> Evidence:
    return Evidence(id=identifier, author=author, work="Histories", locator="Book III", excerpt=text, text=text, book="III", page_start=page, page_end=page, source_file="polybius.pdf", source_type="pdf")


def build(extractor: HistoricalRouteExtractor, items: list[Evidence]):
    return extractor.build(items, event_id="demo-event", name="Evidence-supported stages", period="218 BCE")


def test_evidence_mentions_are_normalized_deduplicated_and_ordered() -> None:
    mentions = HistoricalPlaceMentionExtractor().extract([
        evidence("nova", "He departed New Carthage."),
        evidence("rhone-1", "He crossed the Rhone."),
        evidence("rhone-2", "The Rhodanus then delayed the army."),
    ])
    assert [mention.normalized_name for mention in mentions] == ["Carthago Nova", "Rhodanus"]
    assert mentions[0].raw_name == "New Carthage"
    assert mentions[1].evidence_refs == ["rhone-1", "rhone-2"]
    assert mentions[0].sequence_hint < mentions[1].sequence_hint


def test_narrative_cues_prioritize_an_attested_origin_before_later_page_summary() -> None:
    mentions = HistoricalPlaceMentionExtractor().extract([
        evidence("march", "After crossing the Iber, he led them through the Pyrenees to the passage of the Rhone.", page=220),
        evidence("summary", "The whole length of his march from New Carthage was measured; the pass over the Alps would bring him into the plains of the Padus.", page=222),
    ])
    assert [mention.normalized_name for mention in mentions] == ["Carthago Nova", "Iberus", "Pyrenaei", "Rhodanus", "Alpes", "Padus"]


def test_evidence_produces_ordered_geojson_route_and_preserves_refs() -> None:
    extractor = HistoricalRouteExtractor(FakeGeographyClient())
    route = build(extractor, [
        evidence("e-nova", "He departed New Carthage."),
        evidence("e-rhone", "Hannibal crossed the Rhone."),
        evidence("e-massalia", "Massalia was discussed beside the Rhodanus."),
        evidence("e-alps", "The army entered the Alps."),
        evidence("e-po", "He reached the Po Valley."),
    ])
    assert route is not None
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Carthago Nova", "Rhodanus", "Massalia", "Alpes", "Padus"]
    assert route.geometry.type == "LineString"
    assert route.geometry.coordinates[0] == (-0.98452, 37.599896)
    assert route.evidence_refs == ["e-nova", "e-rhone", "e-massalia", "e-alps", "e-po"]
    assert route.ordered_points[3].coordinate_role == "regional_centroid"
    assert route.assumptions and route.limitations and route.historical_confidence > 0


def test_unmentioned_places_are_never_added() -> None:
    mentions = HistoricalPlaceMentionExtractor().extract([evidence("other", "The army made camp before dawn.")])
    assert mentions == []


def test_missing_evidence_never_fills_route_points() -> None:
    assert build(HistoricalRouteExtractor(FakeGeographyClient()), [evidence("only-one", "Hannibal crossed the Rhone.")]) is None


def test_unresolved_place_is_preserved_without_inventing_coordinates() -> None:
    route = build(HistoricalRouteExtractor(FakeGeographyClient({"Alpes"})), [
        evidence("n", "New Carthage."), evidence("a", "The Alps."), evidence("p", "Po Valley."),
    ])
    assert route is not None
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Carthago Nova", "Padus"]
    assert route.unresolved_mentions[0].normalized_name == "Alpes"
    assert route.unresolved_mentions[0].unresolved_reason
    assert all(point.historical_place.canonical_name != "Alpes" for point in route.ordered_points)
