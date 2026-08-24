from backend.app.models import Evidence
from backend.app.routes.extractor import HistoricalRouteExtractor


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
            "Carthago Nova": ("pleiades-265849", 37.599896, -0.98452, "265849"),
            "Rhodanus": ("pleiades-148168", 43.33167, 4.84861, "148168"),
            "Padus": ("pleiades-393469", 44.952389, 12.432028, "393469"),
        }
        place_id, latitude, longitude, source_id = places[name]
        return {"found": True, "id": place_id, "canonical_name": name, "latitude": latitude, "longitude": longitude, "source": "Pleiades: A Gazetteer of Past Places", "source_id": source_id, "source_url": f"https://pleiades.stoa.org/places/{source_id}", "confidence": 0.9, "uncertain": False}


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Polybius", work="Histories", locator="Book III", excerpt=text, text=text, book="III", page_start=1, page_end=1, source_file="polybius.pdf", source_type="pdf")


def test_evidence_produces_ordered_geojson_route_and_preserves_refs() -> None:
    extractor = HistoricalRouteExtractor(FakeGeographyClient())
    route = extractor.extract_hannibal_218([evidence("e-po", "He reached the Po Valley."), evidence("e-rhone", "Hannibal crossed the Rhone."), evidence("e-nova", "He departed New Carthage.")])
    assert route is not None
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Carthago Nova", "Rhodanus", "Padus"]
    assert route.geometry.type == "LineString"
    assert route.geometry.coordinates == [(-0.98452, 37.599896), (4.84861, 43.33167), (12.432028, 44.952389)]
    assert route.evidence_refs == ["e-nova", "e-rhone", "e-po"]
    assert route.assumptions and route.limitations and route.historical_confidence > 0


def test_missing_evidence_never_fills_route_points() -> None:
    assert HistoricalRouteExtractor(FakeGeographyClient()).extract_hannibal_218([evidence("only-one", "Hannibal crossed the Rhone.")]) is None


def test_unresolved_place_is_omitted_without_inventing_coordinates() -> None:
    route = HistoricalRouteExtractor(FakeGeographyClient({"Padus"})).extract_hannibal_218([evidence("r", "Rhone"), evidence("p", "Po Valley")])
    assert route is None
