from backend.app.models import Evidence
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor, HistoricalRouteExtractor


class FakeGeographyClient:
    def __init__(self, unresolved=None): self.unresolved, self.calls = unresolved or set(), []
    def call(self, tool, arguments):
        self.calls.append((tool, arguments)); name = arguments["name"]
        if name in self.unresolved: return {"found": False}
        places = {"Carthago Nova": (37.6, -0.98), "Rhodanus": (43.3, 4.8), "Alpes": (44.0, 7.0), "Italia": (42.5, 12.5)}
        lat, lon = places[name]
        return {"found": True, "id": name, "canonical_name": name, "latitude": lat, "longitude": lon, "source": "test", "confidence": .9, "coordinate_role": "regional_centroid"}


def evidence(identifier, text):
    return Evidence(id=identifier, author="Polybius", work="Histories", locator="Book III", excerpt=text, text=text, book="III", source_file="polybius.pdf")


def test_mentions_are_audit_data_not_route_order():
    mentions = HistoricalPlaceMentionExtractor().extract([evidence("m", "New Carthage and the Rhone were discussed.")])
    assert [mention.normalized_name for mention in mentions] == ["Carthago Nova", "Rhodanus"]
    assert HistoricalRouteExtractor(FakeGeographyClient()).build([evidence("m", "New Carthage and the Rhone were discussed.")], event_id="x", name="x", period="218 BCE") is None


def test_explicit_linked_edges_form_a_route_without_adding_incidental_places():
    route = HistoricalRouteExtractor(FakeGeographyClient()).build([evidence("a", "The army marched from New Carthage to the Rhone."), evidence("b", "Having crossed the Alps, the army came to Italy.")], event_id="x", name="x", period="218 BCE")
    assert route is not None
    # Disconnected claims are not silently fused into one invented itinerary.
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Carthago Nova", "Rhodanus"]
    assert route.evidence_refs == ["a", "b"]


def test_single_traversal_and_unresolved_required_edge_fail_closed():
    assert HistoricalRouteExtractor(FakeGeographyClient()).build([evidence("a", "Hannibal crossed the Alps.")], event_id="x", name="x", period="218 BCE") is None
    assert HistoricalRouteExtractor(FakeGeographyClient({"Italia"})).build([evidence("a", "Having crossed the Alps, the army came to Italy.")], event_id="x", name="x", period="218 BCE") is None
