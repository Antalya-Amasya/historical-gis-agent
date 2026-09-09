from backend.app.models import Evidence
from backend.app.routes.extractor import HistoricalRouteExtractor


def evidence(identifier, text, document="polybius_histories_v1"):
    return Evidence(id=identifier, author="Polybius", work="Histories", locator="Book 3", excerpt=text, text=text, metadata={"document_id": document})


class Geography:
    def __init__(self, missing=()): self.missing = set(missing)
    def call(self, tool, arguments):
        name = arguments["name"]
        if name in self.missing: return {"found": False}
        coords = {
            "Carthago Nova": (37.6, -0.98, "port", "exact_site"),
            "Genava": (46.2, 6.1, "settlement", "exact_site"),
            "Rhodanus": (43.3, 4.8, "river", "representative_point"),
            "Alpes": (44.0, 7.0, "mountain_region", "regional_centroid"),
            "Italia": (42.5, 12.5, "region", "regional_centroid"),
        }
        if name not in coords:
            return {"found": False}
        latitude, longitude, semantics, role = coords[name]
        return {
            "found": True,
            "id": name.lower(),
            "canonical_name": name,
            "latitude": latitude,
            "longitude": longitude,
            "source": "local_test_resolver",
            "confidence": 0.7,
            "uncertain": role != "exact_site",
            "spatial_semantics": semantics,
            "coordinate_role": role,
        }


def build(items, *, geography=None):
    return HistoricalRouteExtractor(geography or Geography()).build(items, event_id="test", name="test", period="218 BCE")


def test_explicit_movement_creates_ordered_provenance_preserving_route():
    route = build([evidence("move", "The army marched from New Carthage to Genava.")])
    assert route is not None
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Carthago Nova", "Genava"]
    claim = next(claim for claim in route.claims if claim.sequence_status == "explicit")
    assert (claim.claim_type, claim.source_place, claim.destination_place) == ("MOVEMENT", "Carthago Nova", "Genava")
    assert claim.supporting_evidence_ids == ["move"] and claim.source_documents == ["polybius_histories_v1"]
    assert all(point.claim_ids and point.evidence_refs for point in route.ordered_points)
    assert "not an exact march track" in " ".join(route.limitations)


def test_reversed_explicit_movement_uses_semantic_not_alias_order():
    route = build([evidence("reverse", "The army marched from Genava to New Carthage.")])
    assert route is not None
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Genava", "Carthago Nova"]


def test_bare_or_incidental_places_never_create_route_anchors():
    assert build([evidence("bare", "Rhone")]) is None
    assert build([evidence("incidental", "New Carthage was discussed alongside the Rhone.")]) is None
    route = build([evidence("third", "After discussing Rome, the army marched from New Carthage to Genava.")])
    assert route is not None
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Carthago Nova", "Genava"]


def test_crossing_without_arrival_is_unordered_and_fails_closed():
    assert build([evidence("cross", "Hannibal crossed the Alps.")]) is None
    assert build([evidence("relation", "Having crossed the Alps, the army came to Italy.", "livy_history_of_rome_books_9_26")]) is None


def test_unresolved_required_anchor_fails_without_fabricated_coordinate():
    assert build([evidence("move", "The army marched from New Carthage to Genava.")], geography=Geography(missing={"Genava"})) is None
