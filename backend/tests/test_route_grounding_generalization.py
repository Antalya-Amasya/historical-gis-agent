from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor, HistoricalRouteExtractor


class Geography:
    coordinates = {
        "Genava": (46.2, 6.1), "Gallia": (46.0, 2.0),
        "Melodunum": (48.5, 2.7), "Lutetia": (48.9, 2.35),
        "Carthago Nova": (37.6, -0.98), "Rhodanus": (43.3, 4.8),
    }

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def call(self, tool, arguments):
        self.calls.append((tool, arguments))
        name = arguments["name"]
        if name in self.missing or name not in self.coordinates:
            return {"found": False}
        latitude, longitude = self.coordinates[name]
        return {"found": True, "id": name.lower(), "canonical_name": name, "latitude": latitude, "longitude": longitude, "source": "test registry", "confidence": 0.8, "coordinate_role": "exact_site"}


class Retriever:
    def __init__(self, items):
        self.items = items

    def retrieve(self, query, top_k=5, filters=None):
        return self.items[:top_k]


def evidence(identifier, text, *, document="caesar_gallic_civil_wars", spine=None, offset=None):
    metadata = {"document_id": document}
    if spine is not None:
        metadata.update({"spine_index": spine, "start_offset": offset})
    return Evidence(id=identifier, author="Julius Caesar", work="Gallic War + Civil War", locator="Book I", text=text, excerpt=text, metadata=metadata)


def test_corpus_observed_arrival_then_lead_form_creates_auditable_edge():
    item = evidence("caesar-lutetia", "He reached Melodunum and then led his army to Lutetia.", spine=2, offset=100)
    route = HistoricalRouteExtractor(Geography()).build([item], event_id="caesar", name="Caesar in Gaul", period="52 BCE")
    assert route is not None
    assert [point.historical_place.canonical_name for point in route.ordered_points] == ["Melodunum", "Lutetia"]
    claim = next(claim for claim in route.claims if claim.sequence_status == "explicit")
    assert claim.movement_relation == "arrival_then_lead"
    assert claim.textual_basis == item.text
    assert claim.supporting_evidence_ids == ["caesar-lutetia"]


def test_extended_explicit_from_toward_form_requires_two_recognized_endpoints():
    extractor = HistoricalPlaceMentionExtractor()
    claims = extractor.movement_claims([evidence("toward", "The army marched from Genava toward Gallia.")], event_id="caesar")
    assert [(claim.source_place, claim.destination_place) for claim in claims] == [("Genava", "Gallia")]
    assert extractor.movement_claims([evidence("unsafe", "Caesar marched toward Gallia.")], event_id="caesar") == []


def test_aliases_preserve_raw_to_canonical_debug_provenance():
    mention = next(item for item in HistoricalPlaceMentionExtractor().extract([evidence("geneva", "Caesar arrives at Geneva.")]) if item.normalized_name == "Genava")
    assert mention.raw_name == "Geneva"
    assert mention.alias_provenance == "frozen_corpus_observed"


def test_retrieval_order_cannot_chain_two_otherwise_connected_edges_without_source_order():
    first = evidence("later", "The army marched from Gallia to Genava.")
    second = evidence("earlier", "The army marched from Genava to Lutetia.")
    outcome = HistoricalRouteExtractor(Geography()).build_with_diagnostics([first, second], event_id="caesar", name="Caesar", period="58 BCE")
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == ["Gallia", "Genava"]
    assert outcome.diagnostics["connected_edge_count"] == 1
    assert outcome.diagnostics["reason_codes"] == ["DISCONNECTED_EVIDENCE"]


def test_structure_order_can_chain_same_document_edges_without_using_retrieval_rank():
    later = evidence("later", "The army marched from Genava to Lutetia.", spine=1, offset=200)
    earlier = evidence("earlier", "The army marched from Gallia to Genava.", spine=1, offset=100)
    outcome = HistoricalRouteExtractor(Geography()).build_with_diagnostics([later, earlier], event_id="caesar", name="Caesar", period="58 BCE")
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == ["Gallia", "Genava", "Lutetia"]
    assert outcome.diagnostics["connected_edge_count"] == 2


def test_duplicate_source_positions_cannot_fall_back_to_retrieval_order():
    first = evidence("first", "The army marched from Gallia to Genava.", spine=1, offset=100)
    second = evidence("second", "The army marched from Genava to Lutetia.", spine=1, offset=100)
    outcome = HistoricalRouteExtractor(Geography()).build_with_diagnostics([first, second], event_id="caesar", name="Caesar", period="58 BCE")
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == ["Gallia", "Genava"]
    assert outcome.diagnostics["reason_codes"] == ["DISCONNECTED_EVIDENCE"]


def test_unresolved_mcp_anchor_returns_diagnostics_without_inventing_coordinate():
    item = evidence("caesar-lutetia", "He reached Melodunum and then led his army to Lutetia.", spine=2, offset=100)
    outcome = HistoricalRouteExtractor(Geography(missing={"Lutetia"})).build_with_diagnostics([item], event_id="caesar", name="Caesar", period="52 BCE")
    assert outcome.route is None
    assert outcome.diagnostics["unresolved_anchor_count"] == 1
    assert outcome.diagnostics["reason_codes"] == ["UNRESOLVED_ANCHOR"]


def test_agent_state_receives_bounded_diagnostics_when_route_is_refused():
    item = evidence("caesar-lutetia", "He reached Melodunum and then led his army to Lutetia.", spine=2, offset=100)
    state = AgentState(session_id="diagnostics")
    tools = AgentToolRegistry(Retriever([item]), Geography(missing={"Melodunum"}))
    tools.execute("search_historical_evidence", {"query": "Caesar", "top_k": 5}, state)
    result, _ = tools.execute("build_historical_route", {"event_id": "caesar", "name": "Caesar", "period": "52 BCE"}, state)
    assert result["success"] is True
    assert result["result"]["route"] is None
    assert state.historical_route_diagnostics == result["result"]["diagnostics"]
    assert state.historical_route_diagnostics["reason_codes"] == ["UNRESOLVED_ANCHOR"]
