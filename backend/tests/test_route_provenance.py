from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever


class Retriever(HistoricalRetriever):
    def __init__(self, items): self.items = items
    def retrieve(self, *_args, **_kwargs): return self.items


class Geography:
    values = {
        "Ostia": (41.76, 12.28, "settlement", "exact_site"),
        "Brundisium": (40.6, 17.9, "port", "exact_site"),
        "Rhodanus": (43.3, 4.8, "river", "representative_point"),
        "Alpes": (43.7, 7.4, "mountain_region", "regional_centroid"),
        "Padus": (44.9, 12.4, "river", "representative_point"),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        if name not in self.values:
            return {"found": False}
        lat, lon, semantics, role = self.values[name]
        return {
            "found": True,
            "id": f"fixture-{name}",
            "canonical_name": name,
            "latitude": lat,
            "longitude": lon,
            "source": "fixture",
            "source_id": name,
            "confidence": 0.8,
            "spatial_semantics": semantics,
            "coordinate_role": role,
        }


def item(identifier, text, **metadata):
    return Evidence(id=identifier, author="Source", work="Work", locator="Book I", text=text, excerpt=text, metadata=metadata)


def build(items):
    tools = AgentToolRegistry(Retriever(items), Geography())
    state = AgentState(session_id="trace")
    tools.execute("search_historical_evidence", {"query": "army movement"}, state)
    result, _ = tools.execute("build_historical_route", {"event_id": "trace", "name": "trace", "period": "218 BCE"}, state)
    return state, result["result"]


def test_event_first_trace_is_observational_and_identifies_success():
    state, result = build([item("move", "The army marched from Ostia to Brundisium.")])
    trace = state.historical_route_diagnostics["provenance_trace"]
    assert result["route"] == state.historical_route.model_dump(mode="json")
    assert trace["route_source"] == "event_anchor"
    assert trace["event_first"]["route_created"] is True
    assert trace["legacy"]["activated"] is False
    assert trace["final_edges"][0]["evidence_ids"] == ["move"]
    assert trace["final_edges"][0]["from"] == "Ostia"


def test_legacy_trace_records_exact_evidence_and_does_not_create_authority():
    state, _ = build([item("cross", "The army crossed the Rhone and entered the Alps.")])
    trace = state.historical_route_diagnostics["provenance_trace"]
    assert trace["route_source"] == "none"
    assert trace["event_first"]["route_created"] is False
    assert trace["legacy"]["activated"] is False
    assert trace["legacy"]["reason_codes"] == ["NON_EXACT_ROUTE_POINT"]
    assert state.historical_route is None


def test_event_first_uses_explicit_departure_arrival_roles_without_fusing_prior_traversal():
    state, _ = build([
        item("first", "The army crossed the Rhone and entered Alpes.", document_id="source", spine_index=1, start_offset=10),
        item("second", "The army marched from Ostia to Brundisium.", document_id="source", spine_index=1, start_offset=20),
    ])
    trace = state.historical_route_diagnostics["provenance_trace"]
    assert trace["route_source"] == "event_anchor"
    assert [(edge["from"], edge["to"]) for edge in trace["final_edges"]] == [("Ostia", "Brundisium")]
    assert [edge["evidence_ids"] for edge in trace["final_edges"]] == [["second"]]


def test_related_place_and_missing_geography_are_explicitly_rejected_in_trace():
    state, _ = build([item("related", "The army campaigned near Padus."), item("missing", "The army marched from Druentia to the Alps.")])
    places = state.historical_route_diagnostics["provenance_trace"]["places"]
    by_name = {entry["normalized_name"]: entry for entry in places}
    assert by_name["Padus"]["anchor_eligible"] is False
    assert by_name["Padus"]["rejection_reason"] == "ROLE_NOT_ANCHOR_ELIGIBLE"
    assert by_name["Druentia"]["rejection_reason"] == "GEOGRAPHY_UNRESOLVED"


def test_contextual_trace_requires_route_intent_and_keeps_authority_tier():
    tools = AgentToolRegistry(Retriever([item("context", "The army campaigned near Padus.")]), Geography())
    state = AgentState(session_id="contextual")
    state.requested_output = "historical_route"
    tools.execute("search_historical_evidence", {"query": "army route"}, state)
    tools.execute("build_historical_route", {"event_id": "trace", "name": "trace", "period": "unspecified"}, state)
    trace = state.historical_route_diagnostics["provenance_trace"]
    padus = next(entry for entry in trace["places"] if entry["normalized_name"] == "Padus")
    assert padus["anchor_eligible"] is True and padus["waypoint_authority"] == "CONTEXTUAL_WAYPOINT"


def test_trace_is_generic_and_never_reads_final_answer_prose():
    state, _ = build([item("move", "The army marched from Ostia to Brundisium.")])
    trace = state.historical_route_diagnostics["provenance_trace"]
    assert "answer" not in trace and "Hannibal" not in str(trace)
    assert all(entry["evidence_ids"] for entry in trace["final_edges"])
