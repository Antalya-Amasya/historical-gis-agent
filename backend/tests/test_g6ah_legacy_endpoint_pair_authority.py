"""G6AH: legacy fallback must authorize endpoint pairs only within the same positive clause."""

from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.extractor import _legacy_od_has_positive_authority

QUERY = "Trace Ariston's route."

S4_01 = (
    "Ariston did not march from Rome to Capua, "
    "but marched from Rome to Brundisium, "
    "then sailed from Capua to Corcyra."
)
SAME_ORIGIN = (
    "Ariston did not march from Rome to Capua, "
    "but marched from Rome to Brundisium."
)
SAME_DEST = (
    "Ariston did not march from Rome to Capua, "
    "but marched from Brundisium to Capua."
)
TWO_POSITIVE = (
    "Ariston marched from Rome to Capua, "
    "then sailed from Brundisium to Corcyra."
)
SIMPLE_POSITIVE = "Ariston marched from Rome to Capua."
UNRELATED_NEGATION = "Ariston did not hesitate and marched from Rome to Capua."
NO_LONGER = "Ariston no longer marched from Rome to Capua."
NEGATED_ONLY = "Ariston did not march from Rome to Capua."
POSITIVE_LEGACY = "The army crossed the Rhone and entered the Alps."


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Source",
        work="Work",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 10},
    )


class Retriever(HistoricalRetriever):
    def __init__(self, items: list[Evidence]):
        self.items = items

    def retrieve(self, *_args, **_kwargs):
        return self.items


class Geography:
    places = {
        "Roma": (41.9, 12.5),
        "Capua": (41.08, 14.25),
        "Brundisium": (40.6, 17.9),
        "Corcyra": (39.6, 19.9),
        "Rhodanus": (43.3, 4.8),
        "Alpes": (43.7, 7.4),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        if name not in self.places:
            return {"found": False}
        lat, lon = self.places[name]
        return {
            "found": True,
            "id": f"fixture-{name}",
            "canonical_name": name,
            "latitude": lat,
            "longitude": lon,
            "source": "fixture",
            "source_id": name,
            "confidence": 0.8,
            "coordinate_role": "representative_point",
        }


def execute_route(
    text: str,
    *,
    query: str = QUERY,
    evidence_id: str = "ev1",
    requested_output: str = "historical_route",
    user_query: str | None = None,
) -> tuple[AgentState, dict, dict]:
    state = AgentState(
        session_id="g6ah",
        user_query=user_query if user_query is not None else query,
        requested_output=requested_output,
    )
    tools = AgentToolRegistry(Retriever([evidence(evidence_id, text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6ah", "name": "G6AH", "period": "100 BCE"},
        state,
    )
    diagnostics = state.historical_route_diagnostics or {}
    trace = diagnostics.get("provenance_trace", {})
    return state, result, {"diagnostics": diagnostics, "trace": trace}


def route_names(state: AgentState) -> list[str]:
    if state.historical_route is None:
        return []
    return [point.historical_place.canonical_name for point in state.historical_route.ordered_points]


def route_edge_pairs(state: AgentState) -> list[tuple[str, str]]:
    names = route_names(state)
    return list(zip(names, names[1:]))


def legacy_claim_endpoints(trace: dict) -> list[tuple[str | None, str | None]]:
    return [
        (claim.get("origin"), claim.get("destination"))
        for claim in trace.get("legacy", {}).get("claims", [])
        if claim.get("accepted")
    ]


def test_a_exact_s4_01_production_path_blocks_denied_pair():
    state, result, meta = execute_route(S4_01)
    trace = meta["trace"]
    diagnostics = meta["diagnostics"]

    assert trace["event_first"]["route_created"] is False
    assert trace["legacy"]["activated"] is True
    assert diagnostics.get("route_source") == "legacy_movement_claims"
    assert ("Roma", "Capua") not in legacy_claim_endpoints(trace)
    assert ("Roma", "Brundisium") in legacy_claim_endpoints(trace)
    assert ("Capua", "Corcyra") in legacy_claim_endpoints(trace)
    assert ("Roma", "Capua") not in route_edge_pairs(state)
    assert result["result"]["route"] is not None


def _authority(
    sentence: str,
    origin: str,
    destination: str,
    *,
    origin_surface: str | None = None,
    destination_surface: str | None = None,
) -> bool:
    return _legacy_od_has_positive_authority(
        sentence,
        origin,
        destination,
        source_surface=origin_surface or origin,
        destination_surface=destination_surface or destination,
    )


def test_b_same_origin_different_destination():
    assert not _authority(SAME_ORIGIN, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert _authority(SAME_ORIGIN, "Roma", "Brundisium", origin_surface="Rome", destination_surface="Brundisium")
    state, _, meta = execute_route(SAME_ORIGIN)
    assert ("Roma", "Capua") not in legacy_claim_endpoints(meta["trace"])
    assert ("Roma", "Brundisium") in legacy_claim_endpoints(meta["trace"])


def test_c_same_destination_different_origin():
    assert not _authority(SAME_DEST, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert _authority(SAME_DEST, "Brundisium", "Capua", origin_surface="Brundisium", destination_surface="Capua")
    state, _, meta = execute_route(SAME_DEST)
    assert ("Roma", "Capua") not in legacy_claim_endpoints(meta["trace"])
    assert ("Brundisium", "Capua") in legacy_claim_endpoints(meta["trace"])


def test_d_two_independent_positive_movements_without_cross_pairs():
    assert _authority(TWO_POSITIVE, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert _authority(TWO_POSITIVE, "Brundisium", "Corcyra", origin_surface="Brundisium", destination_surface="Corcyra")
    assert not _authority(TWO_POSITIVE, "Roma", "Corcyra", origin_surface="Rome", destination_surface="Corcyra")
    assert not _authority(TWO_POSITIVE, "Brundisium", "Capua", origin_surface="Brundisium", destination_surface="Capua")
    state, _, meta = execute_route(TWO_POSITIVE)
    accepted = set(legacy_claim_endpoints(meta["trace"]))
    assert ("Roma", "Capua") in accepted
    assert ("Brundisium", "Corcyra") in accepted
    assert ("Roma", "Corcyra") not in accepted
    assert ("Brundisium", "Capua") not in accepted


def test_e_simple_positive_fallback_preserved():
    assert _authority(SIMPLE_POSITIVE, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    state, result, meta = execute_route(SIMPLE_POSITIVE)
    trace = meta["trace"]
    assert route_names(state) == ["Roma", "Capua"]
    assert result["result"]["route"] is not None
    assert trace["event_first"]["route_created"] or ("Roma", "Capua") in legacy_claim_endpoints(trace)


def test_f_unrelated_negation_preserves_positive_route():
    assert _authority(UNRELATED_NEGATION, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    state, result, meta = execute_route(
        UNRELATED_NEGATION,
        query="Ariston movement Rome Capua",
        user_query="",
    )
    assert route_names(state) == ["Roma", "Capua"]
    assert result["result"]["route"] is not None


def test_g_explicit_and_no_longer_negation_regression():
    assert not _authority(NEGATED_ONLY, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert not _authority(NO_LONGER, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    state, _, meta = execute_route(NEGATED_ONLY)
    assert ("Roma", "Capua") not in legacy_claim_endpoints(meta["trace"])
    state, _, meta = execute_route(NO_LONGER)
    assert ("Roma", "Capua") not in legacy_claim_endpoints(meta["trace"])


def test_positive_legacy_fallback_still_works():
    state, result, meta = execute_route(
        POSITIVE_LEGACY,
        query="army movement",
        user_query="",
    )
    trace = meta["trace"]
    assert trace["legacy"]["activated"] is True
    assert route_names(state) == ["Rhodanus", "Alpes"]
    assert result["result"]["route"] is not None
