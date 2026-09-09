"""G6AD: legacy fallback must not resurrect explicitly denied movement."""

from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever


NEGATED_ARISTON = "Ariston did not march from Rome to Capua."
NEGATED_ARMY = "The army did not march from Rome to Capua."
NO_LONGER = "Ariston no longer marched from Rome to Capua."
UNRELATED_NEGATION = "Ariston did not hesitate and marched from Rome to Capua."
MIXED_POLARITY = (
    "Ariston did not march from Rome to Capua, but later sailed from Brundisium to Corcyra."
)
POSITIVE_LEGACY = "The army crossed the Rhone and entered the Alps."
QUERY = "Trace Ariston's route."


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
        "Rhodanus": (43.3, 4.8),
        "Alpes": (43.7, 7.4),
        "Brundisium": (40.6, 17.9),
        "Corcyra": (39.6, 19.9),
    }
    place_roles = {
        "Roma": ("settlement", "exact_site"),
        "Capua": ("settlement", "exact_site"),
        "Brundisium": ("port", "exact_site"),
        "Corcyra": ("settlement", "exact_site"),
        "Rhodanus": ("river", "representative_point"),
        "Alpes": ("mountain_region", "regional_centroid"),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        if name not in self.places:
            return {"found": False}
        lat, lon = self.places[name]
        semantics, role = self.place_roles.get(name, ("settlement", "exact_site"))
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


def execute_route(
    text: str,
    *,
    query: str = QUERY,
    evidence_id: str = "ev1",
    requested_output: str = "answer",
    user_query: str | None = None,
) -> tuple[AgentState, dict, dict]:
    state = AgentState(
        session_id="g6ad",
        user_query=user_query if user_query is not None else query,
        requested_output=requested_output,
    )
    tools = AgentToolRegistry(Retriever([evidence(evidence_id, text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6ad", "name": "G6AD", "period": "100 BCE"},
        state,
    )
    diagnostics = state.historical_route_diagnostics or {}
    trace = diagnostics.get("provenance_trace", {})
    return state, result, {"diagnostics": diagnostics, "trace": trace}


def route_names(state: AgentState) -> list[str]:
    if state.historical_route is None:
        return []
    return [point.historical_place.canonical_name for point in state.historical_route.ordered_points]


def legacy_claim_endpoints(trace: dict) -> list[tuple[str | None, str | None]]:
    return [
        (claim.get("origin"), claim.get("destination"))
        for claim in trace.get("legacy", {}).get("claims", [])
        if claim.get("accepted")
    ]


def test_a_exact_negated_movement_production_fallback_is_blocked():
    state, result, meta = execute_route(NEGATED_ARISTON)
    trace = meta["trace"]
    diagnostics = meta["diagnostics"]

    assert trace["event_first"]["route_created"] is False
    assert route_names(state) != ["Roma", "Capua"]
    assert result["result"]["route"] is None or route_names(state) != ["Roma", "Capua"]
    assert ("Roma", "Capua") not in legacy_claim_endpoints(trace)
    assert diagnostics.get("route_source") in {None, "none", "legacy_movement_claims"}


def test_b_military_negated_movement_production_fallback_is_blocked():
    state, result, meta = execute_route(NEGATED_ARMY, query="Trace the army route.")
    trace = meta["trace"]

    assert trace["event_first"]["route_created"] is False
    assert route_names(state) != ["Roma", "Capua"]
    assert result["result"]["route"] is None or route_names(state) != ["Roma", "Capua"]
    assert ("Roma", "Capua") not in legacy_claim_endpoints(trace)


def test_c_no_longer_marched_is_not_positive_fallback_movement():
    state, _, meta = execute_route(NO_LONGER)
    trace = meta["trace"]

    assert trace["event_first"]["route_created"] is False
    assert route_names(state) != ["Roma", "Capua"]
    assert ("Roma", "Capua") not in legacy_claim_endpoints(trace)


def test_d_positive_legacy_fallback_blocks_non_exact_route_points():
    state, result, meta = execute_route(
        POSITIVE_LEGACY,
        query="army movement",
        user_query="",
    )
    trace = meta["trace"]
    diagnostics = meta["diagnostics"]

    assert trace["event_first"]["route_created"] is False
    assert trace["legacy"]["activated"] is False
    assert diagnostics.get("route_source") == "none"
    assert route_names(state) == []
    assert result["result"]["route"] is None
    assert diagnostics.get("reason_codes") == ["NON_EXACT_ROUTE_POINT"]


def test_e_unrelated_negation_preserves_positive_route():
    state, result, meta = execute_route(
        UNRELATED_NEGATION,
        query="Ariston movement Rome Capua",
        user_query="",
    )

    assert route_names(state) == ["Roma", "Capua"]
    assert result["result"]["route"] is not None


def test_f_mixed_polarity_does_not_resurrect_denied_segment():
    state, result, meta = execute_route(
        MIXED_POLARITY,
        query="Ariston sailed Brundisium Corcyra",
        user_query="",
    )

    assert route_names(state) == ["Brundisium", "Corcyra"]
    assert ("Roma", "Capua") not in zip(route_names(state), route_names(state)[1:])
    assert result["result"]["route"] is not None
