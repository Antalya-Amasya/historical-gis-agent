"""G6AP: explicit person identity must not depend on query wording like route."""

from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    classify_evidence_relevance,
    explicit_person_identities,
    normalized_query_person_identities,
    normalized_query_subject_terms,
    person_identities_conflict,
    person_identities_match,
)
from backend.tests.test_g6ag_query_subject_scope_normalization import classify_route, evidence

LUCIUS_MOVEMENT = "Lucius Valerius marched from Rome to Capua."
LUCIUS_MOVEMENT_200 = "In 200 BCE Lucius Valerius marched from Rome to Capua."
LUCIUS_CAMPAIGN = "During Campaign Alpha, Lucius Valerius marched from Rome to Capua."
MARCUS_MOVEMENT = "Marcus Valerius marched from Rome to Capua."
MARCUS_ANTONIUS = "Marcus Antonius marched from Rome to Capua."
SURNAME_ONLY = "Valerius marched from Rome to Capua."
ARISTON_MOVEMENT = "Ariston marched from Rome to Capua."

QUERY_POSSESSIVE = "Trace Marcus Valerius's route."
QUERY_ENDPOINT = "Trace Marcus Valerius from Rome to Capua."
QUERY_YEAR = "Trace Marcus Valerius in 200 BCE."
QUERY_CAMPAIGN = "Trace Marcus Valerius during Campaign Alpha."
QUERY_ROUTE_OF = "Trace the route of Marcus Valerius."
QUERY_ARISTON = "Trace Ariston from Rome to Capua."

LEGACY_LUCIUS = (
    "Ariston did not march from Rome to Capua, "
    "but Lucius Valerius marched from Capua to Rome."
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


def execute_route(text: str, *, query: str) -> tuple[AgentState, dict, dict]:
    state = AgentState(
        session_id="g6ap",
        user_query=query,
        requested_output="historical_route",
    )
    tools = AgentToolRegistry(Retriever([evidence("ev1", text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6ap", "name": "G6AP", "period": "100 BCE"},
        state,
    )
    diagnostics = state.historical_route_diagnostics or {}
    trace = diagnostics.get("provenance_trace", {})
    return state, result, {"diagnostics": diagnostics, "trace": trace}


def route_edge_pairs(state: AgentState) -> list[tuple[str, str]]:
    if state.historical_route is None:
        return []
    names = [point.historical_place.canonical_name for point in state.historical_route.ordered_points]
    return list(zip(names, names[1:]))


def assert_wrong_person_rejected(query: str, movement: str):
    ctx = (query,)
    assert normalized_query_person_identities(ctx) == [("marcus", "valerius")]
    assert explicit_person_identities(movement) == [("lucius", "valerius")]
    assert person_identities_match(movement, ctx) is False
    assert person_identities_conflict(movement, ctx) is True
    assert classify_evidence_relevance(movement, ctx) is EvidenceRelevance.OTHER_CAMPAIGN
    _, _, detail, admitted, route_points = classify_route(query, movement)
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2
    state, result, _ = execute_route(movement, query=query)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []


def test_a_exact_gpt6_s6_01_endpoint_query_wrong_person():
    assert_wrong_person_rejected(QUERY_ENDPOINT, LUCIUS_MOVEMENT)


def test_b_possessive_query_same_rejection():
    assert_wrong_person_rejected(QUERY_POSSESSIVE, LUCIUS_MOVEMENT)


def test_c_temporal_query_wrong_person():
    assert_wrong_person_rejected(QUERY_YEAR, LUCIUS_MOVEMENT_200)


def test_d_campaign_query_wrong_person():
    assert_wrong_person_rejected(QUERY_CAMPAIGN, LUCIUS_CAMPAIGN)


def test_e_exact_same_person_endpoint_may_admit():
    ctx = (QUERY_ENDPOINT,)
    assert normalized_query_person_identities(ctx) == [("marcus", "valerius")]
    assert person_identities_match(MARCUS_MOVEMENT, ctx) is True
    _, _, detail, admitted, route_points = classify_route(QUERY_ENDPOINT, MARCUS_MOVEMENT)
    assert detail["admitted"] is True
    assert route_points >= 2


def test_f_single_token_ariston_endpoint_viable():
    ctx = (QUERY_ARISTON,)
    assert normalized_query_person_identities(ctx) == [("ariston",)]
    assert person_identities_match(ARISTON_MOVEMENT, ctx) is True
    _, _, detail, admitted, route_points = classify_route(QUERY_ARISTON, ARISTON_MOVEMENT)
    assert detail["admitted"] is True
    assert route_points >= 2


def test_g_surname_only_insufficient():
    ctx = (QUERY_ENDPOINT,)
    assert person_identities_match(SURNAME_ONLY, ctx) is False
    assert person_identities_conflict(SURNAME_ONLY, ctx) is False
    _, _, detail, admitted, route_points = classify_route(QUERY_ENDPOINT, SURNAME_ONLY)
    assert detail["admitted"] is False
    assert route_points < 2


def test_h_query_identity_equivalence():
    expected = [("marcus", "valerius")]
    for query in (QUERY_POSSESSIVE, QUERY_ENDPOINT, QUERY_YEAR, QUERY_CAMPAIGN, QUERY_ROUTE_OF):
        assert normalized_query_person_identities((query,)) == expected


def test_i_legacy_fallback_without_route_word_blocks_wrong_person():
    state, result, meta = execute_route(LEGACY_LUCIUS, query=QUERY_ENDPOINT)
    trace = meta["trace"]
    assert normalized_query_person_identities((QUERY_ENDPOINT,)) == [("marcus", "valerius")]
    assert ("Roma", "Capua") not in route_edge_pairs(state)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []


def test_j_event_first_endpoint_query_blocks_wrong_person():
    state, result, _ = execute_route(LUCIUS_MOVEMENT, query=QUERY_ENDPOINT)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []


def test_k_endpoint_query_does_not_use_token_bag_fallback():
    ctx = (QUERY_ENDPOINT,)
    assert normalized_query_person_identities(ctx)
    overlap_terms = normalized_query_subject_terms(ctx)
    assert "rome" in overlap_terms or "capua" in overlap_terms
    assert person_identities_match(LUCIUS_MOVEMENT, ctx) is False


def test_l_shared_first_name_still_rejects():
    ctx = (QUERY_ENDPOINT,)
    assert person_identities_match(MARCUS_ANTONIUS, ctx) is False
    assert classify_evidence_relevance(MARCUS_ANTONIUS, ctx) is EvidenceRelevance.OTHER_CAMPAIGN
