"""G6AO: shared name tokens must not prove same person identity."""

from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence, TemporalPrecision
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.episode_relevance import EpisodeRelevance
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    classify_evidence_relevance,
    explicit_person_identities,
    has_normalized_subject_overlap,
    has_subject_campaign_conflict,
    normalized_query_person_identities,
    person_identities_conflict,
    person_identities_match,
)
from backend.app.routes.temporal import EvidenceTemporalResolver
from backend.tests.test_g6ag_query_subject_scope_normalization import (
    POSSESSIVE_QUERY,
    classify_route,
    evidence,
)

QUERY_MARCUS = "Trace Marcus Valerius's route."
QUERY_JULIUS = "Trace Julius Caesar's route."
LUCIUS_MOVEMENT = "In 200 BCE Lucius Valerius marched from Rome to Capua."
MARCUS_MOVEMENT = "Marcus Valerius marched from Rome to Capua."
MARCUS_ANTONIUS = "Marcus Antonius marched from Rome to Capua."
GAIUS_CAESAR = "Gaius Caesar marched from Rome to Capua."
SURNAME_ONLY = "Valerius marched from Rome to Capua."
ARISTON_MOVEMENT = "Ariston marched from Rome to Capua."
CAESAR_PREFIX = "In 49 BCE Julius Caesar marched from Rome to Brundisium."
CAESAR_SUFFIX = "Julius Caesar marched from Rome to Brundisium in 49 BCE."


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
        session_id="g6ao",
        user_query=query,
        requested_output="historical_route",
    )
    tools = AgentToolRegistry(Retriever([evidence("ev1", text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6ao", "name": "G6AO", "period": "100 BCE"},
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


def test_a_exact_gpt6_marcus_lucius_valerius_not_admitted():
    ctx = (QUERY_MARCUS,)
    assert normalized_query_person_identities(ctx) == [("marcus", "valerius")]
    assert explicit_person_identities(LUCIUS_MOVEMENT) == [("lucius", "valerius")]
    assert has_normalized_subject_overlap(LUCIUS_MOVEMENT, ctx) is False
    assert person_identities_match(LUCIUS_MOVEMENT, ctx) is False
    assert person_identities_conflict(LUCIUS_MOVEMENT, ctx) is True
    assert has_subject_campaign_conflict(LUCIUS_MOVEMENT, ctx) is True
    assert classify_evidence_relevance(LUCIUS_MOVEMENT, ctx) is EvidenceRelevance.OTHER_CAMPAIGN

    _, _, detail, admitted, route_points = classify_route(QUERY_MARCUS, LUCIUS_MOVEMENT)
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2

    state, result, _ = execute_route(LUCIUS_MOVEMENT, query=QUERY_MARCUS)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []


def test_b_exact_full_name_positive():
    ctx = (QUERY_MARCUS,)
    assert person_identities_match(MARCUS_MOVEMENT, ctx) is True
    assert classify_evidence_relevance(MARCUS_MOVEMENT, ctx) is EvidenceRelevance.DIRECT_SUBJECT
    _, _, detail, admitted, route_points = classify_route(QUERY_MARCUS, MARCUS_MOVEMENT)
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_c_shared_first_name_not_match():
    ctx = (QUERY_MARCUS,)
    assert person_identities_match(MARCUS_ANTONIUS, ctx) is False
    assert person_identities_conflict(MARCUS_ANTONIUS, ctx) is True
    assert classify_evidence_relevance(MARCUS_ANTONIUS, ctx) is EvidenceRelevance.OTHER_CAMPAIGN


def test_d_shared_surname_julius_gaius_caesar_not_match():
    ctx = (QUERY_JULIUS,)
    assert person_identities_match(GAIUS_CAESAR, ctx) is False
    assert person_identities_conflict(GAIUS_CAESAR, ctx) is True
    assert classify_evidence_relevance(GAIUS_CAESAR, ctx) is EvidenceRelevance.OTHER_CAMPAIGN


def test_e_single_token_ariston_regression():
    assert person_identities_match(ARISTON_MOVEMENT, (POSSESSIVE_QUERY,)) is True
    _, _, detail, admitted, route_points = classify_route(POSSESSIVE_QUERY, ARISTON_MOVEMENT)
    assert detail["admitted"] is True
    assert route_points >= 2


def test_f_surname_only_insufficient_not_same_person():
    ctx = (QUERY_MARCUS,)
    assert person_identities_match(SURNAME_ONLY, ctx) is False
    assert person_identities_conflict(SURNAME_ONLY, ctx) is False
    assert classify_evidence_relevance(SURNAME_ONLY, ctx) is not EvidenceRelevance.DIRECT_SUBJECT
    _, _, detail, admitted, route_points = classify_route(QUERY_MARCUS, SURNAME_ONLY)
    assert detail["admitted"] is False
    assert route_points < 2


def test_g_possessive_normalization_same_identity():
    query = "Trace Marcus Valerius's route."
    plain = "Trace Marcus Valerius route."
    assert normalized_query_person_identities((query,)) == normalized_query_person_identities((plain,))


def test_h_date_prefix_suffix_identity_equivalence():
    prefix_ids = explicit_person_identities(CAESAR_PREFIX)
    suffix_ids = explicit_person_identities(CAESAR_SUFFIX)
    assert prefix_ids == suffix_ids == [("julius", "caesar")]
    assert person_identities_match(CAESAR_PREFIX, (QUERY_JULIUS,)) is True
    assert person_identities_match(CAESAR_SUFFIX, (QUERY_JULIUS,)) is True
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve(CAESAR_PREFIX, "probe")
    primary = resolver.primary(readings, "probe")
    assert primary.precision is TemporalPrecision.YEAR
    assert primary.normalized_start == "-49"
    assert codes == ["TEMPORAL_RESOLVED"]
