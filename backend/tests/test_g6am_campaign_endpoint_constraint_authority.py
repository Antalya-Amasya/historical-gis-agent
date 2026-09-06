"""G6AM: explicit campaign/war episode constraints and directed endpoint matching."""

from __future__ import annotations

import backend.app.routes.episode_relevance as er
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.evidence_relevance import (
    event_relevance,
    relation_admission_allowed,
)
from backend.tests.test_g6ag_query_subject_scope_normalization import (
    COORDS,
    classify_route,
    evidence,
    movement_event,
)
from backend.tests.test_g4b_route_components import relation

QUERY_ALPHA = "Trace Ariston during Campaign Alpha."
QUERY_SICILIAN = "Trace Ariston during the Sicilian War."
QUERY_NORTHERN = "Trace Ariston during the Northern campaign."
QUERY_ENDPOINT = "Trace Ariston from Rome to Capua."
QUERY_UNRESTRICTED = "Trace Ariston's route."
QUERY_ALPHA_YEAR = "Trace Ariston during Campaign Alpha in 200 BCE."
QUERY_YEAR_ONLY = "Trace Ariston's route in 200 BCE."

MOVEMENT = "Ariston marched from Rome to Capua."
ALPHA_MOVEMENT = "During Campaign Alpha, Ariston marched from Rome to Capua."
SICILIAN_MOVEMENT = "During the Sicilian War, Ariston marched from Rome to Capua."
NORTHERN_MOVEMENT = "During the Northern campaign, Ariston marched from Rome to Capua."
REVERSE_MOVEMENT = "Ariston marched from Capua to Rome."
SAME_ORIGIN = "Ariston marched from Rome to Brundisium."
SAME_DEST = "Ariston marched from Brundisium to Capua."
DISJOINT = "Ariston marched from Athens to Sparta."
YEAR_ONLY_MOVEMENT = "In 200 BCE Ariston marched from Rome to Capua."
FALLBACK_NO_ALPHA = "Ariston crossed the Rhone and entered the Alps."
FALLBACK_REVERSE = (
    "Ariston marched from Capua to Rome, "
    "then sailed from Brundisium to Corcyra."
)


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
        "Athens": (37.98, 23.73),
        "Sparta": (37.08, 22.43),
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


def classify_endpoint_route(
    query: str,
    statement: str,
    *,
    origin: str,
    destination: str,
):
    extra_coords = {
        "Brundisium": (40.6, 17.9),
    }
    for name, coords in extra_coords.items():
        COORDS.setdefault(name, coords)
    event = movement_event("e1", statement, origin, destination)
    ev = evidence("ev1", statement)
    rel = relation(
        origin,
        destination,
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=(event.id,),
    )
    tag = event_relevance(event, {ev.id: ev}, (query,))
    episode, detail = er.classify_event_anchor_episode(
        rel,
        {event.id: event},
        {ev.id: ev},
        (query,),
        subject_relevance=tag,
    )
    admitted = relation_admission_allowed(
        rel, {event.id: event}, {ev.id: ev}, (query,), rule=rel.rule,
    )
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        [ev],
        event_id="g6am",
        name="Ariston",
        period="unknown",
        query_contexts=(query,),
    )
    route_points = len(outcome.route.ordered_points) if outcome.route else 0
    return tag, episode, detail, admitted, route_points


def execute_route(text: str, *, query: str) -> tuple[AgentState, dict, dict]:
    state = AgentState(
        session_id="g6am",
        user_query=query,
        requested_output="historical_route",
    )
    tools = AgentToolRegistry(Retriever([evidence("ev1", text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6am", "name": "G6AM", "period": "100 BCE"},
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


def test_a_campaign_alpha_unsupported_not_admitted():
    assert er._query_has_campaign_episode_phrase((QUERY_ALPHA,))
    _, _, detail, admitted, route_points = classify_route(QUERY_ALPHA, MOVEMENT)
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_b_campaign_alpha_supported_may_admit():
    _, _, detail, admitted, route_points = classify_route(QUERY_ALPHA, ALPHA_MOVEMENT)
    assert er._statement_supports_query_campaign(ALPHA_MOVEMENT, (QUERY_ALPHA,))
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_c_sicilian_war_unsupported_not_admitted():
    assert er._query_has_campaign_episode_phrase((QUERY_SICILIAN,))
    _, _, detail, admitted, route_points = classify_route(QUERY_SICILIAN, MOVEMENT)
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_d_sicilian_war_supported_may_admit():
    _, _, detail, admitted, route_points = classify_route(QUERY_SICILIAN, SICILIAN_MOVEMENT)
    assert er._statement_supports_query_campaign(SICILIAN_MOVEMENT, (QUERY_SICILIAN,))
    assert detail["admitted"] is True
    assert route_points >= 2


def test_e_reverse_endpoint_not_admitted():
    _, _, detail, admitted, route_points = classify_endpoint_route(
        QUERY_ENDPOINT,
        REVERSE_MOVEMENT,
        origin="Capua",
        destination="Roma",
    )
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_f_exact_endpoint_positive():
    _, _, detail, admitted, route_points = classify_route(QUERY_ENDPOINT, MOVEMENT)
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_g_same_origin_wrong_destination_not_admitted():
    _, _, detail, admitted, route_points = classify_endpoint_route(
        QUERY_ENDPOINT,
        SAME_ORIGIN,
        origin="Roma",
        destination="Brundisium",
    )
    assert detail["admitted"] is False
    assert route_points < 2


def test_h_wrong_origin_same_destination_not_admitted():
    _, _, detail, admitted, route_points = classify_endpoint_route(
        QUERY_ENDPOINT,
        SAME_DEST,
        origin="Brundisium",
        destination="Capua",
    )
    assert detail["admitted"] is False
    assert route_points < 2


def test_disjoint_endpoint_not_admitted():
    _, _, detail, admitted, route_points = classify_endpoint_route(
        QUERY_ENDPOINT,
        DISJOINT,
        origin="Athens",
        destination="Sparta",
    )
    assert detail["admitted"] is False
    assert route_points < 2


def test_i_legacy_campaign_bypass_blocked():
    state, result, meta = execute_route(FALLBACK_NO_ALPHA, query=QUERY_ALPHA)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []
    assert meta["trace"].get("legacy", {}).get("activated") in {True, False}


def test_j_legacy_reverse_endpoint_bypass_blocked():
    state, result, meta = execute_route(FALLBACK_REVERSE, query=QUERY_ENDPOINT)
    assert ("Roma", "Capua") not in route_edge_pairs(state)
    assert result["result"]["route"] is None or route_edge_pairs(state) != [("Roma", "Capua")]


def test_campaign_year_conjunction_missing_year():
    _, _, detail, admitted, route_points = classify_route(QUERY_ALPHA_YEAR, ALPHA_MOVEMENT)
    assert detail["admitted"] is False
    assert route_points < 2


def test_campaign_year_conjunction_missing_campaign():
    _, _, detail, admitted, route_points = classify_route(QUERY_ALPHA_YEAR, YEAR_ONLY_MOVEMENT)
    assert detail["admitted"] is False
    assert route_points < 2


def test_northern_campaign_unsupported_still_rejected():
    _, _, detail, admitted, route_points = classify_route(QUERY_NORTHERN, MOVEMENT)
    assert detail["admitted"] is False
    assert route_points < 2


def test_unrestricted_subject_route_still_works():
    _, _, detail, admitted, route_points = classify_route(QUERY_UNRESTRICTED, MOVEMENT)
    assert detail["admitted"] is True
    assert route_points >= 2


def test_query_year_unknown_evidence_still_fail_closed():
    _, _, detail, admitted, route_points = classify_route(QUERY_YEAR_ONLY, MOVEMENT)
    assert detail["admitted"] is False
    assert route_points < 2
