"""G6AQ: fragment admission requires proven directed chain membership."""

from __future__ import annotations

import backend.app.routes.episode_relevance as er
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence, TemporalPrecision
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.episode_relevance import EpisodeRelevance, classify_event_anchor_episode
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.evidence_relevance import event_relevance, relation_admission_allowed
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g6ag_query_subject_scope_normalization import COORDS, evidence, movement_event

QUERY = "Trace Ariston from Rome to Capua."
QUERY_ALPHA = "Trace Ariston during Campaign Alpha from Rome to Capua."
QUERY_YEAR = "Trace Ariston in 200 BCE from Rome to Capua."

DIRECT = "Ariston marched from Rome to Capua."
REVERSE = "Ariston marched from Capua to Rome."
FALSE_DEST_ORIGIN = "Ariston marched from Capua to Brundisium."
FALSE_TO_ORIGIN = "Ariston marched from Brundisium to Rome."
ORIGIN_ONLY = "Ariston marched from Rome to Brundisium."
DEST_ONLY = "Ariston marched from Brundisium to Capua."
CHAIN = (
    "Ariston marched from Rome to Brundisium. "
    "Ariston then marched from Brundisium to Capua."
)
WRONG_CHAIN = (
    "Ariston marched from Capua to Brundisium. "
    "Ariston then marched from Brundisium to Rome."
)
LUCIUS_CHAIN = (
    "Lucius Valerius marched from Rome to Brundisium. "
    "Lucius Valerius then marched from Brundisium to Capua."
)
NO_CAMPAIGN_CHAIN = CHAIN
CAMPAIGN_CHAIN = "During Campaign Alpha, " + CHAIN
YEAR_CHAIN_200 = "In 200 BCE, " + CHAIN
YEAR_CHAIN_100 = "In 100 BCE, " + CHAIN
LEG1 = "Ariston marched from Rome to Brundisium."
LEG2 = "Ariston then marched from Brundisium to Capua."
LEG1_200 = "In 200 BCE Ariston marched from Rome to Brundisium."
LEG2_100 = "In 100 BCE Ariston then marched from Brundisium to Capua."


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
    for name, coords in Geography.places.items():
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
    tag = event_relevance(event, {event.id: event}, (query,))
    episode, detail = classify_event_anchor_episode(
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
        event_id="g6aq",
        name="Ariston",
        period="unknown",
        query_contexts=(query,),
    )
    route_points = len(outcome.route.ordered_points) if outcome.route else 0
    return tag, episode, detail, admitted, route_points


def execute_route(text: str, *, query: str) -> tuple[AgentState, dict, dict]:
    state = AgentState(
        session_id="g6aq",
        user_query=query,
        requested_output="historical_route",
    )
    tools = AgentToolRegistry(Retriever([evidence("ev1", text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6aq", "name": "G6AQ", "period": "100 BCE"},
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


def component_chains(state: AgentState) -> list[list[str]]:
    if state.historical_route is None:
        return []
    return [
        [point.historical_place.canonical_name for point in component.ordered_points]
        for component in state.historical_route.route_components
    ]


def classify_chain_legs(query: str, full_text: str):
    COORDS["Brundisium"] = Geography.places["Brundisium"]
    e1 = movement_event("e1", LEG1, "Roma", "Brundisium")
    e2 = movement_event("e2", LEG2, "Brundisium", "Capua")
    ev = evidence("ev1", full_text)
    events = {e1.id: e1, e2.id: e2}
    results = []
    for event, statement, origin, destination in (
        (e1, LEG1, "Roma", "Brundisium"),
        (e2, LEG2, "Brundisium", "Capua"),
    ):
        rel = relation(origin, destination, OrderingRule.SAME_MOVEMENT_EVENT, refs=("ev1",), event_ids=(event.id,))
        tag = event_relevance(event, events, (query,))
        episode, detail = classify_event_anchor_episode(
            rel, events, {ev.id: ev}, (query,), subject_relevance=tag,
        )
        admitted = relation_admission_allowed(rel, events, {ev.id: ev}, (query,), rule=rel.rule)
        results.append((episode, detail, admitted))
    return results


def assert_not_admitted(query: str, statement: str, *, origin: str, destination: str):
    alignment = er._explicit_endpoint_alignment(origin, destination, (query,))
    _, episode, detail, admitted, route_points = classify_endpoint_route(
        query, statement, origin=origin, destination=destination,
    )
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2
    return alignment


def test_a_false_destination_origin_fragment_not_admitted():
    alignment = assert_not_admitted(
        QUERY, FALSE_DEST_ORIGIN, origin="Capua", destination="Brundisium",
    )
    assert alignment == "fragment"
    state, result, _ = execute_route(FALSE_DEST_ORIGIN, query=QUERY)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []
    assert ("Roma", "Capua") not in route_edge_pairs(state)


def test_b_false_destination_to_origin_fragment_not_admitted():
    alignment = assert_not_admitted(
        QUERY, FALSE_TO_ORIGIN, origin="Brundisium", destination="Roma",
    )
    assert alignment == "fragment"
    state, result, _ = execute_route(FALSE_TO_ORIGIN, query=QUERY)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []


def test_c_valid_two_leg_chain_retained():
    (ep1, det1, adm1), (ep2, det2, adm2) = classify_chain_legs(QUERY, CHAIN)
    assert ep1 is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert ep2 is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert det1["admitted"] is True
    assert det2["admitted"] is True
    assert adm1 is True
    assert adm2 is True
    state, result, _ = execute_route(CHAIN, query=QUERY)
    assert result["result"]["route"] is not None
    chains = component_chains(state)
    assert ["Roma", "Brundisium"] in chains
    assert ["Brundisium", "Capua"] in chains


def test_d_direct_exact_route_works():
    _, _, detail, admitted, route_points = classify_endpoint_route(
        QUERY, DIRECT, origin="Roma", destination="Capua",
    )
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_e_reverse_route_rejected():
    assert_not_admitted(QUERY, REVERSE, origin="Capua", destination="Roma")


def test_f_unproven_origin_fragment_not_admitted():
    assert_not_admitted(QUERY, ORIGIN_ONLY, origin="Roma", destination="Brundisium")


def test_g_unproven_destination_fragment_not_admitted():
    assert_not_admitted(QUERY, DEST_ONLY, origin="Brundisium", destination="Capua")


def test_h_wrong_person_valid_shaped_chain_rejected():
    state, result, _ = execute_route(LUCIUS_CHAIN, query=QUERY)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []
    assert ("Roma", "Capua") not in route_edge_pairs(state)


def test_i_wrong_period_participant_chain_rejected():
    (ep1, det1, _), (ep2, det2, _) = classify_chain_legs(
        QUERY_YEAR,
        LEG1_200 + " " + LEG2_100,
    )
    assert det1["admitted"] is False or det2["admitted"] is False
    state, result, _ = execute_route(LEG1_200 + " " + LEG2_100, query=QUERY_YEAR)
    assert result["result"]["route"] is None or ("Roma", "Capua") not in route_edge_pairs(state)


def test_j_wrong_direction_chain_not_treated_as_query_chain():
    assert_not_admitted(QUERY, WRONG_CHAIN, origin="Capua", destination="Brundisium")
    state, result, _ = execute_route(WRONG_CHAIN, query=QUERY)
    assert ("Roma", "Capua") not in route_edge_pairs(state)


def test_k_campaign_constraint_blocks_chain_without_support():
    state, result, _ = execute_route(NO_CAMPAIGN_CHAIN, query=QUERY_ALPHA)
    assert result["result"]["route"] is None or ("Roma", "Capua") not in route_edge_pairs(state)


def test_l_legacy_fallback_cannot_return_unproven_fragment():
    legacy_text = (
        "Ariston did not march from Rome to Capua, "
        "but marched from Capua to Brundisium."
    )
    state, result, _ = execute_route(legacy_text, query=QUERY)
    assert ("Roma", "Capua") not in route_edge_pairs(state)
    assert result["result"]["route"] is None or route_edge_pairs(state) == []
