"""G6AW: proven qualified chain members are admitted as route fragments."""

from __future__ import annotations

import backend.app.routes.episode_relevance as er
from backend.tests.test_g6ag_query_subject_scope_normalization import COORDS
from backend.tests.test_g6aq_directed_fragment_chain_authority import (
    component_chains,
    execute_route,
    route_edge_pairs,
)
from backend.tests.test_g6as_occurrence_qualified_chain_witness import classify_legs

COORDS["Corcyra"] = (39.6, 19.9)

QUERY = "Trace Ariston from Rome to Capua."
THREE_LEGS = (
    "Ariston marched from Rome to Brundisium. "
    "Ariston marched from Brundisium to Corcyra. "
    "Ariston marched from Corcyra to Capua."
)


def test_a_proven_three_leg_chain_is_admitted_in_production():
    legs = [
        ("Ariston marched from Rome to Brundisium.", "Roma", "Brundisium"),
        ("Ariston marched from Brundisium to Corcyra.", "Brundisium", "Corcyra"),
        ("Ariston marched from Corcyra to Capua.", "Corcyra", "Capua"),
    ]
    classified = classify_legs(QUERY, legs)
    assert all(detail["admitted"] and admitted for _, detail, admitted in classified)
    state, result, _ = execute_route(THREE_LEGS, query=QUERY)
    assert result["result"]["route"] is not None
    assert len(component_chains(state)) >= 3


def test_b_proven_two_leg_chain_remains_admitted():
    state, result, _ = execute_route(
        "Ariston marched from Rome to Brundisium. Ariston marched from Brundisium to Capua.",
        query=QUERY,
    )
    assert result["result"]["route"] is not None


def test_c_isolated_internal_edge_is_not_admitted():
    state, result, _ = execute_route("Ariston marched from Brundisium to Corcyra.", query=QUERY)
    assert result["result"]["route"] is None


def test_d_wrong_person_middle_leg_cannot_complete_chain():
    state, result, _ = execute_route(
        "Marcus Valerius marched from Rome to Brundisium. "
        "Lucius Valerius marched from Brundisium to Corcyra. "
        "Marcus Valerius marched from Corcyra to Capua.",
        query="Trace Marcus Valerius from Rome to Capua.",
    )
    assert result["result"]["route"] is None


def test_e_wrong_campaign_middle_leg_cannot_complete_chain():
    state, result, _ = execute_route(
        "During Campaign Alpha, Ariston marched from Rome to Brundisium. "
        "During Campaign Beta, Ariston marched from Brundisium to Corcyra. "
        "During Campaign Alpha, Ariston marched from Corcyra to Capua.",
        query="Trace Ariston during Campaign Alpha from Rome to Capua.",
    )
    assert result["result"]["route"] is None


def test_f_wrong_period_middle_leg_cannot_complete_chain():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Rome to Brundisium. "
        "In 100 BCE Ariston marched from Brundisium to Corcyra. "
        "In 200 BCE Ariston marched from Corcyra to Capua.",
        query="Trace Ariston in 200 BCE from Rome to Capua.",
    )
    assert result["result"]["route"] is None


def test_g_negated_middle_leg_cannot_complete_chain():
    state, result, _ = execute_route(
        "Ariston marched from Rome to Brundisium. "
        "Ariston did not march from Brundisium to Corcyra. "
        "Ariston marched from Corcyra to Capua.",
        query=QUERY,
    )
    assert result["result"]["route"] is None


def test_h_s704_endpoint_contradiction_remains_rejected():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Brundisium to Rome.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert result["result"]["route"] is None


def test_i_unrelated_edge_is_not_admitted_as_a_chain_fragment():
    state, result, _ = execute_route(
        THREE_LEGS + " Athens marched from Athens to Sparta.", query=QUERY,
    )
    assert result["result"]["route"] is not None
    assert ("Athens", "Sparta") not in route_edge_pairs(state)


def test_j_four_leg_chain_has_no_admission_length_limit():
    text = (
        "Ariston marched from Rome to Brundisium. Ariston marched from Brundisium to Corcyra. "
        "Ariston marched from Corcyra to Athens. Ariston marched from Athens to Capua."
    )
    assert er._proven_query_chain_member("Corcyra", "Athens", text, text, (QUERY,)) is True
