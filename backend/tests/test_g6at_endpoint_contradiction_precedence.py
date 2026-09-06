"""G6AT: explicit endpoint contradiction vetoes event-anchor admission."""

from __future__ import annotations

from backend.tests.test_g6aq_directed_fragment_chain_authority import (
    component_chains,
    execute_route,
    route_edge_pairs,
)


def test_a_s704_reverse_fragment_has_no_route():
    state, result, diagnostics = execute_route(
        "In 200 BCE Ariston marched from Brundisium to Rome.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert result["result"]["route"] is None
    assert diagnostics["diagnostics"]["route_source"] != "event_anchor"


def test_b_forward_exact_movement_remains_valid():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Rome to Capua.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert result["result"]["route"] is not None


def test_c_reverse_exact_movement_is_rejected():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Capua to Rome.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert result["result"]["route"] is None


def test_d_same_origin_wrong_destination_is_rejected():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Rome to Brundisium.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert result["result"]["route"] is None


def test_e_same_destination_wrong_origin_is_rejected():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Brundisium to Capua.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert result["result"]["route"] is None


def test_f_qualified_two_leg_chain_is_preserved():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Rome to Brundisium. "
        "In 200 BCE Ariston marched from Brundisium to Capua.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert result["result"]["route"] is not None
    assert component_chains(state)


def test_g_legacy_fallback_does_not_reintroduce_reverse_fragment():
    state, result, _ = execute_route(
        "In 200 BCE Ariston marched from Brundisium to Rome, then sailed from Corcyra to Capua.",
        query="Trace Ariston from Rome to Capua in 200 BCE.",
    )
    assert ("Brundisium", "Roma") not in route_edge_pairs(state)
