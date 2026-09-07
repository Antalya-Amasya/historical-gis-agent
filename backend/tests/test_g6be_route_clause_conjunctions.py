"""G6BE: route-clause-aware conjunction handling in query role extraction."""
from __future__ import annotations

from backend.app.rag.query_roles import analyze_query

CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)
ALEXANDER_QUERY = (
    "Reconstruct Alexander the Great's route from Bactria through the Hindu Kush into the "
    "Indian campaign, ending near the Hydaspes."
)
MITHRIDATES_QUERY = (
    "Reconstruct the major movements of Mithridates VI during the First Mithridatic War, "
    "from Pontus through Asia Minor and Greece, and back toward Anatolia."
)
LUCULLUS_ACTUAL_QUERY = (
    "Trace Lucullus's campaign movements against Mithridates from Pontus through "
    "Armenia and into Asia Minor."
)


def test_a_lucullus_and_pompey_both_person():
    roles = analyze_query("Trace Lucullus and Pompey from Rome to Greece.")
    assert {"lucullus", "pompey"} <= roles.person_terms
    assert {"rome", "greece"} <= roles.location_terms
    assert "pompey" not in roles.location_terms


def test_b_alexander_and_ptolemy_both_person():
    roles = analyze_query("Follow Alexander and Ptolemy from Pella into Egypt.")
    assert {"alexander", "ptolemy"} <= roles.person_terms
    assert {"pella", "egypt"} <= roles.location_terms
    assert "ptolemy" not in roles.location_terms


def test_c_marcus_valerius_and_lucius_cornelius_not_location():
    roles = analyze_query("Trace Marcus Valerius from Rome and Lucius Cornelius from Capua.")
    assert {"marcus", "valerius", "lucius", "cornelius"} <= roles.person_terms
    assert {"rome", "capua"} <= roles.location_terms
    assert "lucius" not in roles.location_terms
    assert "cornelius" not in roles.location_terms


def test_d_lucullus_and_into_asia_minor():
    roles = analyze_query(LUCULLUS_ACTUAL_QUERY)
    assert {"lucullus", "mithridates"} <= roles.person_terms
    assert {"pontus", "armenia", "asia", "minor"} <= roles.location_terms
    assert "asia" not in roles.person_terms
    assert "minor" not in roles.person_terms


def test_e_valid_geographic_coordination():
    roles = analyze_query("Trace Ariston from Rome through Capua and Brundisium into Epirus.")
    assert roles.person_terms == frozenset({"ariston"})
    assert {"rome", "capua", "brundisium", "epirus"} <= roles.location_terms


def test_f_explicit_and_into_continues_route():
    roles = analyze_query("Trace Ariston from Rome through Armenia and into Asia Minor.")
    assert roles.person_terms == frozenset({"ariston"})
    assert {"rome", "armenia", "asia", "minor"} <= roles.location_terms


def test_g_subject_coordination_outside_route_clause():
    roles = analyze_query("Trace Ariston and Bion during the campaign.")
    assert {"ariston", "bion"} <= roles.person_terms
    assert "bion" not in roles.location_terms


def test_h_caesar_g6bc_route_roles_regression():
    roles = analyze_query(CAESAR_QUERY)
    assert roles.person_terms == frozenset({"julius", "caesar"})
    assert {"italy", "adriatic", "epirus", "pharsalus"} <= roles.location_terms


def test_i_alexander_g6bc_route_roles_regression():
    roles = analyze_query(ALEXANDER_QUERY)
    assert roles.person_terms == frozenset({"alexander"})
    assert {"bactria", "hindu", "kush", "hydaspes", "indian"} <= roles.location_terms


def test_j_mithridates_g6bc_route_roles_regression():
    roles = analyze_query(MITHRIDATES_QUERY)
    assert {"mithridates", "vi"} <= roles.person_terms
    assert {"pontus", "asia", "minor", "greece", "anatolia"} <= roles.location_terms


def test_simple_destination_coordination():
    roles = analyze_query("Trace Ariston from Rome to Capua and Brundisium.")
    assert roles.person_terms == frozenset({"ariston"})
    assert {"rome", "capua", "brundisium"} <= roles.location_terms
