"""G6BC: common rerank route-role extraction corrections."""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.query_roles import analyze_query, location_support, person_support

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
POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through the "
    "eastern Mediterranean until his arrival in Egypt in 48 BCE."
)


def _route_locations(roles) -> frozenset[str]:
    return roles.location_terms


def test_a_synthetic_route_syntax_roles():
    roles = analyze_query("Trace Marcus Valerius from Rome through Brundisium into Epirus.")
    assert roles.person_terms == frozenset({"marcus", "valerius"})
    assert {"rome", "brundisium", "epirus"} <= _route_locations(roles)
    assert "trace" not in roles.person_terms


def test_b_synthetic_multi_token_geography_roles():
    roles = analyze_query("Reconstruct Ariston from Asia Minor through the Hindu Kush.")
    assert roles.person_terms == frozenset({"ariston"})
    assert {"asia", "minor", "hindu", "kush"} <= _route_locations(roles)
    assert "reconstruct" not in roles.person_terms
    assert "kush" not in roles.person_terms


def test_c_synthetic_time_scaffolding_not_person():
    roles = analyze_query("Trace Ariston from Rome to Capua in 200 BCE.")
    assert roles.person_terms == frozenset({"ariston"})
    assert {"rome", "capua"} <= _route_locations(roles)
    assert "200" not in roles.person_terms
    assert "bce" not in roles.person_terms


def test_d_synthetic_campaign_context_not_person():
    roles = analyze_query("Trace Ariston from Pontus to Greece during the First War.")
    assert roles.person_terms == frozenset({"ariston"})
    assert {"pontus", "greece"} <= _route_locations(roles)
    assert "war" not in roles.person_terms


def test_caesar_route_roles():
    roles = analyze_query(CAESAR_QUERY)
    assert roles.person_terms == frozenset({"julius", "caesar"})
    assert {"italy", "adriatic", "epirus", "pharsalus"} <= _route_locations(roles)
    assert not (roles.person_terms & {"trace", "bce", "pharsalus"})


def test_alexander_route_roles():
    roles = analyze_query(ALEXANDER_QUERY)
    assert roles.person_terms == frozenset({"alexander"})
    assert {"bactria", "hindu", "kush", "hydaspes", "indian"} <= _route_locations(roles)
    assert not (roles.person_terms & {"reconstruct", "kush"})


def test_mithridates_route_roles():
    roles = analyze_query(MITHRIDATES_QUERY)
    assert {"mithridates", "vi"} <= roles.person_terms
    assert {"pontus", "asia", "minor", "greece", "anatolia"} <= _route_locations(roles)
    assert not (roles.person_terms & {"reconstruct", "war", "minor", "greece", "anatolia"})


def test_pompey_route_roles_regression():
    roles = analyze_query(POMPEY_QUERY)
    assert roles.person_terms == frozenset({"pompey"})
    assert {"greece", "egypt", "pharsalus", "mediterranean"} <= _route_locations(roles)
    assert "trace" not in roles.person_terms


def test_endpoint_alignment_distinguishes_route_segment():
    query = "Trace Marcus Valerius from Rome through Brundisium into Epirus."
    roles = analyze_query(query)
    adriatic_segment = Evidence(
        id="adriatic-seg",
        author="Author",
        work="Work",
        locator="section",
        excerpt="Valerius crossed from Brundisium toward Epirus by sea.",
        text="Valerius crossed from Brundisium toward Epirus by sea.",
        score=0.5,
        metadata={"semantic_candidate": True, "vector_rank": 5},
    )
    unrelated = Evidence(
        id="unrelated-seg",
        author="Author",
        work="Work",
        locator="section",
        excerpt="Valerius reviewed supplies in Rome before winter.",
        text="Valerius reviewed supplies in Rome before winter.",
        score=0.5,
        metadata={"semantic_candidate": True, "vector_rank": 5},
    )
    ranked = rerank_evidence(query, [unrelated, adriatic_segment], pool_relative=False)
    details = {item.id: item.metadata["retrieval_ranking"] for item in ranked}
    assert location_support(roles, set("brundisium epirus crossed".split())) > 0
    assert details["adriatic-seg"]["location_support"] >= details["unrelated-seg"]["location_support"]
    assert ranked[0].id == "adriatic-seg"


def test_caesar_adriatic_improves_child_local_alignment():
    roles = analyze_query(CAESAR_QUERY)
    passage = (
        "When the civil strife broke out into war Caesar crossed the Adriatic from Brundisium "
        "in the winter, with what forces he had, and opened his"
    )
    tokens = set(passage.casefold().split())
    assert location_support(roles, tokens) > 0
    assert person_support(roles, tokens) > 0
