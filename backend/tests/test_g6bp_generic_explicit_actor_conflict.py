"""G6BP: generic explicit-actor conflict for route-fragment relevance."""
from __future__ import annotations

import pytest

from backend.app.rag.evidence_ranking import rerank_evidence
from backend.tests.test_g6bn_child_local_route_fragment_relevance import (
    MARCUS_NAV,
    MARCUS_QUERY,
    _common_rank,
    _evidence,
    _ranking,
    production_retriever,
)

DEMETRIUS_QUERY = "Trace Demetrius Poliorcetes from Syria across the sea into Cilicia."
DEMETRIUS_NAV = {"navigation_path_json": '["DEMETRIUS"]', "heading": "DEMETRIUS"}
ARISTON_QUERY = "Trace Ariston from Rome toward Capua across the Apennines."
ARISTON_NAV = {"navigation_path_json": '["ARISTON"]', "heading": "ARISTON"}


def test_wrong_actor_at_sentence_start_blocks_fragment():
    score = _ranking(MARCUS_QUERY, "Bion crossed the sea toward Corcyra.")
    assert score["route_fragment_relevance"] == 0.0


def test_wrong_actor_after_clause_blocks_fragment():
    text = "After the preparations, Bion put to sea, crossed the Ionian Sea, and landed near Oricum."
    score = _ranking(MARCUS_QUERY, text)
    assert score["route_fragment_relevance"] == 0.0


def test_unseen_names_demtrius_seleucus_conflict():
    text = "Seleucus put to sea, crossed into Cilicia, and landed near Tarsus."
    score = _ranking(DEMETRIUS_QUERY, text, metadata={**DEMETRIUS_NAV})
    assert score["route_fragment_relevance"] == 0.0


def test_unseen_names_ariston_bion_conflict():
    text = "After mustering troops, Bion put to sea, crossed from Rome, and marched toward Capua."
    score = _ranking(ARISTON_QUERY, text, metadata={**ARISTON_NAV})
    assert score["route_fragment_relevance"] == 0.0


def test_implicit_pronoun_fragment_still_allowed():
    from backend.app.rag.evidence_ranking import _explicit_fragment_actor_conflict
    from backend.app.rag.query_roles import analyze_query

    roles = analyze_query(MARCUS_QUERY)
    assert not _explicit_fragment_actor_conflict(roles, "He crossed the sea toward Corcyra.")
    score = _ranking(MARCUS_QUERY, "He put to sea in winter, crossed the Ionian Sea, and landed near Oricum.")
    assert score["route_fragment_relevance"] > 0.0


def test_explicit_correct_actor_preserves_normal_relevance():
    text = "Marcus Valerius marched from Italy across the Adriatic into Epirus."
    score = _ranking(MARCUS_QUERY, text)
    assert score["route_fragment_relevance"] == 0.0
    assert score["final_score"] >= 1.0


def test_generic_movement_without_route_context_still_no_fragment():
    score = _ranking(MARCUS_QUERY, "He travelled quickly with the army.")
    assert score["route_fragment_relevance"] == 0.0


@pytest.mark.integration
def test_caesar_002_fragment_gain_preserved(production_retriever):
    result = _common_rank(production_retriever, "g5r-caesar-002")
    assert result["ranking"]["route_fragment_relevance"] > 0
    assert result["rank"] <= 800


@pytest.mark.integration
def test_alexander_002_behavior_preserved(production_retriever):
    result = _common_rank(production_retriever, "g5r-alexander-002")
    assert result["ranking"]["route_fragment_relevance"] > 0
    assert result["rank"] <= 384
    assert result["score"] >= 0.82


@pytest.mark.integration
def test_existing_final_controls_preserved(production_retriever):
    for benchmark_id in ("g5r-pompey-001", "g5r-mithridates-001", "g5r-lucullus-002"):
        assert _common_rank(production_retriever, benchmark_id)["final"] is True
