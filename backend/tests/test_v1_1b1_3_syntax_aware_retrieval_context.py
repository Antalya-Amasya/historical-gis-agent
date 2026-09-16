"""V1.1B1.3: syntax-aware bounded retrieval person context."""
from __future__ import annotations

import pytest

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import (
    RetrievalIntent,
    _retrieval_person_context,
    decompose_movement_query,
    primary_route_subject,
)
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.tests.g5r_trusted_benchmark import hard_benchmark_queries, trusted_references
from backend.tests.test_v1_1b1_2_subject_authority_retrieval_context import (
    MITHRIDATES_QUERY,
    _ref,
    _trace_mithridates_ref,
)

MATRIX = [
    ("Trace Lucullus during the war against Mithridates.", "Lucullus"),
    ("Trace the route of Lucullus during the war against Mithridates.", "Lucullus"),
    ("Follow Lucullus from Pontus to Armenia.", "Lucullus"),
    ("Show Pompey fleeing after Pharsalus.", "Pompey"),
    ("Against Mithridates, trace Lucullus from Pontus.", "Lucullus"),
    ("While Mithridates pursued him, trace Pompey fleeing.", "Pompey"),
    ("Mithridates pursued Pompey; trace Pompey fleeing.", "Pompey"),
    ("Trace Pompey while Mithridates fled.", "Pompey"),
    ("Trace Mithridates while Pompey pursued him.", "Mithridates"),
    ("Pompey and Mithridates moved through Asia.", ""),
    ("Compare the routes of Pompey and Caesar.", ""),
    ("From Pontus, follow Lucullus into Armenia.", "Lucullus"),
]

SCAFFOLD_QUERIES = [
    ("Against Mithridates, trace Lucullus from Pontus.", "Against"),
    ("While Mithridates pursued him, trace Pompey fleeing.", "While"),
    ("From Pontus, follow Lucullus into Armenia.", "From"),
    ("After Pharsalus, trace Pompey toward Egypt.", "After"),
    ("Trace Lucullus during the war against Mithridates.", "During"),
]


@pytest.mark.parametrize(("query", "expected"), MATRIX)
def test_retrieval_person_context_matrix(query: str, expected: str) -> None:
    roles = analyze_query(query)
    primary = primary_route_subject(query, roles)
    context = _retrieval_person_context(query, roles, primary)
    assert primary is None or context == primary
    assert context == expected


@pytest.mark.parametrize(("query", "forbidden"), SCAFFOLD_QUERIES)
def test_scaffold_tokens_never_become_context(query: str, forbidden: str) -> None:
    roles = analyze_query(query)
    context = _retrieval_person_context(query, roles, None)
    assert context != forbidden
    assert context.casefold() != forbidden.casefold()


def test_opponent_first_clause_does_not_win_over_trace_clause() -> None:
    roles = analyze_query("Mithridates pursued Pompey; trace Pompey fleeing.")
    assert _retrieval_person_context("Mithridates pursued Pompey; trace Pompey fleeing.", roles, None) == "Pompey"


def test_mithridates_benchmark_context_and_episode() -> None:
    roles = analyze_query(MITHRIDATES_QUERY)
    context = _retrieval_person_context(MITHRIDATES_QUERY, roles, None)
    episode = next(
        intent.query for intent in decompose_movement_query(MITHRIDATES_QUERY) if intent.kind == "EPISODE"
    )
    assert context == "Mithridates"
    assert episode == "Mithridates war Pontus Anatolia"
    assert "SUBJECT" not in {intent.kind for intent in decompose_movement_query(MITHRIDATES_QUERY)}


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from backend.app.core.config import settings
        from backend.app.rag.http_store import build_production_retriever

        retriever = build_production_retriever(settings)
        retriever.retrieve("Rome", 1)
        return retriever
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"production chroma unavailable: {exc}")


@pytest.mark.integration
def test_mithridates_trusted_evidence_recovery(production_retriever):
    ref = _ref("g5r-mithridates-001")
    trace = _trace_mithridates_ref(production_retriever, ref)
    assert trace["episode_query"] == "Mithridates war Pontus Anatolia"
    assert trace["proposal_present"] is True
    assert trace["union_present"] is True
    assert trace["final_present"] is True
