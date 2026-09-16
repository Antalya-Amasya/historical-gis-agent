"""V1.1B1.4: validate grammar-captured retrieval person context against query roles."""
from __future__ import annotations

import pytest

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import (
    _retrieval_person_context,
    decompose_movement_query,
    primary_route_subject,
)
from backend.tests.test_v1_1b1_2_subject_authority_retrieval_context import (
    MITHRIDATES_QUERY,
    _ref,
    _trace_mithridates_ref,
)
from backend.tests.test_v1_1b1_3_syntax_aware_retrieval_context import production_retriever

NEGATIVE = [
    "Trace the route of Rome to Capua.",
    "Trace Battle of Pharsalus to Egypt.",
    "Show the road from Rome to Capua.",
    "Follow the road from Rome.",
    "Trace the army from Rome to Capua.",
    "Trace the campaign from Pontus to Armenia.",
]

POSITIVE = [
    ("Trace Lucullus during the war against Mithridates.", "Lucullus"),
    ("Trace the route of Lucullus during the war against Mithridates.", "Lucullus"),
    ("Follow Lucullus from Pontus to Armenia.", "Lucullus"),
    ("Show Pompey fleeing after Pharsalus.", "Pompey"),
    ("Movements of Mithridates in Pontus.", "Mithridates"),
    ("Against Mithridates, trace Lucullus from Pontus.", "Lucullus"),
    ("While Mithridates pursued him, trace Pompey fleeing.", "Pompey"),
]


def _context(query: str) -> str:
    roles = analyze_query(query)
    return _retrieval_person_context(query, roles, primary_route_subject(query, roles))


@pytest.mark.parametrize("query", NEGATIVE)
def test_non_person_grammar_captures_rejected(query: str) -> None:
    assert _context(query) == ""


@pytest.mark.parametrize(("query", "expected"), POSITIVE)
def test_person_grammar_captures_retained(query: str, expected: str) -> None:
    assert _context(query) == expected
    assert "SUBJECT" not in {intent.kind for intent in decompose_movement_query(query)}


def test_mithridates_benchmark_context_and_episode() -> None:
    roles = analyze_query(MITHRIDATES_QUERY)
    context = _retrieval_person_context(MITHRIDATES_QUERY, roles, None)
    episode = next(
        intent.query for intent in decompose_movement_query(MITHRIDATES_QUERY) if intent.kind == "EPISODE"
    )
    assert context == "Mithridates"
    assert episode == "Mithridates war Pontus Anatolia"


@pytest.mark.integration
def test_mithridates_trusted_evidence_still_competitive(production_retriever):
    ref = _ref("g5r-mithridates-001")
    trace = _trace_mithridates_ref(production_retriever, ref)
    assert trace["episode_query"] == "Mithridates war Pontus Anatolia"
    assert trace["proposal_present"] is True
    assert trace["union_present"] is True
    assert trace["final_present"] is True
