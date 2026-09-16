"""V1.1B1.4.1: allow validated single-person route-of context."""
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
    "Trace the route of Asia.",
    "Trace the route of Egypt.",
    "Trace the route of the Senate.",
    "Trace the route of the legion.",
    "Trace the route of the fleet.",
    "Trace Battle of Pharsalus to Egypt.",
    "Show the road from Rome to Capua.",
    "Follow the road from Rome.",
    "Trace the army from Rome to Capua.",
    "Trace the campaign from Pontus to Armenia.",
]

POSITIVE = [
    ("Trace the route of Lucullus.", "Lucullus"),
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


def test_route_of_lucullus_alone_resolves() -> None:
    assert _context("Trace the route of Lucullus.") == "Lucullus"


@pytest.mark.parametrize("query", NEGATIVE)
def test_non_person_route_of_and_scaffold_queries_rejected(query: str) -> None:
    assert _context(query) == ""


@pytest.mark.parametrize(("query", "expected"), POSITIVE)
def test_person_context_positive_controls(query: str, expected: str) -> None:
    assert _context(query) == expected
    assert "SUBJECT" not in {intent.kind for intent in decompose_movement_query(query)}


@pytest.mark.integration
def test_mithridates_trusted_evidence_still_competitive(production_retriever):
    ref = _ref("g5r-mithridates-001")
    trace = _trace_mithridates_ref(production_retriever, ref)
    assert trace["episode_query"] == "Mithridates war Pontus Anatolia"
    assert trace["proposal_present"] is True
    assert trace["union_present"] is True
    assert trace["final_present"] is True
