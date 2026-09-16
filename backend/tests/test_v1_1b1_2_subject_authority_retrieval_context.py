"""V1.1B1.2: separate SUBJECT authority from retrieval person context."""
from __future__ import annotations

import pytest

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import (
    RetrievalIntent,
    _retrieval_person_context,
    _subject_phrase,
    decompose_movement_query,
    primary_route_subject,
)
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.tests.g5r_trusted_benchmark import hard_benchmark_queries, trusted_references

MITHRIDATES_QUERY = (
    "Reconstruct the major movements of Mithridates VI during the First Mithridatic War, "
    "from Pontus through Asia Minor and Greece, and back toward Anatolia."
)

POSSESSIVE_MATRIX = [
    (
        "Trace Pompey's route after the Battle of Pharsalus to Egypt.",
        "Pompey",
        frozenset({"mithridates"}),
    ),
    (
        "Trace Lucullus's campaign movements against Mithridates from Pontus to Armenia.",
        "Lucullus",
        frozenset({"mithridates"}),
    ),
    (
        "Trace Alexander's march against Porus toward the Hydaspes.",
        "Alexander",
        frozenset({"porus"}),
    ),
    (
        "Trace Sulla's campaign against Mithridates in Greece.",
        "Sulla",
        frozenset({"mithridates"}),
    ),
    (
        "Trace Commander Alpha's campaign against Commander Beta from Gamma Harbor.",
        "Commander Alpha",
        frozenset({"beta"}),
    ),
]

NON_POSSESSIVE_MATRIX = [
    (MITHRIDATES_QUERY, None, "Mithridates"),
    ("Trace Lucullus during the war against Mithridates.", None, "Lucullus"),
    (
        "Trace the route of Lucullus during the war against Mithridates.",
        None,
        "Lucullus",
    ),
    ("Follow Lucullus from Pontus to Armenia.", None, "Lucullus"),
    ("Show Pompey fleeing after Pharsalus.", None, "Pompey"),
]


def _intent_map(query: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for intent in decompose_movement_query(query):
        grouped.setdefault(intent.kind, []).append(intent.query)
    return grouped


def _context_movement(grouped: dict[str, list[str]], context: str) -> str | None:
    matches = [q for q in grouped.get("MOVEMENT", []) if q.startswith(context)]
    return matches[0] if len(matches) == 1 else None


def _ref(benchmark_id: str) -> dict:
    return next(ref for ref in trusted_references() if ref["benchmark_id"] == benchmark_id)


def _trace_mithridates_ref(retriever: ChromaHistoricalRetriever, ref: dict) -> dict[str, object]:
    query = next(
        item["query"]
        for item in hard_benchmark_queries()
        if item["query_id"] == ref["query_id"]
    )
    target_id = ref["evidence_id"]
    channels = [RetrievalIntent("CANONICAL", query), *decompose_movement_query(query)]
    proposal_present = False
    union_ids: set[str] = set()
    best_episode_lexical_rank: int | None = None

    for intent in channels:
        candidates = retriever.retrieve_candidates(intent.query, 60)
        union_ids.update(item.id for item in candidates)
        if target_id in union_ids:
            proposal_present = True
        if intent.kind == "EPISODE" and target_id in {item.id for item in candidates}:
            lexical_items = sorted(
                (item for item in candidates if item.metadata.get("lexical_candidate")),
                key=lambda item: (-float(item.metadata.get("lexical_score") or 0.0), item.id),
            )
            for rank, item in enumerate(lexical_items, start=1):
                if item.id == target_id:
                    best_episode_lexical_rank = rank
                    break

    final = retriever.retrieve_with_coverage(query)
    return {
        "proposal_present": proposal_present,
        "union_present": target_id in union_ids,
        "final_present": target_id in {item.id for item in final},
        "best_episode_lexical_rank": best_episode_lexical_rank,
        "episode_query": next(
            (intent.query for intent in decompose_movement_query(query) if intent.kind == "EPISODE"),
            None,
        ),
    }


@pytest.mark.parametrize(("query", "expected_primary", "opponents"), POSSESSIVE_MATRIX)
def test_possessive_subject_and_context_match(
    query: str,
    expected_primary: str,
    opponents: frozenset[str],
) -> None:
    roles = analyze_query(query)
    primary = primary_route_subject(query, roles)
    context = _retrieval_person_context(query, roles, primary)
    grouped = _intent_map(query)

    assert primary == expected_primary
    assert context == expected_primary
    assert grouped.get("SUBJECT") == [expected_primary]
    assert grouped["EPISODE"][0].startswith(expected_primary)

    movement = _context_movement(grouped, expected_primary)
    assert movement is not None
    assert grouped["ENDPOINT"][0].startswith(expected_primary)

    for opponent in opponents:
        assert opponent.casefold() not in {
            token.casefold() for token in grouped["EPISODE"][0].split()
        }


@pytest.mark.parametrize(("query", "expected_primary", "expected_context"), NON_POSSESSIVE_MATRIX)
def test_non_possessive_has_no_subject_but_may_have_context(
    query: str,
    expected_primary: str | None,
    expected_context: str,
) -> None:
    roles = analyze_query(query)
    primary = primary_route_subject(query, roles)
    context = _retrieval_person_context(query, roles, primary)
    grouped = _intent_map(query)

    assert primary == expected_primary
    assert context == expected_context
    assert "SUBJECT" not in grouped


def test_mithridates_episode_restores_person_context_without_subject() -> None:
    roles = analyze_query(MITHRIDATES_QUERY)
    primary = primary_route_subject(MITHRIDATES_QUERY, roles)
    context = _retrieval_person_context(MITHRIDATES_QUERY, roles, primary)
    grouped = _intent_map(MITHRIDATES_QUERY)

    assert primary is None
    assert context == "Mithridates"
    assert "SUBJECT" not in grouped
    assert grouped["EPISODE"][0].startswith("Mithridates")
    assert "war" in grouped["EPISODE"][0].casefold()
    assert "Pontus" in grouped["EPISODE"][0] or "pontus" in grouped["EPISODE"][0].casefold()


def test_ambiguous_lucullus_does_not_publish_contaminated_bag() -> None:
    query = "Trace Lucullus during the war against Mithridates."
    roles = analyze_query(query)
    assert _subject_phrase(query, roles) == "Lucullus Mithridates"
    assert _retrieval_person_context(query, roles, None) == "Lucullus"
    grouped = _intent_map(query)
    assert "SUBJECT" not in grouped
    assert "Lucullus Mithridates" not in grouped.get("EPISODE", [""])[0]


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
def test_mithridates_trusted_evidence_returns_to_proposal_competition(production_retriever):
    """B1.1 dropped g5r-mithridates-001 from proposal/union/final without person EPISODE context."""
    ref = _ref("g5r-mithridates-001")
    trace = _trace_mithridates_ref(production_retriever, ref)
    assert trace["episode_query"] == "Mithridates war Pontus Anatolia"
    assert trace["proposal_present"] is True
    assert trace["union_present"] is True
    assert trace["final_present"] is True
    rank = trace["best_episode_lexical_rank"]
    assert rank is not None and rank <= 40
