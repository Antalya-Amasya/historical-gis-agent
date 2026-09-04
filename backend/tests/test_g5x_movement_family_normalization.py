"""G5X: movement-family inflection and generic route expansion."""
from __future__ import annotations

import pytest

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import decompose_movement_query
from backend.tests.g5r_trusted_benchmark import load_trusted_benchmark

HIGH_IDS = ("g5r-caesar-002", "g5r-pompey-001", "g5r-lucullus-001")
SIDE_ID = "g5r-mithridates-001"
G5W_RANKS = {
    "g5r-caesar-002": None,
    "g5r-pompey-001": None,
    "g5r-lucullus-001": None,
    "g5r-mithridates-001": 21,
}


def _family_available(query: str, expected: set[str]) -> None:
    roles = analyze_query(query)
    available = roles.action_terms | roles.expanded_action_terms | roles.movement_inflection_terms
    assert expected <= available, (
        f"query={query!r} got action={roles.action_terms} "
        f"inflection={roles.movement_inflection_terms} expanded={roles.expanded_action_terms}"
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Caesar march into Epirus", {"march", "marched", "marching"}),
        ("Pompey sailed to Egypt", {"sail", "sailed", "sailing"}),
        ("Lucullus cross into Armenia", {"cross", "crossed", "crossing"}),
        ("army passing the river", {"pass", "passed", "passing"}),
        ("they fled overnight", {"flee", "fled", "fleeing", "fly"}),
        ("he came over into Armenia", {"come", "came", "coming"}),
        ("Pompey left Greece", {"leave", "left", "leaving"}),
        ("Caesar went to Rome", {"go", "went", "going"}),
    ],
    ids=["march", "sail", "cross", "pass", "flee", "come", "leave", "go"],
)
def test_movement_family_inflection_available(query: str, expected: set[str]):
    _family_available(query, expected)


def test_route_query_triggers_generic_movement_expansion():
    roles = analyze_query("Trace Pompey's movements after Pharsalus")
    expanded = roles.expanded_action_terms
    assert {"march", "cross", "sail", "depart", "left", "arrive", "fled"} & expanded


def test_non_movement_query_does_not_trigger_generic_expansion():
    roles = analyze_query("Discuss Caesar's political reforms")
    assert not (roles.expanded_action_terms & {"marched", "sailed", "crossed", "departed"})


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from backend.app.core.config import settings
        from backend.app.rag.http_store import build_production_retriever

        retriever = build_production_retriever(settings)
        retriever.retrieve("Rome", 1)
        return retriever
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"production chroma unavailable: {exc}")


def _ref(bid: str) -> dict:
    return next(r for r in load_trusted_benchmark()["references"] if r["benchmark_id"] == bid)


def _query_for(bid: str) -> str:
    ref = _ref(bid)
    return next(q for q in load_trusted_benchmark()["queries"] if q["query_id"] == ref["query_id"])["query"]


def _best_lexical_rank(store, bid: str, top_k: int = 50) -> int | None:
    ref = _ref(bid)
    canonical = _query_for(bid)
    best = None
    for query in [canonical] + [i.query for i in decompose_movement_query(canonical)]:
        items = store.lexical_candidates(query, top_k)
        rank = next((i for i, c in enumerate(items, 1) if c.id == ref["evidence_id"]), None)
        if rank is not None and (best is None or rank < best):
            best = rank
    return best


@pytest.mark.integration
def test_g5x_high_lexical_recall(production_retriever):
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    ranks = {bid: _best_lexical_rank(store, bid, 20) for bid in HIGH_IDS}
    hits = sum(1 for bid in HIGH_IDS if ranks[bid] is not None)
    assert ranks["g5r-pompey-001"] is not None, f"Pompey should reach lexical top-20, ranks={ranks}"
    assert hits >= 1, f"expected >=1/3 HIGH lexical top-20, ranks={ranks} (G5W={G5W_RANKS})"


@pytest.mark.integration
def test_g5x_mithridates_side_lexical_rank(production_retriever):
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    rank = _best_lexical_rank(store, SIDE_ID, 50)
    assert rank is not None and rank <= G5W_RANKS[SIDE_ID], f"mithridates rank={rank}, G5W={G5W_RANKS[SIDE_ID]}"
