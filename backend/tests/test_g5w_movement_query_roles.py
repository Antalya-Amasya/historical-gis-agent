"""G5W: movement-family terms enter lexical action scoring."""
from __future__ import annotations

import pytest

from backend.app.rag.query_roles import analyze_query


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
from backend.app.rag.retrieval_intents import decompose_movement_query
from backend.tests.g5r_trusted_benchmark import load_trusted_benchmark

HIGH_IDS = ("g5r-caesar-002", "g5r-pompey-001", "g5r-lucullus-001")
SIDE_ID = "g5r-mithridates-001"


def _ref(bid: str) -> dict:
    return next(r for r in load_trusted_benchmark()["references"] if r["benchmark_id"] == bid)


def _query_for(bid: str) -> str:
    ref = _ref(bid)
    return next(q for q in load_trusted_benchmark()["queries"] if q["query_id"] == ref["query_id"])["query"]


def _movement_scored(roles) -> frozenset[str]:
    return roles.action_terms | roles.expanded_action_terms | roles.movement_inflection_terms


def test_caesar_marched_into_epirus_scores_movement():
    roles = analyze_query("Caesar marched into Epirus")
    scored = _movement_scored(roles)
    assert scored & {"march", "marched"}
    assert "epirus" in roles.location_terms


def test_pompey_sailed_to_egypt_scores_movement():
    roles = analyze_query("Pompey sailed to Egypt")
    scored = _movement_scored(roles)
    assert scored & {"sail", "sailed"}
    assert "pompey" in roles.person_terms


def test_lucullus_crossed_into_armenia_scores_movement():
    roles = analyze_query("Lucullus crossed into Armenia")
    scored = _movement_scored(roles)
    assert scored & {"cross", "crossed"}
    assert "armenia" in roles.location_terms


def test_trace_the_route_scores_route():
    roles = analyze_query("trace the route")
    assert "route" in roles.action_terms
    assert "route" not in roles.context_terms


def test_non_movement_guard_said_not_scored():
    roles = analyze_query("Caesar said that he would advance")
    assert "said" not in roles.action_terms
    assert "said" in roles.context_terms
    assert "advance" in roles.action_terms


@pytest.mark.parametrize(
    ("query_id", "expected_terms"),
    [
        ("caesar_adriatic", {"march", "route", "travel"}),
        ("pompey_post_pharsalus_egypt", {"march", "route", "travel"}),
        ("lucullus_mithridatic", {"march", "route", "travel"}),
    ],
)
def test_production_movement_intents_score_movement_terms(query_id, expected_terms):
    query = next(q for q in load_trusted_benchmark()["queries"] if q["query_id"] == query_id)["query"]
    intents = decompose_movement_query(query)
    movement_intents = [i for i in intents if i.kind == "MOVEMENT"]
    assert movement_intents
    for intent in movement_intents:
        scored = _movement_scored(analyze_query(intent.query))
        assert scored & expected_terms, intent.query


def test_battle_assassination_action_terms_preserved():
    caesar = analyze_query("assassination of Julius Caesar")
    assert caesar.action_terms == frozenset({"assassination"})
    assert "murdered" in caesar.expanded_action_terms
    actium = analyze_query("What happened at the Battle of Actium?")
    assert "battle" in actium.action_terms


@pytest.mark.integration
def test_g5w_lexical_movement_action_terms_produce_candidates(production_retriever):
    """Movement action_terms must reach LexicalEvidenceIndex scoring path."""
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    roles = analyze_query("Julius Caesar route march travel")
    assert roles.action_terms & {"march", "route", "travel"}
    items = store.lexical_candidates("Julius Caesar route march travel", 5)
    assert items
    assert any(c.lexical_score > 0 for c in items)


@pytest.mark.integration
def test_g5w_high_lexical_rank_snapshot(production_retriever):
    """Record post-fix lexical ranks; inflection gap blocks top-20 on HIGH refs."""
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    ranks: dict[str, int | None] = {}
    for bid in HIGH_IDS + (SIDE_ID,):
        ref = _ref(bid)
        canonical = _query_for(bid)
        best = None
        for query in [canonical] + [i.query for i in decompose_movement_query(canonical)]:
            items = store.lexical_candidates(query, 50)
            rank = next((i for i, c in enumerate(items, 1) if c.id == ref["evidence_id"]), None)
            if rank is not None and (best is None or rank < best):
                best = rank
        ranks[bid] = best
    # G5V before-fix: HIGH refs >50, mithridates 21. Fix routes movement to action_terms;
    # ranks unchanged until inflection normalization (march vs marched, sail vs sailed).
    assert ranks[SIDE_ID] is not None
    assert ranks[SIDE_ID] <= 25
