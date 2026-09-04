"""G5Y: episode/location-aware conjunctive lexical scoring on route queries."""
from __future__ import annotations

import pytest

from backend.app.rag.lexical_index import LexicalEvidenceIndex
from backend.app.rag.query_roles import (
    analyze_query,
    episode_context_terms,
    route_movement_query,
)
from backend.app.rag.retrieval_intents import decompose_movement_query
from backend.tests.g5r_trusted_benchmark import load_trusted_benchmark

HIGH_IDS = ("g5r-caesar-002", "g5r-pompey-001", "g5r-lucullus-001")
G5X_RANKS = {"g5r-caesar-002": None, "g5r-pompey-001": 6, "g5r-lucullus-001": None}
THESEUS_ID = "3c7ef8ef4efa1d6e26cb3e199a73ba7a:1335:2228"


class _Corpus:
    def get(self, include=None):
        return {
            "ids": ["c1", "c2", "c3", "c4"],
            "documents": [
                "Caesar marched into Epirus with his legions.",
                "Epirus is a region of northwestern Greece.",
                "Theseus marched from Athens to the Peloponnesus.",
                "Caesar fought in Gaul without crossing the sea.",
            ],
            "metadatas": [{}, {}, {}, {}],
        }


def _index():
    return LexicalEvidenceIndex(_Corpus())


def test_movement_and_location_gets_conjunctive_boost():
    index = _index()
    results = index.query("Trace Caesar route march into Epirus", 4)
    marched = next(c for c in results if "marched into Epirus" in c.text)
    geography = next(c for c in results if "region of northwestern Greece" in c.text)
    assert marched.lexical_score > geography.lexical_score


def test_location_only_without_movement_no_boost_advantage():
    index = _index()
    results = index.query("Epirus", 3)
    geography = next(c for c in results if "region of northwestern Greece" in c.text)
    marched = next(c for c in results if "marched into Epirus" in c.text)
    assert geography.lexical_score >= marched.lexical_score


def test_movement_wrong_episode_stays_behind_target():
    index = _index()
    results = index.query("Trace Caesar route march into Epirus", 4)
    epirus_rank = next(i for i, c in enumerate(results, 1) if "Epirus" in c.text and "marched" in c.text)
    theseus_rank = next(i for i, c in enumerate(results, 1) if "Theseus" in c.text)
    assert epirus_rank < theseus_rank


def test_movement_and_episode_context_terms_present_on_route_query():
    roles = analyze_query("Trace Lucullus route march through Armenia")
    assert route_movement_query({"trace", "route", "march", "through", "armenia"}, roles)
    assert "armenia" in episode_context_terms(roles)


def test_non_route_query_disables_conjunctive_route_path():
    roles = analyze_query("Discuss Caesar's political reforms")
    assert not route_movement_query({"discuss", "caesar", "political", "reforms"}, roles)


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


def _ref(bid: str) -> dict:
    return next(r for r in load_trusted_benchmark()["references"] if r["benchmark_id"] == bid)


def _query_for(bid: str) -> str:
    ref = _ref(bid)
    return next(q for q in load_trusted_benchmark()["queries"] if q["query_id"] == ref["query_id"])["query"]


def _best_lexical_rank(store, bid: str, top_k: int = 20) -> int | None:
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
def test_g5y_high_lexical_recall(production_retriever):
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    ranks = {bid: _best_lexical_rank(store, bid, 20) for bid in HIGH_IDS}
    hits = sum(1 for bid in HIGH_IDS if ranks[bid] is not None)
    assert ranks["g5r-pompey-001"] is not None and ranks["g5r-pompey-001"] <= 20
    assert hits >= 1, f"ranks={ranks} G5X={G5X_RANKS}"


@pytest.mark.integration
def test_g5y_theseus_not_boosted_by_caesar_route(production_retriever):
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    q = _query_for("g5r-caesar-002")
    rank = next(
        (i for i, c in enumerate(store.lexical_candidates(q, 20), 1) if c.id == THESEUS_ID),
        None,
    )
    assert rank is None or rank > 10
