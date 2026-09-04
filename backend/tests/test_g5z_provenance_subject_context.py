"""G5Z: provenance-aware subject context for movement-bearing pronoun passages."""
from __future__ import annotations

import json

import pytest

from backend.app.rag.lexical_index import LexicalEvidenceIndex
from backend.app.rag.query_roles import analyze_query, extract_subject_context_terms, route_movement_query
from backend.app.rag.retrieval_intents import decompose_movement_query
from backend.tests.g5r_trusted_benchmark import load_trusted_benchmark

HIGH_IDS = ("g5r-caesar-002", "g5r-pompey-001", "g5r-lucullus-001")
G5Y_RANKS = {"g5r-caesar-002": None, "g5r-pompey-001": 3, "g5r-lucullus-001": 65}
THESEUS_ID = "3c7ef8ef4efa1d6e26cb3e199a73ba7a:1335:2228"


class _Corpus:
    def __init__(self, docs, metas):
        self._docs = docs
        self._metas = metas

    def get(self, include=None):
        return {"ids": [f"c{i}" for i in range(len(self._docs))], "documents": self._docs, "metadatas": self._metas}


def _index(docs, metas):
    return LexicalEvidenceIndex(_Corpus(docs, metas))


def test_extract_subject_context_from_navigation_and_heading():
    meta = {"navigation_path_json": json.dumps(["CAESAR"]), "heading": "CAESAR"}
    assert extract_subject_context_terms(meta) == frozenset({"caesar"})
    assert extract_subject_context_terms({"navigation_path_json": json.dumps(["BOOK III"])}) == frozenset()
    assert extract_subject_context_terms({"heading": "CHAPTER 4"}) == frozenset()


def test_movement_implicit_body_matching_navigation_gets_boost():
    docs = [
        "He marched so fast and put to sea in winter, passing the Ionian Sea.",
        "He marched so fast and put to sea in winter, passing the Ionian Sea.",
    ]
    metas = [
        {"heading": "CAESAR", "navigation_path_json": json.dumps(["CAESAR"])},
        {"heading": "THESEUS", "navigation_path_json": json.dumps(["THESEUS"])},
    ]
    results = _index(docs, metas).query("Trace Caesar route march across the Adriatic", 2)
    caesar = next(c for c in results if c.metadata.get("heading") == "CAESAR")
    theseus = next(c for c in results if c.metadata.get("heading") == "THESEUS")
    assert caesar.metadata["lexical_provenance_subject_score"] > 0
    assert theseus.metadata["lexical_provenance_subject_score"] == 0
    assert caesar.lexical_score > theseus.lexical_score


def test_wrong_navigation_subject_no_boost():
    docs = ["He marched from Athens to the Peloponnesus."]
    metas = [{"heading": "THESEUS", "navigation_path_json": json.dumps(["THESEUS"])}]
    results = _index(docs, metas).query("Trace Caesar route march", 1)
    assert results[0].metadata["lexical_provenance_subject_score"] == 0


def test_matching_navigation_without_movement_no_boost():
    docs = ["Caesar discussed political reforms in Rome."]
    metas = [{"heading": "CAESAR", "navigation_path_json": json.dumps(["CAESAR"])}]
    results = _index(docs, metas).query("Discuss Caesar political reforms", 1)
    assert results[0].metadata["lexical_provenance_subject_score"] == 0


def test_explicit_body_subject_preserved():
    docs = ["Caesar marched into Epirus with his legions."]
    metas = [{"heading": "CAESAR", "navigation_path_json": json.dumps(["CAESAR"])}]
    results = _index(docs, metas).query("Trace Caesar route march into Epirus", 1)
    assert results[0].metadata["lexical_body_subject_score"] > 0
    assert results[0].metadata["lexical_provenance_subject_score"] == 0


def test_explicit_body_mismatch_blocks_provenance():
    docs = ["Theseus marched from Athens to the Peloponnesus."]
    metas = [{"heading": "CAESAR", "navigation_path_json": json.dumps(["CAESAR"])}]
    results = _index(docs, metas).query("Trace Caesar route march", 1)
    assert results[0].metadata["lexical_provenance_subject_score"] == 0


def test_non_route_query_disables_provenance_route_boost():
    roles = analyze_query("Discuss Caesar's political reforms")
    assert not route_movement_query({"discuss", "caesar", "political", "reforms"}, roles)
    docs = ["He marched quickly through the province."]
    metas = [{"heading": "CAESAR", "navigation_path_json": json.dumps(["CAESAR"])}]
    results = _index(docs, metas).query("Discuss Caesar's political reforms", 1)
    assert results[0].metadata["lexical_provenance_subject_score"] == 0


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
def test_g5z_high_lexical_recall(production_retriever):
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    ranks = {bid: _best_lexical_rank(store, bid, 20) for bid in HIGH_IDS}
    hits = sum(1 for bid in HIGH_IDS if ranks[bid] is not None)
    assert ranks["g5r-pompey-001"] is not None and ranks["g5r-pompey-001"] <= 20
    assert hits >= 2, f"ranks={ranks} G5Y={G5Y_RANKS}"


@pytest.mark.integration
def test_g5z_theseus_not_boosted_by_caesar_route(production_retriever):
    from backend.app.rag.http_store import ChromaHttpEvidenceStore

    store = production_retriever.store
    assert isinstance(store, ChromaHttpEvidenceStore)
    q = _query_for("g5r-caesar-002")
    rank = next(
        (i for i, c in enumerate(store.lexical_candidates(q, 20), 1) if c.id == THESEUS_ID),
        None,
    )
    assert rank is None or rank > 10
