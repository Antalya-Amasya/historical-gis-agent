"""G6M: parent semantic admission vs child-level relevance separation."""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.tests.g5r_trusted_benchmark import (
    hard_benchmark_queries,
    references_for_query,
    trusted_references,
)


def _evidence(
    identifier: str,
    text: str,
    *,
    parent: str,
    vector_rank: int = 1,
    lexical: bool = False,
    lexical_score: float = 0.0,
    semantic: bool = True,
) -> Evidence:
    metadata = {
        "document_id": "doc",
        "source_chunk_id": parent,
        "semantic_candidate": semantic,
        "vector_rank": vector_rank,
    }
    if lexical:
        metadata["lexical_candidate"] = True
        metadata["lexical_score"] = lexical_score
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt=text[:500],
        text=text,
        score=0.5,
        metadata=metadata,
    )


def _ranking(query: str, items: list[Evidence]) -> dict[str, dict]:
    return {item.id: item.metadata["retrieval_ranking"] for item in rerank_evidence(query, items)}


def _order(query: str, items: list[Evidence]) -> list[str]:
    return [item.id for item in rerank_evidence(query, items)]


ROUTE_QUERY = (
    "Trace the commander's route from the first camp across the sea "
    "into the eastern province in winter."
)


def test_a_strong_parent_weak_sibling_movement_child_outranks_generic_prose():
    parent = "parent-a"
    movement = _evidence(
        "movement-child",
        "The commander crossed the sea from the port and sailed into the eastern province in winter.",
        parent=parent,
        vector_rank=49,
    )
    generic = _evidence(
        "generic-sibling",
        "campaign against the rival in Macedonia.",
        parent=parent,
        vector_rank=49,
    )
    ranked = rerank_evidence(ROUTE_QUERY, [generic, movement])
    details = {item.id: item.metadata["retrieval_ranking"] for item in ranked}
    assert ranked[0].id == "movement-child"
    assert details["generic-sibling"]["semantic_relevance"] == 0.0
    assert details["movement-child"]["semantic_relevance"] > details["generic-sibling"]["semantic_relevance"]


def test_b_trusted_lexical_child_beats_wrong_same_parent_sibling():
    parent = "parent-b"
    trusted = _evidence(
        "trusted-child",
        "As soon as it was resolved that he should fly into Egypt, setting sail from the island "
        "together with his wife, he passed over sea without danger.",
        parent=parent,
        vector_rank=49,
        lexical=True,
        lexical_score=33.9,
    )
    wrong = _evidence(
        "wrong-sibling",
        "When he considered flying into Africa, he began to complain and blame himself to his friends.",
        parent=parent,
        vector_rank=49,
    )
    assert _order(ROUTE_QUERY, [wrong, trusted])[0] == "trusted-child"
    wrong_score = _ranking(ROUTE_QUERY, [wrong, trusted])["wrong-sibling"]
    assert wrong_score["lexical_support"] == 0.0
    assert wrong_score["semantic_relevance"] <= 0.2


def test_c_semantic_only_good_child_remains_viable_without_lexical_hit():
    item = _evidence(
        "semantic-good",
        "The commander crossed the sea from the first camp into the eastern province in winter.",
        parent="semantic-only-parent",
        vector_rank=53,
        lexical=False,
    )
    details = rerank_evidence(ROUTE_QUERY, [item])[0].metadata["retrieval_ranking"]
    assert details["lexical_support"] == 0.0
    assert details["semantic_relevance"] > 0.2
    assert details["passage_relevance"] == details["semantic_relevance"]
    assert details["final_score"] > 0.3


def test_d_semantic_only_weak_child_with_strong_parent_rank_stays_low():
    item = _evidence(
        "semantic-weak",
        "The weather remained mild throughout the distant province.",
        parent="strong-parent",
        vector_rank=1,
        lexical=False,
    )
    details = rerank_evidence(ROUTE_QUERY, [item])[0].metadata["retrieval_ranking"]
    assert details["parent_semantic_prior"] == 1.0
    assert details["passage_local_support"] == 0.0
    assert details["semantic_relevance"] == 0.0
    assert details["passage_relevance"] == 0.0
    assert details["final_score"] < 0.1


def test_e_evidence_identity_and_offsets_unchanged():
    parent = "identity-parent"
    start, end = 2415, 2929
    child_id = f"{parent}:{start}:{end}"
    text = "He passed over sea without danger toward the province."
    item = _evidence(child_id, text, parent=parent, vector_rank=49)
    ranked = rerank_evidence(ROUTE_QUERY, [item])[0]
    assert ranked.id == child_id
    assert ranked.text == text
    assert ranked.metadata["source_chunk_id"] == parent
    assert ranked.metadata["vector_rank"] == 49


def test_g6l_caesar_pattern_movement_child_beats_toc_heading_sibling():
    parent = "caesar-like-parent"
    movement = _evidence(
        f"{parent}:1391:1640",
        "When the civil strife broke out into war the leader crossed the sea from the port in winter.",
        parent=parent,
        vector_rank=55,
    )
    heading = _evidence(
        f"{parent}:0:105",
        "campaign against the rival in Macedonia. Of the rest THE ILLYRIAN WARS concentration of forces.",
        parent=parent,
        vector_rank=55,
    )
    ranked = rerank_evidence(ROUTE_QUERY, [heading, movement])
    assert ranked[0].id == f"{parent}:1391:1640"
    assert ranked[0].metadata["retrieval_ranking"]["semantic_relevance"] > 0.0
    assert ranked[1].metadata["retrieval_ranking"]["semantic_relevance"] == 0.0


def test_g6l_lucullus_pattern_lexical_movement_beats_generic_biography_sibling():
    parent = "lucullus-like-parent"
    trusted = _evidence(
        f"{parent}:626:942",
        "Wherefore hastening his march, and passing the river, he came over into Armenia.",
        parent=parent,
        vector_rank=53,
        lexical=True,
        lexical_score=49.2,
    )
    biography = _evidence(
        f"{parent}:0:363",
        "of the goddess, a torch, branded on them; and it is no such light or easy thing.",
        parent=parent,
        vector_rank=53,
    )
    ranked = rerank_evidence(ROUTE_QUERY, [biography, trusted])
    assert ranked[0].id == f"{parent}:626:942"
    assert ranked[0].metadata["retrieval_ranking"]["lexical_support"] > 0.0


def test_parent_semantic_prior_retained_as_provenance_only():
    high_parent = _evidence("high", "The commander marched into the province.", parent="p1", vector_rank=1)
    low_parent = _evidence("low", "The commander marched into the province.", parent="p2", vector_rank=40)
    details = _ranking(ROUTE_QUERY, [low_parent, high_parent])
    assert details["high"]["parent_semantic_prior"] > details["low"]["parent_semantic_prior"]
    assert details["high"]["semantic_relevance"] == details["low"]["semantic_relevance"]


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


def _trusted_metrics(retriever, top_k: int = 20) -> dict:
    refs = trusted_references()
    per_query = {}
    total_hit = 0
    total_expected = 0
    high_hit = high_expected = medium_hit = medium_expected = 0
    zero_recall = 0
    for query in hard_benchmark_queries():
        qid = query["query_id"]
        evidence = retriever.retrieve(query["query"], top_k=top_k)
        evidence_ids = {item.id for item in evidence}
        qrefs = references_for_query(qid)
        expected = {ref["evidence_id"] for ref in qrefs}
        hit_ids = expected & evidence_ids
        per_query[qid] = {
            "expected": len(expected),
            "hit": len(hit_ids),
            "hit_ids": sorted(hit_ids),
        }
        total_expected += len(expected)
        total_hit += len(hit_ids)
        if not hit_ids:
            zero_recall += 1
        for ref in qrefs:
            if ref["quality"] == "HIGH":
                high_expected += 1
                if ref["evidence_id"] in hit_ids:
                    high_hit += 1
            else:
                medium_expected += 1
                if ref["evidence_id"] in hit_ids:
                    medium_hit += 1
    return {
        "trusted_hits": total_hit,
        "trusted_expected": total_expected,
        "high_hits": high_hit,
        "high_expected": high_expected,
        "medium_hits": medium_hit,
        "medium_expected": medium_expected,
        "zero_recall_queries": zero_recall,
        "per_query": per_query,
    }


@pytest.mark.integration
def test_g6m_trusted_retrieval_does_not_regress(production_retriever):
    metrics = _trusted_metrics(production_retriever, top_k=20)
    assert metrics["trusted_hits"] >= 2, metrics
    assert metrics["zero_recall_queries"] <= 3, metrics
