"""G6BA: remove unsafe top-parent bundle / parent-only proposal fallback."""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_RAW_OBSERVATION_K,
    _qualifies_semantic_proposal,
    select_qualified_local_proposals,
)
from backend.app.rag.evidence_ranking import rerank_evidence

MITHRIDATES_QUERY = (
    "Reconstruct the major movements of Mithridates VI during the First Mithridatic War, "
    "from Pontus through Asia Minor and Greece, and back toward Anatolia."
)
CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)

G6BA_MANDATORY_TARGETS = frozenset({
    "g5r-caesar-001",
    "g5r-caesar-002",
    "g5r-alexander-002",
})


def _evidence(
    identifier: str,
    text: str,
    *,
    score: float = 0.5,
    lexical_score: float | None = None,
    semantic: bool = True,
    vector_rank: int = 10,
    source: str | None = None,
) -> Evidence:
    metadata = {
        "document_id": "doc",
        "source_chunk_id": source or identifier.split(":", 1)[0],
        "semantic_candidate": semantic,
        "vector_rank": vector_rank,
    }
    if lexical_score is not None:
        metadata["lexical_candidate"] = True
        metadata["lexical_score"] = lexical_score
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt=text[:500],
        text=text,
        score=score,
        metadata=metadata,
    )


def test_g6ba_zero_local_support_not_specially_reserved():
    zero_local = _evidence(
        "top-parent:357:434",
        "Neutral administrative note without movement detail.",
        score=0.0,
        vector_rank=10,
        source="top-parent",
    )
    flood = [
        _evidence(
            f"flood-{index}:0:100",
            f"Julius Caesar marched from Italy toward Epirus in passage {index}.",
            score=0.99 - index * 0.0001,
            vector_rank=index,
            source=f"source-{index}",
        )
        for index in range(1, 200)
    ]
    ranked = rerank_evidence(CAESAR_QUERY, [*flood, zero_local])
    zero_item = next(item for item in ranked if item.id == zero_local.id)
    ranking = dict(zero_item.metadata.get("retrieval_ranking") or {})
    ranking.update(
        {
            "semantic_relevance": 0.0,
            "passage_local_support": 0.0,
            "parent_semantic_prior": 0.9,
            "navigation_penalty": 0.0,
        }
    )
    zero_item = zero_item.model_copy(
        update={"metadata": {**zero_item.metadata, "retrieval_ranking": ranking, "rank": 500}}
    )
    ranked = [item for item in ranked if item.id != zero_local.id] + [zero_item]
    assert not _qualifies_semantic_proposal(zero_item)

    preserved = select_qualified_local_proposals(ranked, 10)
    assert zero_local.id not in {item.id for item in preserved}


def test_g6ba_top_parent_bundle_children_not_bulk_reserved():
    parent = "compact-top-parent"
    children = [
        _evidence(
            f"{parent}:{index * 10}:{index * 10 + 50}",
            "Neutral administrative note without movement detail.",
            score=0.001,
            vector_rank=5,
            source=parent,
        )
        for index in range(1, 9)
    ]
    flood = [
        _evidence(
            f"flood-{index}:0:100",
            f"Julius Caesar marched from Italy toward Epirus in passage {index}.",
            score=0.99 - index * 0.0001,
            vector_rank=index,
            source=f"source-{index}",
        )
        for index in range(1, 200)
    ]
    ranked = rerank_evidence(CAESAR_QUERY, [*children, *flood])
    for child in children:
        item = next(entry for entry in ranked if entry.id == child.id)
        assert int(item.metadata.get("rank") or 999) > 10
    preserved = select_qualified_local_proposals(ranked, 10)
    reserved_children = [item for item in preserved if item.metadata.get("source_chunk_id") == parent]
    assert len(reserved_children) == 0


def test_g6ba_combined_top_ranked_candidate_reaches_proposal_via_fill():
    top = _evidence(
        "top:0:100",
        "Julius Caesar marched from Italy across the Adriatic into Epirus toward Pharsalus.",
        score=0.99,
        vector_rank=1,
        source="top-parent",
    )
    lexical_reserved = [
        _evidence(
            f"lex-{index}:0:50",
            f"Caesar put to sea toward Oricum variant {index}.",
            score=0.05,
            lexical_score=40.0 - index,
            semantic=False,
            vector_rank=999,
            source=f"lex-{index}",
        )
        for index in range(6)
    ]
    semantic_families = [
        _evidence(
            f"sem-{index}:0:100",
            f"Julius Caesar marched toward Epirus in passage {index}.",
            score=0.9 - index * 0.0001,
            vector_rank=index + 2,
            source=f"parent-{index}",
        )
        for index in range(1, 40)
    ]
    ranked = rerank_evidence(CAESAR_QUERY, [top, *lexical_reserved, *semantic_families])
    assert ranked[0].id == top.id

    preserved = select_qualified_local_proposals(ranked, 10)
    assert top.id in {item.id for item in preserved}


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from backend.app.rag.retriever import ChromaHistoricalRetriever
        from backend.app.core.config import settings
        from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
        from backend.app.rag.http_store import ChromaHttpEvidenceStore, build_production_retriever
        from backend.app.rag.query_bridge import HistoricalQueryBridge
        import chromadb
        from pathlib import Path

        provider = SentenceTransformerEmbeddingProvider(
            settings.rag_embedding_model,
            settings.rag_embedding_device,
            settings.rag_embedding_batch_size,
        )
        for path in [
            Path(__file__).resolve().parents[0].parent / "data" / "chroma_server_roman_republic_v2",
            Path(r"C:\D\python\202608231533\data\chroma_server_roman_republic_v2"),
        ]:
            if not path.exists():
                continue
            client = chromadb.PersistentClient(path=str(path))
            if settings.rag_collection in [collection.name for collection in client.list_collections()]:
                collection = client.get_collection(settings.rag_collection)
                if collection.count() > 1000:
                    retriever = ChromaHistoricalRetriever(
                        ChromaHttpEvidenceStore(collection, provider),
                        HistoricalQueryBridge(settings.rag_query_bridge_enabled),
                    )
                    retriever.retrieve("Rome", 1)
                    return retriever
        retriever = build_production_retriever(settings)
        retriever.retrieve("Rome", 1)
        return retriever
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"production chroma unavailable: {exc}")


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", sorted(G6BA_MANDATORY_TARGETS))
def test_g6ba_mandatory_targets_reach_proposal_and_union(production_retriever, benchmark_id: str):
    from backend.tests.test_g6ay_local_proposal_preservation import _ref, _trace_ref

    ref = _ref(benchmark_id)
    trace = _trace_ref(production_retriever, ref)
    assert trace["raw_present"], trace
    assert trace["proposal_present"], trace
    assert trace["union_present"], trace
