"""G6BN: child-local route-fragment relevance."""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag import evidence_ranking
from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.query_roles import analyze_query
from backend.tests.g5r_trusted_benchmark import hard_benchmark_queries, trusted_references

MARCUS_QUERY = "Trace Marcus Valerius from Italy across the Adriatic into Epirus."
MARCUS_NAV = {"navigation_path_json": '["MARCUS"]', "heading": "MARCUS"}


def _evidence(
    text: str,
    *,
    metadata: dict | None = None,
    identifier: str = "item",
) -> Evidence:
    base = {
        "document_id": "doc",
        "semantic_candidate": True,
        "vector_rank": 5,
        **MARCUS_NAV,
    }
    if metadata:
        base.update(metadata)
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt=text,
        text=text,
        score=0.5,
        metadata=base,
    )


def _ranking(query: str, text: str, **kwargs) -> dict:
    ranked = rerank_evidence(query, [_evidence(text, **kwargs)], pool_relative=False)
    return ranked[0].metadata["retrieval_ranking"]


def test_implicit_subject_route_fragment_gets_bounded_relevance():
    text = "He put to sea in winter, crossed the Ionian Sea, and landed near Oricum."
    score = _ranking(MARCUS_QUERY, text)
    assert score["route_fragment_relevance"] > 0
    assert score["person_support"] == 0.0
    assert score["semantic_relevance"] > 0.1
    assert score["final_score"] > 0.1


def test_generic_movement_without_route_context_not_boosted():
    text = "He travelled quickly with the army."
    score = _ranking(MARCUS_QUERY, text)
    assert score["route_fragment_relevance"] == 0.0
    assert score["semantic_relevance"] <= 0.1


def test_explicit_wrong_actor_blocks_fragment_subject_hint():
    text = "Bion crossed the Ionian Sea and landed near Oricum."
    score = _ranking(MARCUS_QUERY, text)
    assert score["route_fragment_relevance"] == 0.0


def test_geography_without_movement_not_boosted():
    text = "Oricum and Apollonia were cities near the Ionian Sea."
    score = _ranking(MARCUS_QUERY, text)
    assert score["route_fragment_relevance"] == 0.0
    assert score["action_support"] == 0.0


def test_explicit_full_route_not_double_boosted():
    text = "Marcus Valerius marched from Italy across the Adriatic into Epirus."
    score = _ranking(MARCUS_QUERY, text)
    assert score["route_fragment_relevance"] == 0.0
    assert score["final_score"] >= 1.0


def test_fragment_signal_is_bounded():
    text = "He put to sea in winter, crossed the Ionian Sea, and landed near Oricum."
    score = _ranking(MARCUS_QUERY, text)
    assert 0 < score["route_fragment_relevance"] <= 0.06
    assert score["entity_support"] <= 0.08


@pytest.mark.parametrize(
    ("text", "expect_fragment"),
    [
        ("Marcus Valerius marched from Italy across the Adriatic into Epirus.", False),
        ("He put to sea in winter, crossed the Ionian Sea, and landed near Oricum.", True),
        ("Bion crossed the Ionian Sea and landed near Oricum.", False),
        ("He travelled quickly with the army.", False),
        ("Oricum and Apollonia were cities near the Ionian Sea.", False),
        ("He debated supply arrangements with the army.", False),
    ],
)
def test_synthetic_safety_matrix(text: str, expect_fragment: bool):
    score = _ranking(MARCUS_QUERY, text)
    if expect_fragment:
        assert score["route_fragment_relevance"] > 0
    else:
        assert score["route_fragment_relevance"] == 0.0


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from pathlib import Path

        import chromadb

        from backend.app.core.config import settings
        from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
        from backend.app.rag.http_store import ChromaHttpEvidenceStore, build_production_retriever
        from backend.app.rag.query_bridge import HistoricalQueryBridge
        from backend.app.rag.retriever import ChromaHistoricalRetriever

        provider = SentenceTransformerEmbeddingProvider(
            settings.rag_embedding_model,
            settings.rag_embedding_device,
            settings.rag_embedding_batch_size,
        )
        for path in [
            Path(__file__).resolve().parents[1] / "data" / "chroma_server_roman_republic_v2",
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


def _common_rank(retriever, benchmark_id: str) -> dict:
    from backend.app.rag.retrieval_intents import RetrievalIntent, decompose_movement_query

    ref = next(item for item in trusted_references() if item["benchmark_id"] == benchmark_id)
    query = next(item["query"] for item in hard_benchmark_queries() if item["query_id"] == ref["query_id"])
    target = ref["evidence_id"]
    channels = [RetrievalIntent("CANONICAL", query), *decompose_movement_query(query)]
    pool = {
        item.id: item
        for channel in channels
        for item in retriever._collect_candidates(channel.query, semantic_k=60, lexical_k=60)
    }
    ranked = rerank_evidence(query, list(pool.values()), pool_relative=False)
    rank = next(index for index, item in enumerate(ranked, 1) if item.id == target)
    score = next(item for item in ranked if item.id == target).metadata["retrieval_ranking"]
    final = target in {item.id for item in retriever.retrieve_with_coverage(query, DEFAULT_COVERAGE_BUDGET)}
    return {"rank": rank, "score": score["final_score"], "ranking": score, "final": final}


@pytest.mark.integration
def test_caesar_002_route_fragment_improves(production_retriever):
    result = _common_rank(production_retriever, "g5r-caesar-002")
    assert result["ranking"]["route_fragment_relevance"] > 0
    assert result["rank"] < 1195
    assert result["score"] > 0.25


@pytest.mark.integration
def test_alexander_002_route_fragment_improves(production_retriever):
    result = _common_rank(production_retriever, "g5r-alexander-002")
    assert result["ranking"]["route_fragment_relevance"] > 0
    assert result["rank"] <= 384


@pytest.mark.integration
def test_existing_final_hits_preserved(production_retriever):
    for benchmark_id in ("g5r-pompey-001", "g5r-mithridates-001", "g5r-lucullus-002"):
        result = _common_rank(production_retriever, benchmark_id)
        assert result["final"] is True


def test_fragment_ablation_zeroes_boost(monkeypatch):
    text = "He put to sea in winter, crossed the Ionian Sea, and landed near Oricum."
    boosted = _ranking(MARCUS_QUERY, text)
    monkeypatch.setattr(evidence_ranking, "route_fragment_relevance", lambda *args, **kwargs: 0.0)
    ablated = _ranking(MARCUS_QUERY, text)
    assert boosted["route_fragment_relevance"] > 0
    assert ablated["route_fragment_relevance"] == 0.0
    assert boosted["final_score"] > ablated["final_score"]
