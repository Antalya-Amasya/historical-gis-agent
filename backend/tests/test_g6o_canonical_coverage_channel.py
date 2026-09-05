"""G6O: canonical user query as a peer coverage retrieval channel."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET, DEFAULT_PER_INTENT_K
from backend.app.rag.retrieval_intents import RetrievalIntent
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.tests.g5r_trusted_benchmark import hard_benchmark_queries, references_for_query, trusted_references

CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)
POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)


def _evidence(identifier: str, text: str, *, score: float = 0.5) -> Evidence:
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt=text[:500],
        text=text,
        score=score,
        metadata={"document_id": "doc", "source_chunk_id": identifier.split(":", 1)[0]},
    )


def _tracking_retriever(responses: dict[tuple[str, int], list[Evidence]] | None = None):
    retriever = ChromaHistoricalRetriever(object(), None)
    calls: list[tuple[str, int]] = []
    responses = responses or {}

    def tracked_retrieve(query: str, top_k: int = 5, filters=None):
        calls.append((query, top_k))
        return responses.get((query, top_k), responses.get((query, top_k), []))

    retriever.retrieve = tracked_retrieve  # type: ignore[method-assign]
    return retriever, calls


def test_a_canonical_executes_once_for_multi_intent_query():
    retriever, calls = _tracking_retriever()
    retriever.retrieve_with_coverage(CAESAR_QUERY)
    canonical_calls = [call for call in calls if call[0] == CAESAR_QUERY and call[1] == DEFAULT_COVERAGE_BUDGET]
    assert len(canonical_calls) == 1
    assert len(calls) > len(canonical_calls)


def test_b_canonical_only_candidate_enters_coverage_pool():
    canon_only = _evidence("canon-only", "Caesar marched from Italy across the Adriatic into Epirus.")
    retriever, _calls = _tracking_retriever({(CAESAR_QUERY, DEFAULT_COVERAGE_BUDGET): [canon_only]})
    merged = retriever.retrieve_with_coverage(CAESAR_QUERY)
    assert "canon-only" in {item.id for item in merged}


def test_c_canonical_does_not_bypass_normal_ranking_policy():
    weak = _evidence("canon-weak", "Generic campaign prose without movement detail.", score=0.1)
    strong = _evidence("intent-strong", "Caesar marched from Italy to Epirus after crossing the sea.", score=0.99)
    responses = {
        (CAESAR_QUERY, DEFAULT_COVERAGE_BUDGET): [weak],
    }
    retriever, _calls = _tracking_retriever(responses)

    def tracked_retrieve(query: str, top_k: int = 5, filters=None):
        _calls.append((query, top_k))
        if query == CAESAR_QUERY and top_k == DEFAULT_COVERAGE_BUDGET:
            return [weak]
        if top_k == DEFAULT_PER_INTENT_K:
            return [strong]
        return []

    retriever.retrieve = tracked_retrieve  # type: ignore[method-assign]
    merged = retriever.retrieve_with_coverage(CAESAR_QUERY, budget=3)
    assert merged[0].id == "intent-strong"


def test_d_duplicate_canonical_and_intent_evidence_dedups():
    shared = _evidence("shared-id", "Caesar marched from Italy to Epirus after Pharsalus.")
    retriever, _calls = _tracking_retriever({(CAESAR_QUERY, DEFAULT_COVERAGE_BUDGET): [shared]})

    def tracked_retrieve(query: str, top_k: int = 5, filters=None):
        _calls.append((query, top_k))
        if query == CAESAR_QUERY and top_k == DEFAULT_COVERAGE_BUDGET:
            return [shared]
        if top_k == DEFAULT_PER_INTENT_K:
            return [shared.model_copy(update={"score": 0.8})]
        return []

    retriever.retrieve = tracked_retrieve  # type: ignore[method-assign]
    merged = retriever.retrieve_with_coverage(CAESAR_QUERY, budget=5)
    assert len(merged) == len({item.id for item in merged})
    assert merged.count(shared) == 0
    assert sum(1 for item in merged if item.id == "shared-id") == 1


def test_e_single_intent_fallback_does_not_duplicate_canonical():
    retriever, calls = _tracking_retriever()
    with patch(
        "backend.app.rag.retriever.decompose_movement_query",
        return_value=(RetrievalIntent("SUBJECT", "Only Subject"),),
    ):
        retriever.retrieve_with_coverage("Only Subject route", budget=DEFAULT_COVERAGE_BUDGET)
    assert calls == [("Only Subject route", DEFAULT_COVERAGE_BUDGET)]


def test_f_authority_and_identity_unchanged():
    original = _evidence(
        "parent:10:20",
        "Caesar marched from Italy to Epirus.",
        score=0.7,
    )
    retriever, _calls = _tracking_retriever({(CAESAR_QUERY, DEFAULT_COVERAGE_BUDGET): [original]})

    def tracked_retrieve(query: str, top_k: int = 5, filters=None):
        _calls.append((query, top_k))
        if query == CAESAR_QUERY and top_k == DEFAULT_COVERAGE_BUDGET:
            return [original]
        return []

    retriever.retrieve = tracked_retrieve  # type: ignore[method-assign]
    merged = retriever.retrieve_with_coverage(CAESAR_QUERY, budget=DEFAULT_COVERAGE_BUDGET)
    item = next(entry for entry in merged if entry.id == "parent:10:20")
    assert item.text == original.text
    assert item.id == original.id
    provenance = item.metadata.get("retrieval_provenance") or {}
    assert provenance.get("retrieval_intent") == "CANONICAL"
    assert provenance.get("retrieval_query") == CAESAR_QUERY


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from backend.app.core.config import settings
        from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
        from backend.app.rag.http_store import ChromaHttpEvidenceStore
        from backend.app.rag.query_bridge import HistoricalQueryBridge
        import chromadb
        from pathlib import Path

        provider = SentenceTransformerEmbeddingProvider(
            settings.rag_embedding_model, settings.rag_embedding_device, settings.rag_embedding_batch_size
        )
        for path in [
            Path(__file__).resolve().parents[0].parent / "data" / "chroma_server_roman_republic_v2",
            Path(r"C:\D\python\202608231533\data\chroma_server_roman_republic_v2"),
        ]:
            if not path.exists():
                continue
            client = chromadb.PersistentClient(path=str(path))
            if settings.rag_collection in [c.name for c in client.list_collections()]:
                col = client.get_collection(settings.rag_collection)
                if col.count() > 1000:
                    retriever = ChromaHistoricalRetriever(
                        ChromaHttpEvidenceStore(col, provider),
                        HistoricalQueryBridge(settings.rag_query_bridge_enabled),
                    )
                    retriever.retrieve("Rome", 1)
                    return retriever
        from backend.app.rag.http_store import build_production_retriever

        retriever = build_production_retriever(settings)
        retriever.retrieve("Rome", 1)
        return retriever
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"production chroma unavailable: {exc}")


def _coverage_metrics(retriever) -> dict:
    refs = trusted_references()
    total_hit = total_exp = high_hit = high_exp = med_hit = med_exp = 0
    zero = 0
    per_query = {}
    for query in hard_benchmark_queries():
        evidence = retriever.retrieve_with_coverage(query["query"], budget=DEFAULT_COVERAGE_BUDGET)
        ids = {item.id for item in evidence}
        qrefs = references_for_query(query["query_id"])
        expected = {ref["evidence_id"] for ref in qrefs}
        hit = expected & ids
        per_query[query["query_id"]] = {
            "hit": len(hit),
            "expected": len(expected),
            "hit_ids": sorted(hit),
        }
        if not hit:
            zero += 1
        total_exp += len(expected)
        total_hit += len(hit)
        for ref in qrefs:
            if ref["quality"] == "HIGH":
                high_exp += 1
                if ref["evidence_id"] in hit:
                    high_hit += 1
            else:
                med_exp += 1
                if ref["evidence_id"] in hit:
                    med_hit += 1
    return {
        "trusted_hits": total_hit,
        "high_hits": high_hit,
        "high_expected": high_exp,
        "medium_hits": med_hit,
        "medium_expected": med_exp,
        "zero_recall_queries": zero,
        "per_query": per_query,
    }


@pytest.mark.integration
def test_g6o_trusted_coverage_acceptance(production_retriever):
    metrics = _coverage_metrics(production_retriever)
    assert metrics["trusted_hits"] >= 2, metrics
    assert metrics["high_hits"] >= 1, metrics
    assert metrics["zero_recall_queries"] <= 3, metrics


@pytest.mark.integration
def test_g6o_pompey_canonical_gap_closed(production_retriever):
    pompey_ref = next(ref for ref in trusted_references() if ref["benchmark_id"] == "g5r-pompey-001")
    coverage = production_retriever.retrieve_with_coverage(POMPEY_QUERY)
    assert pompey_ref["evidence_id"] in {item.id for item in coverage}
