"""G6AY: qualified local proposal preservation before coverage union."""
from __future__ import annotations

from collections import defaultdict

import pytest

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    DEFAULT_RAW_OBSERVATION_K,
    merge_coverage_results,
    select_qualified_local_proposals,
)
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent, decompose_movement_query
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.tests.g5r_trusted_benchmark import hard_benchmark_queries, trusted_references

CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)
MITHRIDATES_QUERY = (
    "Reconstruct the major movements of Mithridates VI during the First Mithridatic War, "
    "from Pontus through Asia Minor and Greece, and back toward Anatolia."
)
ALEXANDER_QUERY = (
    "Reconstruct Alexander the Great's route from Bactria through the Hindu Kush into the "
    "Indian campaign, ending near the Hydaspes."
)
POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through the "
    "eastern Mediterranean until his arrival in Egypt in 48 BCE."
)

G6AY_TARGETS = frozenset({
    "g5r-caesar-001",
    "g5r-caesar-002",
    "g5r-mithridates-002",
    "g5r-alexander-002",
})
PRESERVE_FINAL = frozenset({"g5r-pompey-001", "g5r-mithridates-001", "g5r-lucullus-002"})


def _ref(benchmark_id: str) -> dict:
    return next(ref for ref in trusted_references() if ref["benchmark_id"] == benchmark_id)


def _query_for_ref(ref: dict) -> str:
    query_id = ref["query_id"]
    return next(q["query"] for q in hard_benchmark_queries() if q["query_id"] == query_id)


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


def _trace_ref(retriever, ref: dict) -> dict:
    query = _query_for_ref(ref)
    target_id = ref["evidence_id"]
    intents = decompose_movement_query(query)
    channels = [RetrievalIntent("CANONICAL", query), *intents]

    raw_present = False
    raw_semantic_rank = None
    raw_lexical_rank = None
    derived_present = False
    best_local_rank = None
    proposal_present = False
    union_present = False
    final_present = False

    for intent in channels:
        semantic_k = min(DEFAULT_RAW_OBSERVATION_K, max(20, DEFAULT_RAW_OBSERVATION_K))
        lexical_k = min(100, max(30, DEFAULT_RAW_OBSERVATION_K))
        candidates = retriever._collect_candidates(
            intent.query,
            semantic_k=semantic_k,
            lexical_k=lexical_k,
        )
        candidate_ids = {item.id for item in candidates}
        if target_id in candidate_ids:
            raw_present = True
            derived_present = True
            ranked_pool = rerank_evidence(intent.query, candidates)
            for rank, item in enumerate(ranked_pool, start=1):
                if item.id != target_id:
                    continue
                best_local_rank = rank if best_local_rank is None else min(best_local_rank, rank)
                meta = item.metadata
                if meta.get("semantic_candidate"):
                    parent_rank = meta.get("vector_rank")
                    if parent_rank is not None:
                        raw_semantic_rank = (
                            parent_rank
                            if raw_semantic_rank is None
                            else min(raw_semantic_rank, parent_rank)
                        )
                if meta.get("lexical_candidate"):
                    lexical_items = sorted(
                        (
                            cand
                            for cand in candidates
                            if cand.metadata.get("lexical_candidate")
                        ),
                        key=lambda cand: (
                            -float(cand.metadata.get("lexical_score") or 0.0),
                            cand.id,
                        ),
                    )
                    for lex_rank, lex_item in enumerate(lexical_items, start=1):
                        if lex_item.id == target_id:
                            raw_lexical_rank = (
                                lex_rank
                                if raw_lexical_rank is None
                                else min(raw_lexical_rank, lex_rank)
                            )
                            break

        proposals = retriever.retrieve_candidates(intent.query, DEFAULT_RAW_OBSERVATION_K)
        for rank, item in enumerate(proposals, start=1):
            if item.id == target_id:
                proposal_present = True
                best_local_rank = rank if best_local_rank is None else min(best_local_rank, rank)

    intent_results = [
        (intent, retriever.retrieve_candidates(intent.query, DEFAULT_RAW_OBSERVATION_K))
        for intent in channels
    ]
    union_ids = {item.id for _intent, items in intent_results for item in items}
    union_present = target_id in union_ids

    final = retriever.retrieve_with_coverage(query, budget=DEFAULT_COVERAGE_BUDGET)
    final_present = target_id in {item.id for item in final}

    return {
        "benchmark_id": ref["benchmark_id"],
        "raw_present": raw_present,
        "raw_semantic_rank": raw_semantic_rank,
        "raw_lexical_rank": raw_lexical_rank,
        "derived_present": derived_present,
        "best_local_rank": best_local_rank,
        "proposal_present": proposal_present,
        "union_present": union_present,
        "final_present": final_present,
    }


@pytest.fixture(scope="module")
def production_retriever():
    try:
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


def test_a_synthetic_lexical_direct_preservation():
    query = CAESAR_QUERY
    filler = "Generic winter camp notes without a crossing."
    semantic_flood = [
        _evidence(
            f"sem:{index}:0:100",
            f"Julius Caesar marched from Italy toward Epirus in passage {index}.",
            score=0.95 - index * 0.0005,
            vector_rank=index,
            source=f"parent-{index // 3}",
        )
        for index in range(1, 500)
    ]
    lexical = _evidence(
        "lex:strong:0:200",
        "Julius Caesar put to sea in winter and passed the Ionian Sea to Oricum.",
        score=0.05,
        lexical_score=42.0,
        semantic=False,
        vector_rank=999,
        source="lex-parent",
    )
    candidates = semantic_flood + [lexical]
    ranked = rerank_evidence(query, candidates)
    naive = ranked[:10]
    assert lexical.id not in {item.id for item in naive}

    preserved = select_qualified_local_proposals(ranked, 10)
    assert lexical.id in {item.id for item in preserved}


def test_b_synthetic_semantic_derived_preservation():
    query = CAESAR_QUERY
    parent_source = "caesar-parent"
    derived = _evidence(
        f"{parent_source}:1391:1640",
        "Caesar crossed the Adriatic from Brundisium in the winter and opened his campaign.",
        score=0.08,
        vector_rank=8,
        source=parent_source,
    )
    semantic_flood = [
        _evidence(
            f"parent-{index}:{index * 10}:{index * 10 + 80}",
            f"Julius Caesar marched from Italy toward Epirus in passage {index}.",
            score=0.9 - index * 0.0005,
            vector_rank=index,
            source=f"parent-{index}",
        )
        for index in range(1, 500)
    ]
    ranked = rerank_evidence(query, [derived, *semantic_flood])
    assert derived.id not in {item.id for item in ranked[:10]}

    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    assert derived.id in {item.id for item in preserved}


def test_c_duplicate_evidence_id_uses_one_slot():
    query = CAESAR_QUERY
    dual = _evidence(
        "dup:10:20",
        "Caesar crossed from Italy across the Adriatic into Epirus.",
        score=0.4,
        lexical_score=20.0,
        vector_rank=15,
        source="dup-parent",
    )
    ranked = rerank_evidence(query, [dual])
    preserved = select_qualified_local_proposals(ranked, 5)
    assert len(preserved) == 1
    assert preserved[0].id == dual.id


def test_d_weak_candidate_not_blindly_reserved():
    query = CAESAR_QUERY
    strong = _evidence(
        "strong:10:20",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        score=0.9,
        lexical_score=30.0,
        vector_rank=5,
        source="strong-parent",
    )
    weak_lexical = _evidence(
        "weak:10:20",
        "The senate debated grain supplies in Rome that winter.",
        score=0.01,
        lexical_score=0.5,
        semantic=False,
        vector_rank=999,
        source="weak-parent",
    )
    ranked = rerank_evidence(query, [strong, weak_lexical])
    preserved = select_qualified_local_proposals(ranked, 1)
    assert preserved[0].id == strong.id


@pytest.mark.integration
@pytest.mark.parametrize(
    "benchmark_id",
    sorted(G6AY_TARGETS),
)
def test_g6ay_target_ref_reaches_proposal_and_union(production_retriever, benchmark_id: str):
    ref = _ref(benchmark_id)
    trace = _trace_ref(production_retriever, ref)
    assert trace["raw_present"], trace
    assert trace["proposal_present"], trace
    assert trace["union_present"], trace


@pytest.mark.integration
def test_i_pompey_002_remains_raw_miss(production_retriever):
    ref = _ref("g5r-pompey-002")
    trace = _trace_ref(production_retriever, ref)
    assert not trace["raw_present"], trace


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", sorted(PRESERVE_FINAL))
def test_j_existing_final_hits_preserved(production_retriever, benchmark_id: str):
    ref = _ref(benchmark_id)
    trace = _trace_ref(production_retriever, ref)
    assert trace["final_present"], trace


@pytest.mark.integration
def test_g6ay_proposal_metric_improves(production_retriever):
    refs = trusted_references()
    proposal_hits = 0
    for ref in refs:
        trace = _trace_ref(production_retriever, ref)
        if trace["proposal_present"]:
            proposal_hits += 1
    assert proposal_hits >= 6, proposal_hits


def test_select_qualified_local_proposals_stays_bounded():
    query = CAESAR_QUERY
    candidates = [
        _evidence(f"id:{index}:0:10", f"Caesar marched toward Epirus passage {index}.", vector_rank=index)
        for index in range(1, 200)
    ]
    ranked = rerank_evidence(query, candidates)
    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    assert len(preserved) == DEFAULT_RAW_OBSERVATION_K


def test_merge_coverage_union_unchanged_for_single_proposal():
    query = CAESAR_QUERY
    item = _evidence("union:10:20", "Caesar crossed from Italy into Epirus.")
    intent = RetrievalIntent("SUBJECT", "Julius Caesar")
    merged = merge_coverage_results(query, [(intent, [item])], 20)
    assert item.id in {entry.id for entry in merged}
