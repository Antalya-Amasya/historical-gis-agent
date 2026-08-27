"""Read-only document-level baseline evaluator for the v2 production retriever."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path


def evaluate(retriever, items, top_k=10):
    """Evaluate read-only dense retrieval at the document relevance-set level.

    ``hit_at`` is Success/Hit@K (any relevant document); ``document_recall_at``
    is the fraction of the complete expected document set retrieved by K.
    """
    results = []
    for item in items:
        evidence = retriever.retrieve(item["query"], top_k)
        docs = [evidence_item.metadata.get("document_id") for evidence_item in evidence]
        expected = set(item["expected_document_ids"])
        first = next((rank + 1 for rank, document_id in enumerate(docs) if document_id in expected), None)

        def at_k(k):
            top_docs = set(docs[:k])
            return {
                "hit": bool(expected & top_docs),
                "document_recall": len(expected & top_docs) / len(expected),
                "source_coverage": len(expected & top_docs) / len(expected),
                "unique_documents": len(top_docs),
                "max_chunks_same_document": max(Counter(docs[:k]).values(), default=0),
            }

        cuts = {str(k): at_k(k) for k in (1, 3, 5, 10)}
        results.append({
            **item,
            "hits": [{
                "rank": rank,
                "chunk_id": evidence_item.id,
                "document_id": evidence_item.metadata.get("document_id"),
                "author": evidence_item.author,
                "work": evidence_item.work,
                "distance": evidence_item.metadata.get("distance"),
                "score": evidence_item.score,
                "navigation_metadata": {key: value for key, value in evidence_item.metadata.items() if key in {"book", "chapter", "section", "navigation_path_json", "page_start", "page_end"}},
                "text_preview": evidence_item.text[:300] if evidence_item.text else None,
            } for rank, evidence_item in enumerate(evidence, start=1)],
            "first_relevant_rank": first,
            "document_reciprocal_rank": 1 / first if first else 0,
            "hit_at": {k: value["hit"] for k, value in cuts.items()},
            "document_recall_at": {k: value["document_recall"] for k, value in cuts.items()},
            "source_coverage_at": {k: value["source_coverage"] for k, value in cuts.items()},
            "unique_documents_at": {k: value["unique_documents"] for k, value in cuts.items()},
            "max_chunks_same_document_at": {k: value["max_chunks_same_document"] for k, value in cuts.items()},
        })

    def metrics(rows):
        if not rows:
            return {"n": 0, "hit_at": {}, "document_recall_at": {}, "mrr": None, "source_coverage_at": {}}
        return {
            "n": len(rows),
            "hit_at": {str(k): sum(row["hit_at"][str(k)] for row in rows) / len(rows) for k in (1, 3, 5, 10)},
            "document_recall_at": {str(k): sum(row["document_recall_at"][str(k)] for row in rows) / len(rows) for k in (1, 3, 5, 10)},
            "mrr": sum(row["document_reciprocal_rank"] for row in rows) / len(rows),
            "source_coverage_at": {str(k): sum(row["source_coverage_at"][str(k)] for row in rows) / len(rows) for k in (5, 10)},
            "crowding": {str(k): {"mean_unique_documents": sum(row["unique_documents_at"][str(k)] for row in rows) / len(rows), "mean_max_chunks_same_document": sum(row["max_chunks_same_document_at"][str(k)] for row in rows) / len(rows)} for k in (5, 10)},
        }

    strict = [row for row in results if row["strict"]]
    groups = {"overall_strict": metrics(strict)}
    for key in ("language", "category"):
        for value in sorted({row[key] for row in strict}):
            groups[f"{key}:{value}"] = metrics([row for row in strict if row[key] == value])

    paired = []
    for pair_id in sorted({row.get("pair_id") for row in strict if row.get("pair_id")}):
        rows = {row["language"]: row for row in strict if row.get("pair_id") == pair_id}
        if set(rows) != {"en", "zh"}:
            continue
        paired.append({
            "pair_id": pair_id,
            "english": {key: rows["en"][key] for key in ("first_relevant_rank", "hit_at", "document_recall_at", "document_reciprocal_rank")},
            "chinese": {key: rows["zh"][key] for key in ("first_relevant_rank", "hit_at", "document_recall_at", "document_reciprocal_rank")},
            "rank_delta_chinese_minus_english": None if rows["en"]["first_relevant_rank"] is None or rows["zh"]["first_relevant_rank"] is None else rows["zh"]["first_relevant_rank"] - rows["en"]["first_relevant_rank"],
        })
    paired_english = [row for row in strict if row.get("pair_id") and row["language"] == "en"]
    paired_chinese = [row for row in strict if row.get("pair_id") and row["language"] == "zh"]
    return {"evaluation_scope": "document_level_only", "metric_semantics": {"hit_at": "Success@K: at least one expected document occurs in Top-K.", "document_recall_at": "Fraction of the full expected-document relevance set occurring in Top-K.", "mrr": "Reciprocal rank of the first expected document.", "source_coverage_at": "Fraction of expected documents represented in Top-K."}, "items": results, "metrics": groups, "paired_results": paired, "paired_metrics": {"english": metrics(paired_english), "chinese": metrics(paired_chinese)}, "strict_count": len(strict), "uncertain_count": len(results) - len(strict)}


def main():
    import chromadb
    from backend.app.core.config import settings
    from backend.app.rag.http_store import ChromaHttpEvidenceStore
    from backend.app.rag.retriever import ChromaHistoricalRetriever
    from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
    items=json.loads(Path('backend/tests/fixtures/roman_republic_retrieval_eval.json').read_text(encoding='utf-8'))
    collection=chromadb.HttpClient(host=settings.rag_chroma_host,port=settings.rag_chroma_port).get_collection(settings.rag_collection)
    retriever=ChromaHistoricalRetriever(ChromaHttpEvidenceStore(collection,SentenceTransformerEmbeddingProvider(settings.rag_embedding_model,settings.rag_embedding_device,settings.rag_embedding_batch_size)))
    count_before = collection.count()
    report = evaluate(retriever, items)
    report["collection_count_before"] = count_before
    report["collection_count_after"] = collection.count()
    Path("data/historical_sources/processed/retrieval_eval_hardened_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == '__main__': main()
