"""Read-only parity report for the production Query Bridge pilot."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    import chromadb
    from backend.app.core.config import settings
    from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
    from backend.app.rag.evaluation.retrieval_eval import evaluate
    from backend.app.rag.http_store import ChromaHttpEvidenceStore, build_production_retriever
    from backend.app.rag.query_bridge import HistoricalQueryBridge
    from backend.app.rag.retriever import ChromaHistoricalRetriever

    items = json.loads(Path("backend/tests/fixtures/roman_republic_retrieval_eval.json").read_text(encoding="utf-8"))
    paired_zh = [item for item in items if item.get("strict") and item.get("pair_id") and item["language"] == "zh"]
    collection = chromadb.HttpClient(host=settings.rag_chroma_host, port=settings.rag_chroma_port).get_collection(settings.rag_collection)
    enabled = build_production_retriever(settings)
    provider = SentenceTransformerEmbeddingProvider(settings.rag_embedding_model, settings.rag_embedding_device, settings.rag_embedding_batch_size)
    disabled = ChromaHistoricalRetriever(ChromaHttpEvidenceStore(collection, provider), HistoricalQueryBridge(enabled=False))
    before = collection.count()
    report = {"paired_zh_count": len(paired_zh), "enabled": evaluate(enabled, paired_zh), "disabled": evaluate(disabled, paired_zh), "collection_count_before": before, "collection_count_after": collection.count()}
    Path("data/historical_sources/processed/production_query_bridge_pilot_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
