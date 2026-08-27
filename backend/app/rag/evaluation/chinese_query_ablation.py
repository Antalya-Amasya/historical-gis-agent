"""Evaluation-only ablation for Chinese historical query representations."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


VARIANTS = ("RAW_ZH", "ZH_ALIAS", "EN_CANONICAL", "ZH_EN_MIXED")


def _query_for(variant, chinese_query, english_query, aliases):
    if variant == "RAW_ZH":
        return chinese_query
    if variant == "ZH_ALIAS":
        return " ".join([chinese_query, *aliases])
    if variant == "EN_CANONICAL":
        return english_query
    if variant == "ZH_EN_MIXED":
        return f"{chinese_query} | {english_query}"
    raise ValueError(f"unknown_variant:{variant}")


def _row(retriever, pair_id, variant, query, aliases, expected_document_ids, top_k):
    evidence = retriever.retrieve(query, top_k)
    docs = [item.metadata.get("document_id") for item in evidence]
    expected = set(expected_document_ids)
    first = next((rank for rank, document_id in enumerate(docs, start=1) if document_id in expected), None)
    def at(k):
        top_documents = set(docs[:k])
        return {"hit": bool(top_documents & expected), "document_recall": len(top_documents & expected) / len(expected), "unique_documents": len(top_documents), "max_chunks_same_document": max(Counter(docs[:k]).values(), default=0)}
    cuts = {str(k): at(k) for k in (1, 3, 5, 10)}
    return {
        "pair_id": pair_id, "variant": variant, "actual_query_text": query, "aliases_used": aliases,
        "expected_document_ids": expected_document_ids, "first_relevant_rank": first,
        "mrr": 1 / first if first else 0,
        "hit_at": {key: value["hit"] for key, value in cuts.items()},
        "document_recall_at": {key: value["document_recall"] for key, value in cuts.items()},
        "unique_documents_at": {key: value["unique_documents"] for key, value in cuts.items()},
        "max_chunks_same_document_at": {key: value["max_chunks_same_document"] for key, value in cuts.items()},
        "hits": [{"rank": rank, "chunk_id": item.id, "document_id": item.metadata.get("document_id"), "author": item.author, "work": item.work, "distance": item.metadata.get("distance"), "score": item.score, "text_preview": item.text[:300] if item.text else None} for rank, item in enumerate(evidence, start=1)],
    }


def _aggregate(rows):
    return {"n": len(rows), "hit_at": {str(k): sum(row["hit_at"][str(k)] for row in rows) / len(rows) for k in (1, 3, 5, 10)}, "document_recall_at": {str(k): sum(row["document_recall_at"][str(k)] for row in rows) / len(rows) for k in (1, 3, 5, 10)}, "mrr": sum(row["mrr"] for row in rows) / len(rows), "crowding": {str(k): {"mean_unique_documents": sum(row["unique_documents_at"][str(k)] for row in rows) / len(rows), "mean_max_chunks_same_document": sum(row["max_chunks_same_document_at"][str(k)] for row in rows) / len(rows)} for k in (5, 10)}}


def run_ablation(retriever, fixture_items, alias_specs, top_k=10):
    strict_pairs = {}
    for item in fixture_items:
        if item.get("strict") and item.get("pair_id"):
            strict_pairs.setdefault(item["pair_id"], {})[item["language"]] = item
    aliases_by_pair = {spec["pair_id"]: spec for spec in alias_specs}
    if set(strict_pairs) != set(aliases_by_pair):
        raise ValueError("alias_spec_pair_set_mismatch")
    rows = []
    for pair_id in sorted(strict_pairs):
        pair = strict_pairs[pair_id]
        if set(pair) != {"en", "zh"}:
            raise ValueError(f"fixture_pair_shape_invalid:{pair_id}")
        spec = aliases_by_pair[pair_id]
        if spec["chinese_query"] != pair["zh"]["query"] or spec["english_query"] != pair["en"]["query"]:
            raise ValueError(f"alias_spec_query_mismatch:{pair_id}")
        aliases = [entry["alias"] for entry in spec["aliases"]]
        for variant in VARIANTS:
            query = _query_for(variant, pair["zh"]["query"], pair["en"]["query"], aliases)
            rows.append(_row(retriever, pair_id, variant, query, aliases if variant == "ZH_ALIAS" else [], pair["zh"]["expected_document_ids"], top_k))
    by_variant = {variant: _aggregate([row for row in rows if row["variant"] == variant]) for variant in VARIANTS}
    comparisons = []
    for pair_id in sorted(strict_pairs):
        runs = {row["variant"]: row for row in rows if row["pair_id"] == pair_id}
        raw_rank = runs["RAW_ZH"]["first_relevant_rank"]
        comparisons.append({"pair_id": pair_id, "runs": runs, "alias_improvement": None if raw_rank is None or runs["ZH_ALIAS"]["first_relevant_rank"] is None else raw_rank - runs["ZH_ALIAS"]["first_relevant_rank"], "mixed_improvement": None if raw_rank is None or runs["ZH_EN_MIXED"]["first_relevant_rank"] is None else raw_rank - runs["ZH_EN_MIXED"]["first_relevant_rank"], "english_gap": None if raw_rank is None or runs["EN_CANONICAL"]["first_relevant_rank"] is None else raw_rank - runs["EN_CANONICAL"]["first_relevant_rank"]})
    return {"variants": by_variant, "runs": rows, "pair_comparisons": comparisons}


def main():
    import chromadb
    from backend.app.core.config import settings
    from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
    from backend.app.rag.http_store import ChromaHttpEvidenceStore
    from backend.app.rag.retriever import ChromaHistoricalRetriever

    fixture_path = Path("backend/tests/fixtures/roman_republic_retrieval_eval.json")
    alias_path = Path("backend/tests/fixtures/roman_republic_retrieval_alias_ablation.json")
    items = json.loads(fixture_path.read_text(encoding="utf-8"))
    aliases = json.loads(alias_path.read_text(encoding="utf-8"))
    collection = chromadb.HttpClient(host=settings.rag_chroma_host, port=settings.rag_chroma_port).get_collection(settings.rag_collection)
    retriever = ChromaHistoricalRetriever(ChromaHttpEvidenceStore(collection, SentenceTransformerEmbeddingProvider(settings.rag_embedding_model, settings.rag_embedding_device, settings.rag_embedding_batch_size)))
    before = collection.count()
    report = run_ablation(retriever, items, aliases)
    report.update({"collection_count_before": before, "collection_count_after": collection.count(), "pair_count": 9, "runs_completed": len(report["runs"]), "alias_spec": aliases})
    Path("data/historical_sources/processed/chinese_query_ablation_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
