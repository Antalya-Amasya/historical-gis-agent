"""Evaluation-only deterministic bilingual query-bridge experiment."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

VARIANTS = ("RAW_ZH", "ENTITY_BRIDGE", "CONCEPT_BRIDGE", "STRUCTURED_BRIDGE", "ORACLE_MIXED_CONTROL")
ENTITY_TYPES = {"person", "place"}
CONCEPT_TYPES = {"concept", "event_entity"}
FORBIDDEN_OUTPUT = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-f0-9]{32}\b", re.I)
FORBIDDEN_AUTHORS = {"Appian", "Cassius Dio", "Livy", "Plutarch", "Polybius", "Sallust", "Suetonius"}


def validate_lexicon(entries):
    errors = []
    for entry in entries:
        if not all(entry.get(field) for field in ("chinese_form", "english_forms", "type", "evidence_source", "reason")):
            errors.append("lexicon:missing_required_field")
            continue
        if entry["type"] not in ENTITY_TYPES | CONCEPT_TYPES:
            errors.append(f"lexicon:invalid_type:{entry['chinese_form']}")
        for form in entry["english_forms"]:
            if FORBIDDEN_OUTPUT.search(form) or form in FORBIDDEN_AUTHORS:
                errors.append(f"lexicon:forbidden_output:{entry['chinese_form']}")
    return errors


def matched_entries(query, lexicon, allowed_types):
    return [entry for entry in lexicon if entry["type"] in allowed_types and entry["chinese_form"] in query]


def transform(query, english_query, lexicon, variant):
    entities = matched_entries(query, lexicon, ENTITY_TYPES)
    concepts = matched_entries(query, lexicon, CONCEPT_TYPES)
    entity_forms = [form for entry in entities for form in entry["english_forms"]]
    concept_forms = [form for entry in concepts for form in entry["english_forms"]]
    if variant == "RAW_ZH":
        return query, []
    if variant == "ENTITY_BRIDGE":
        return " ".join([query, *entity_forms]), entities
    if variant == "CONCEPT_BRIDGE":
        return " ".join([query, *entity_forms, *concept_forms]), entities + concepts
    if variant == "STRUCTURED_BRIDGE":
        return f"{query} | entities: {', '.join(entity_forms) or 'none'} | concepts: {', '.join(concept_forms) or 'none'}", entities + concepts
    if variant == "ORACLE_MIXED_CONTROL":
        return f"{query} | {english_query}", []
    raise ValueError(f"unknown_variant:{variant}")


def _retrieve(retriever, pair_id, variant, original, transformed, matched, expected, top_k=10):
    evidence = retriever.retrieve(transformed, top_k)
    documents = [item.metadata.get("document_id") for item in evidence]
    expected_set = set(expected)
    first = next((rank for rank, document_id in enumerate(documents, 1) if document_id in expected_set), None)
    def metrics(k):
        docs = set(documents[:k])
        return {"hit": bool(docs & expected_set), "document_recall": len(docs & expected_set) / len(expected_set), "unique_documents": len(docs), "max_chunks_same_document": max(Counter(documents[:k]).values(), default=0)}
    cuts = {str(k): metrics(k) for k in (1, 3, 5, 10)}
    return {"pair_id": pair_id, "variant": variant, "original_query": original, "transformed_query": transformed, "matched_lexicon_entries": matched, "expected_document_ids": expected, "first_relevant_rank": first, "mrr": 1 / first if first else 0, "hit_at": {key: value['hit'] for key, value in cuts.items()}, "document_recall_at": {key: value['document_recall'] for key, value in cuts.items()}, "unique_documents_at": {key: value['unique_documents'] for key, value in cuts.items()}, "max_chunks_same_document_at": {key: value['max_chunks_same_document'] for key, value in cuts.items()}, "hits": [{"rank": rank, "chunk_id": item.id, "document_id": item.metadata.get("document_id"), "author": item.author, "work": item.work, "distance": item.metadata.get("distance"), "score": item.score, "text_preview": item.text[:300] if item.text else None} for rank, item in enumerate(evidence, 1)]}


def aggregate(rows):
    return {"n": len(rows), "hit_at": {str(k): sum(row['hit_at'][str(k)] for row in rows) / len(rows) for k in (1, 3, 5, 10)}, "document_recall_at": {str(k): sum(row['document_recall_at'][str(k)] for row in rows) / len(rows) for k in (1, 3, 5, 10)}, "mrr": sum(row['mrr'] for row in rows) / len(rows), "crowding": {str(k): {"mean_unique_documents": sum(row['unique_documents_at'][str(k)] for row in rows) / len(rows), "mean_max_chunks_same_document": sum(row['max_chunks_same_document_at'][str(k)] for row in rows) / len(rows)} for k in (5, 10)}}


def run_experiment(retriever, items, lexicon):
    errors = validate_lexicon(lexicon)
    if errors:
        raise ValueError(";".join(errors))
    grouped = {}
    for item in items:
        if item.get("strict") and item.get("pair_id"):
            grouped.setdefault(item["pair_id"], {})[item["language"]] = item
    rows = []
    for pair_id, pair in sorted(grouped.items()):
        zh, en = pair["zh"], pair["en"]
        for variant in VARIANTS:
            transformed, matched = transform(zh["query"], en["query"], lexicon, variant)
            rows.append(_retrieve(retriever, pair_id, variant, zh["query"], transformed, matched, zh["expected_document_ids"]))
    metrics = {variant: aggregate([row for row in rows if row["variant"] == variant]) for variant in VARIANTS}
    return {"lexicon_validation_errors": errors, "runs": rows, "metrics": metrics}


def main():
    import chromadb
    from backend.app.core.config import settings
    from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
    from backend.app.rag.http_store import ChromaHttpEvidenceStore
    from backend.app.rag.retriever import ChromaHistoricalRetriever
    base = Path("backend/tests/fixtures")
    items = json.loads((base / "roman_republic_retrieval_eval.json").read_text(encoding="utf-8"))
    lexicon = json.loads((base / "roman_republic_query_bridge_lexicon.json").read_text(encoding="utf-8"))
    collection = chromadb.HttpClient(host=settings.rag_chroma_host, port=settings.rag_chroma_port).get_collection(settings.rag_collection)
    retriever = ChromaHistoricalRetriever(ChromaHttpEvidenceStore(collection, SentenceTransformerEmbeddingProvider(settings.rag_embedding_model, settings.rag_embedding_device, settings.rag_embedding_batch_size)))
    before = collection.count()
    report = run_experiment(retriever, items, lexicon)
    secondary_rows = []
    for item in items:
        if item.get("strict") and item.get("language") == "zh" and not item.get("pair_id"):
            for variant in VARIANTS[:-1]:
                transformed, matched = transform(item["query"], "", lexicon, variant)
                secondary_rows.append(_retrieve(retriever, item["id"], variant, item["query"], transformed, matched, item["expected_document_ids"]))
    typo_expected = next(item["expected_document_ids"] for item in items if item["id"] == "c02")
    sensitivity = []
    for label, query in (("FROZEN_TYPO", "第二次布匏战争"), ("CORRECTED_QUERY", "第二次布匿战争")):
        sensitivity.append(_retrieve(retriever, "second_punic_war_typo_sensitivity", label, query, query, [], typo_expected))
    report.update({"lexicon": lexicon, "typo_sensitivity": sensitivity, "secondary_sanity_runs": secondary_rows, "secondary_sanity_metrics": {variant: aggregate([row for row in secondary_rows if row["variant"] == variant]) for variant in VARIANTS[:-1]}, "runs_completed": len(report["runs"]), "collection_count_before": before, "collection_count_after": collection.count()})
    Path("data/historical_sources/processed/chinese_query_bridge_experiment_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
