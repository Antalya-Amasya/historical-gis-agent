"""Frozen, evaluation-only Lexicon V2 expansion experiment."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from backend.app.rag.evaluation.chinese_query_bridge_experiment import _retrieve, aggregate, transform as transform_v1, validate_lexicon as validate_v1

V2_TYPES = {"PERSON", "PLACE", "POLITICAL_EVENT", "MILITARY_EVENT", "MILITARY_ACTION", "MOVEMENT", "GEOGRAPHIC_ACTION", "POLITICAL_ACTION", "CONFLICT_RELATION", "TEMPORAL_CONCEPT"}
FORBIDDEN = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-f0-9]{32}\b", re.I)
AUTHORS = {"Appian", "Cassius Dio", "Livy", "Plutarch", "Polybius", "Sallust", "Suetonius"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_v2(expansion):
    errors = []
    rejected = []
    for entry in expansion.get("entries", []):
        if not all(entry.get(key) for key in ("chinese_form", "english_forms", "type", "rationale", "derivation_source")) or entry.get("benchmark_specific") is not False:
            errors.append(f"invalid_entry:{entry.get('chinese_form')}"); rejected.append({"entry": entry.get("chinese_form"), "reason": "missing_fields_or_not_general"}); continue
        if entry["type"] not in V2_TYPES:
            errors.append(f"invalid_type:{entry['chinese_form']}")
        if any(FORBIDDEN.search(form) or form in AUTHORS for form in entry["english_forms"]):
            errors.append(f"leakage:{entry['chinese_form']}"); rejected.append({"entry": entry["chinese_form"], "reason": "forbidden_hint"})
    return errors, rejected


def transform_v2(query, v1, additions):
    # Preserve frozen V1 mappings, then append only frozen general V2 mappings.
    base, base_matched = transform_v1(query, "", v1, "CONCEPT_BRIDGE")
    matched = [entry for entry in additions if entry["chinese_form"] in query]
    forms = [form for entry in matched for form in entry["english_forms"]]
    return " ".join([base, *forms]), base_matched + matched


def classify(v1, v2, oracle):
    if v1["hit_at"]["5"] and not v2["hit_at"]["5"]: return "REGRESSION"
    if not v1["hit_at"]["5"] and v2["hit_at"]["5"]: return "NEW_GENERALIZABLE_RECOVERY"
    if v1["hit_at"]["5"] and v2["hit_at"]["5"] and (v2["first_relevant_rank"] or 99) < (v1["first_relevant_rank"] or 99): return "RANK_IMPROVEMENT"
    if not v2["hit_at"]["10"] and oracle["hit_at"]["10"]: return "STILL_ORACLE_ONLY"
    if v1["hit_at"]["5"] == v2["hit_at"]["5"]: return "V1_RECOVERY_PRESERVED" if v1["hit_at"]["5"] else "NO_CHANGE"
    return "PERSISTENT_FAILURE"


def run(retriever, items, v1, additions):
    grouped = {}
    for item in items:
        if item.get("strict") and item.get("pair_id"): grouped.setdefault(item["pair_id"], {})[item["language"]] = item
    rows, classifications = [], []
    for pair_id, pair in sorted(grouped.items()):
        zh, en = pair["zh"], pair["en"]
        variants = {"RAW_ZH": (zh["query"], []), "CONCEPT_BRIDGE_V1": transform_v1(zh["query"], en["query"], v1, "CONCEPT_BRIDGE"), "CONCEPT_BRIDGE_V2": transform_v2(zh["query"], v1, additions), "ORACLE_MIXED_CONTROL": (f"{zh['query']} | {en['query']}", [])}
        pair_rows = {}
        for variant, (query, matched) in variants.items():
            row = _retrieve(retriever, pair_id, variant, zh["query"], query, matched, zh["expected_document_ids"]); row["lexicon_version"] = "v1" if variant == "CONCEPT_BRIDGE_V1" else ("v2" if variant == "CONCEPT_BRIDGE_V2" else None); rows.append(row); pair_rows[variant] = row
        classifications.append({"pair_id": pair_id, "classification": classify(pair_rows["CONCEPT_BRIDGE_V1"], pair_rows["CONCEPT_BRIDGE_V2"], pair_rows["ORACLE_MIXED_CONTROL"])})
    metrics = {variant: aggregate([row for row in rows if row["variant"] == variant]) for variant in ("RAW_ZH", "CONCEPT_BRIDGE_V1", "CONCEPT_BRIDGE_V2", "ORACLE_MIXED_CONTROL")}
    return rows, metrics, classifications


def main():
    import chromadb
    from backend.app.core.config import settings
    from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
    from backend.app.rag.http_store import ChromaHttpEvidenceStore
    from backend.app.rag.retriever import ChromaHistoricalRetriever
    fixture = Path("backend/tests/fixtures/roman_republic_retrieval_eval.json")
    v1_path = Path("backend/tests/fixtures/roman_republic_query_bridge_lexicon.json")
    v2_path = Path("backend/tests/fixtures/roman_republic_query_bridge_lexicon_v2.json")
    items, v1, expansion = json.loads(fixture.read_text(encoding="utf-8")), json.loads(v1_path.read_text(encoding="utf-8")), json.loads(v2_path.read_text(encoding="utf-8"))
    errors, rejected = validate_v2(expansion)
    if validate_v1(v1) or errors: raise ValueError("lexicon_audit_failed")
    collection = chromadb.HttpClient(host=settings.rag_chroma_host, port=settings.rag_chroma_port).get_collection(settings.rag_collection)
    retriever = ChromaHistoricalRetriever(ChromaHttpEvidenceStore(collection, SentenceTransformerEmbeddingProvider(settings.rag_embedding_model, settings.rag_embedding_device, settings.rag_embedding_batch_size)))
    before = collection.count(); rows, metrics, classes = run(retriever, items, v1, expansion["entries"])
    secondary = []
    for item in items:
        if item.get("strict") and item.get("language") == "zh" and not item.get("pair_id"):
            for variant, transform in (("RAW_ZH", lambda: (item["query"], [])), ("CONCEPT_BRIDGE_V1", lambda: transform_v1(item["query"], "", v1, "CONCEPT_BRIDGE")), ("CONCEPT_BRIDGE_V2", lambda: transform_v2(item["query"], v1, expansion["entries"]))):
                query, matched = transform(); secondary.append(_retrieve(retriever, item["id"], variant, item["query"], query, matched, item["expected_document_ids"]))
    typo_expected = next(item["expected_document_ids"] for item in items if item["id"] == "c02"); typo = []
    for text in ("第二次布匏战争", "第二次布匿战争"):
        for variant, transform in (("RAW_ZH", lambda: (text, [])), ("CONCEPT_BRIDGE_V1", lambda: transform_v1(text, "", v1, "CONCEPT_BRIDGE")), ("CONCEPT_BRIDGE_V2", lambda: transform_v2(text, v1, expansion["entries"]))):
            query, matched = transform(); typo.append(_retrieve(retriever, text, variant, text, query, matched, typo_expected))
    report = {"experiment_configuration": {"top_k": 10, "variants": ["RAW_ZH", "CONCEPT_BRIDGE_V1", "CONCEPT_BRIDGE_V2", "ORACLE_MIXED_CONTROL"]}, "lexicon_hashes": {"v1": digest(v1_path), "v2_expansion": digest(v2_path)}, "fixture_hash": digest(fixture), "lexicon_audit": {"v1_count": len(v1), "v2_effective_count": len(v1) + len(expansion["entries"]), "new_entries_by_type": {kind: sum(entry["type"] == kind for entry in expansion["entries"]) for kind in sorted(V2_TYPES)}, "rejected_entries": rejected, "errors": errors}, "runs": rows, "metrics": metrics, "pair_classifications": classes, "secondary_sanity_runs": secondary, "secondary_sanity_metrics": {variant: aggregate([row for row in secondary if row["variant"] == variant]) for variant in ("RAW_ZH", "CONCEPT_BRIDGE_V1", "CONCEPT_BRIDGE_V2")}, "typo_sensitivity": typo, "runs_completed": len(rows), "collection_count_before": before, "collection_count_after": collection.count()}
    Path("data/historical_sources/processed/chinese_query_bridge_lexicon_expansion_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
