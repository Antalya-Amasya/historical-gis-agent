"""Validation for the evaluation-only Roman Republic ground-truth fixture.

This module deliberately accepts a read-only chunk lookup callable.  It has no
dependency on the production retriever, embedding provider, or ingestion code.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


VALID_LANGUAGES = {"en", "zh"}
VALID_CATEGORIES = {"event", "person", "person_conflict", "multi_source", "geography_movement"}


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    counts: dict[str, int]

    @property
    def valid(self) -> bool:
        return not self.errors


def _add_if_blank(errors: list[str], value: Any, label: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()) or value == []:
        errors.append(f"{label}:missing")


def validate_items(
    items: list[dict[str, Any]],
    chunk_lookup: Callable[[str], dict[str, Any] | None],
    *,
    min_strict: int = 20,
    min_chinese_strict: int = 8,
    min_bilingual_pairs: int = 8,
) -> ValidationResult:
    """Validate semantic labels and provenance against read-only chunk metadata."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(items, list):
        return ValidationResult(["fixture:not_array"], warnings, {})

    ids = [item.get("id") for item in items if isinstance(item, dict)]
    queries = [item.get("query") for item in items if isinstance(item, dict)]
    if len(ids) != len(items) or any(not isinstance(item, dict) for item in items):
        errors.append("fixture:item_not_object")
    if len(set(ids)) != len(ids):
        errors.append("fixture:duplicate_id")
    if len(set(queries)) != len(queries):
        errors.append("fixture:duplicate_query_text")

    strict_items: list[dict[str, Any]] = []
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "<missing>"))
        _add_if_blank(errors, item.get("id"), f"{item_id}:id")
        _add_if_blank(errors, item.get("query"), f"{item_id}:query")
        if item.get("language") not in VALID_LANGUAGES:
            errors.append(f"{item_id}:language_invalid")
        if item.get("category") not in VALID_CATEGORIES:
            errors.append(f"{item_id}:category_invalid")
        if not isinstance(item.get("strict"), bool):
            errors.append(f"{item_id}:strict_not_bool")
            continue
        if item["strict"]:
            strict_items.append(item)
            _add_if_blank(errors, item.get("expected_document_ids"), f"{item_id}:expected_documents")
            _add_if_blank(errors, item.get("expected_authors"), f"{item_id}:expected_authors")
            provenance = item.get("ground_truth_provenance")
            _add_if_blank(errors, provenance, f"{item_id}:provenance")
            expected_docs = set(item.get("expected_document_ids") or [])
            provenance_docs: set[str] = set()
            for entry in provenance or []:
                if not isinstance(entry, dict):
                    errors.append(f"{item_id}:provenance_not_object")
                    continue
                document_id = entry.get("document_id")
                _add_if_blank(errors, document_id, f"{item_id}:provenance_document")
                _add_if_blank(errors, entry.get("supporting_chunk_ids"), f"{item_id}:supporting_chunks")
                _add_if_blank(errors, entry.get("text_preview"), f"{item_id}:text_preview")
                _add_if_blank(errors, entry.get("reason"), f"{item_id}:reason")
                if entry.get("relevance") != "substantive":
                    errors.append(f"{item_id}:provenance_not_substantive")
                if document_id:
                    provenance_docs.add(document_id)
                for chunk_id in entry.get("supporting_chunk_ids") or []:
                    metadata = chunk_lookup(chunk_id)
                    if metadata is None:
                        errors.append(f"{item_id}:supporting_chunk_missing:{chunk_id}")
                    elif metadata.get("document_id") != document_id:
                        errors.append(f"{item_id}:chunk_document_mismatch:{chunk_id}")
            if provenance_docs != expected_docs:
                errors.append(f"{item_id}:expected_provenance_document_mismatch")
        pair_id = item.get("pair_id")
        if pair_id is not None:
            if not isinstance(pair_id, str) or not pair_id.strip():
                errors.append(f"{item_id}:pair_id_malformed")
            else:
                pair_groups[pair_id].append(item)

    strict_zh = [item for item in strict_items if item.get("language") == "zh"]
    bilingual_pairs = 0
    for pair_id, pair_items in pair_groups.items():
        languages = [item.get("language") for item in pair_items]
        if len(pair_items) != 2 or set(languages) != {"en", "zh"}:
            errors.append(f"pair:{pair_id}:language_shape_invalid")
            continue
        if not all(item.get("strict") is True for item in pair_items):
            errors.append(f"pair:{pair_id}:not_all_strict")
        en, zh = sorted(pair_items, key=lambda item: item["language"])
        if en.get("expected_document_ids") != zh.get("expected_document_ids"):
            errors.append(f"pair:{pair_id}:expected_documents_mismatch")
        if en.get("expected_authors") != zh.get("expected_authors"):
            errors.append(f"pair:{pair_id}:expected_authors_mismatch")
        en_docs = {entry.get("document_id") for entry in en.get("ground_truth_provenance") or []}
        zh_docs = {entry.get("document_id") for entry in zh.get("ground_truth_provenance") or []}
        if en_docs != zh_docs:
            errors.append(f"pair:{pair_id}:provenance_documents_mismatch")
        bilingual_pairs += 1

    counts = {"total": len(items), "strict": len(strict_items), "chinese_strict": len(strict_zh), "bilingual_pairs": bilingual_pairs}
    if counts["strict"] < min_strict:
        errors.append("gate:insufficient_strict")
    if counts["chinese_strict"] < min_chinese_strict:
        errors.append("gate:insufficient_chinese_strict")
    if counts["bilingual_pairs"] < min_bilingual_pairs:
        errors.append("gate:insufficient_bilingual_pairs")

    repeated_terms = [query for query, count in Counter(queries).items() if count > 1]
    if repeated_terms:
        warnings.append("fixture:near_duplicate_review_recommended")
    return ValidationResult(errors, warnings, counts)


def load_fixture(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        raise ValueError("fixture:not_array")
    return loaded


def validate_chroma_fixture(path: str | Path, collection: Any) -> ValidationResult:
    """Validate a fixture against a Chroma collection using only ``get`` calls."""
    items = load_fixture(path)
    chunk_ids = sorted({chunk_id for item in items for entry in item.get("ground_truth_provenance", []) for chunk_id in entry.get("supporting_chunk_ids", [])})
    response = collection.get(ids=chunk_ids, include=["metadatas"])
    metadata_by_id = dict(zip(response.get("ids", []), response.get("metadatas", [])))
    return validate_items(items, metadata_by_id.get)


def main() -> None:
    import chromadb
    from backend.app.core.config import settings

    fixture = Path("backend/tests/fixtures/roman_republic_retrieval_eval.json")
    collection = chromadb.HttpClient(host=settings.rag_chroma_host, port=settings.rag_chroma_port).get_collection(settings.rag_collection)
    result = validate_chroma_fixture(fixture, collection)
    print(json.dumps({"valid": result.valid, "counts": result.counts, "errors": result.errors, "warnings": result.warnings}, ensure_ascii=False, indent=2))
    if not result.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
