from backend.app.rag.evaluation.ground_truth_validator import validate_items


def _item(item_id="q1", *, language="en", pair_id=None, documents=("doc-a",), chunks=("chunk-a",), strict=True):
    item = {
        "id": item_id, "query": item_id, "language": language, "category": "event", "strict": strict,
        "expected_document_ids": list(documents), "expected_authors": ["Author"] * len(documents),
        "ground_truth_provenance": [
            {"document_id": doc, "supporting_chunk_ids": [chunk], "text_preview": "evidence", "relevance": "substantive", "reason": "direct evidence"}
            for doc, chunk in zip(documents, chunks)
        ],
    }
    if pair_id:
        item["pair_id"] = pair_id
    return item


def _validate(items, chunks):
    return validate_items(items, chunks.get, min_strict=1, min_chinese_strict=0, min_bilingual_pairs=0)


def test_accepts_valid_single_document_strict_item():
    assert _validate([_item()], {"chunk-a": {"document_id": "doc-a"}}).valid


def test_accepts_valid_multi_document_strict_item():
    item = _item(documents=("doc-a", "doc-b"), chunks=("chunk-a", "chunk-b"))
    assert _validate([item], {"chunk-a": {"document_id": "doc-a"}, "chunk-b": {"document_id": "doc-b"}}).valid


def test_accepts_valid_bilingual_pair():
    en, zh = _item("en", pair_id="pair"), _item("zh", language="zh", pair_id="pair")
    assert _validate([en, zh], {"chunk-a": {"document_id": "doc-a"}}).valid


def test_rejects_missing_provenance():
    item = _item(); item.pop("ground_truth_provenance")
    assert "q1:provenance:missing" in _validate([item], {}).errors


def test_rejects_missing_supporting_chunk():
    assert "q1:supporting_chunk_missing:chunk-a" in _validate([_item()], {}).errors


def test_rejects_chunk_document_ownership_mismatch():
    assert "q1:chunk_document_mismatch:chunk-a" in _validate([_item()], {"chunk-a": {"document_id": "other"}}).errors


def test_rejects_expected_provenance_document_mismatch():
    item = _item(); item["expected_document_ids"] = ["other"]
    assert "q1:expected_provenance_document_mismatch" in _validate([item], {"chunk-a": {"document_id": "doc-a"}}).errors


def test_rejects_bilingual_document_and_provenance_mismatches():
    en, zh = _item("en", pair_id="pair"), _item("zh", language="zh", pair_id="pair", documents=("doc-b",), chunks=("chunk-b",))
    errors = _validate([en, zh], {"chunk-a": {"document_id": "doc-a"}, "chunk-b": {"document_id": "doc-b"}}).errors
    assert "pair:pair:expected_documents_mismatch" in errors
    assert "pair:pair:provenance_documents_mismatch" in errors


def test_rejects_duplicate_id():
    one, two = _item("same"), _item("same")
    assert "fixture:duplicate_id" in _validate([one, two], {"chunk-a": {"document_id": "doc-a"}}).errors


def test_rejects_each_minimum_gate():
    result = validate_items([_item()], {"chunk-a": {"document_id": "doc-a"}}.get, min_strict=2, min_chinese_strict=1, min_bilingual_pairs=1)
    assert {"gate:insufficient_strict", "gate:insufficient_chinese_strict", "gate:insufficient_bilingual_pairs"} <= set(result.errors)
