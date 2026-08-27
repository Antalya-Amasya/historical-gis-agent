from backend.app.rag.http_store import canonical_e5_query
from backend.app.rag.http_store import ChromaHttpEvidenceStore
from backend.app.rag.query_bridge import HistoricalQueryBridge, V1_ENTRIES
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.app.rag.evaluation.chinese_query_bridge_experiment import transform as evaluation_transform


def test_chinese_concept_bridge_preserves_original_and_matches_v1_golden_transform():
    entries = [{"chinese_form": form, "english_forms": list(forms), "type": kind, "evidence_source": "frozen", "reason": "frozen"} for form, forms, kind in V1_ENTRIES]
    for query, english in (("凯撒与庞培的内战", "Caesar and Pompey in the civil war"), ("汉尼拔翻越阿尔卑斯山", "Hannibal crossing the Alps"), ("朱古达战争", "Jugurthine War")):
        result = HistoricalQueryBridge().transform(query)
        expected, _ = evaluation_transform(query, english, entries, "CONCEPT_BRIDGE")
        assert result.original_query == query
        assert result.retrieval_query == expected
        assert result.applied
        assert canonical_e5_query(result.retrieval_query).count("query: ") == 1


def test_bridge_applicability_disable_dedup_and_v2_only_term_exclusion():
    bridge = HistoricalQueryBridge()
    assert bridge.transform("Caesar and Pompey").applied is False
    assert bridge.transform("未知中文问题").retrieval_query == "未知中文问题"
    assert HistoricalQueryBridge(enabled=False).transform("凯撒与庞培的内战").retrieval_query == "凯撒与庞培的内战"
    result = bridge.transform("汉尼拔翻越阿尔卑斯山")
    assert result.retrieval_query.count("crossing") == 1
    assert "march" not in bridge.transform("罗马行军").retrieval_query


def test_bridge_failure_falls_back_and_entry_forms_have_no_storage_hints():
    bad = (("凯撒", None, "person"),)
    result = HistoricalQueryBridge(entries=bad).transform("凯撒")
    assert result.retrieval_query == "凯撒" and result.failure == "TypeError"
    all_forms = [form for _, forms, _ in V1_ENTRIES for form in forms]
    assert not any("_" in form or len(form) == 32 for form in all_forms)
    assert not any(form in {"Appian", "Livy", "Sallust"} for form in all_forms)


def test_shared_retriever_applies_bridge_only_to_read_only_store_query():
    class Embedding:
        def __init__(self): self.values = []
        def embed(self, values): self.values.extend(values); return [[0.1]]
    class Collection:
        def __init__(self): self.calls = []
        def query(self, **kwargs):
            self.calls.append(kwargs)
            return {"ids": [["x"]], "documents": [["text"]], "metadatas": [[{"document_id": "doc", "author": "Author", "work": "Work"}]], "distances": [[0.1]]}
    embedding, collection = Embedding(), Collection()
    ChromaHistoricalRetriever(ChromaHttpEvidenceStore(collection, embedding), HistoricalQueryBridge()).retrieve("凯撒与庞培的内战", 1)
    assert embedding.values == ["query: 凯撒与庞培的内战 Caesar Julius Caesar Pompey civil war"]
    assert not hasattr(collection, "add") and not hasattr(collection, "upsert")
