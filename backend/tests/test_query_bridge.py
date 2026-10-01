from backend.app.rag.http_store import canonical_e5_query
from backend.app.rag.http_store import ChromaHttpEvidenceStore
from backend.app.rag.query_bridge import HistoricalQueryBridge, _registry_entries
from backend.app.rag.retriever import ChromaHistoricalRetriever


def test_registry_bridge_preserves_original_and_e5_prefix():
    for query, expected in (("凯撒与庞培的内战", "caesar"), ("汉尼拔翻越阿尔卑斯山", "alps"), ("第二次布匿战争", "Second Punic War")):
        result = HistoricalQueryBridge().transform(query)
        assert result.original_query == query
        assert result.retrieval_query.startswith(query)
        assert expected in result.retrieval_query
        assert result.applied
        assert canonical_e5_query(result.retrieval_query).count("query: ") == 1


def test_bridge_applicability_disable_and_original_preservation():
    bridge = HistoricalQueryBridge()
    assert bridge.transform("Caesar and Pompey").applied is False
    assert bridge.transform("未知中文问题").retrieval_query == "未知中文问题"
    assert HistoricalQueryBridge(enabled=False).transform("凯撒与庞培的内战").retrieval_query == "凯撒与庞培的内战"
    result = bridge.transform("汉尼拔翻越阿尔卑斯山")
    assert result.retrieval_query.count("Hannibal's Alpine Crossing") == 1
    assert bridge.transform("罗马行军").retrieval_query == "罗马行军"


def test_route_intent_bridge_is_generic_and_contains_no_historical_place_hint():
    result = HistoricalQueryBridge().transform("展示汉尼拔218 BCE路线")
    assert "hannibal" in result.retrieval_query
    assert result.retrieval_query.startswith("展示汉尼拔218 BCE路线")
    assert "Alps" not in result.retrieval_query and "Padus" not in result.retrieval_query


def test_bridge_failure_falls_back_and_entry_forms_have_no_storage_hints():
    bad = (("凯撒", None, "person"),)
    result = HistoricalQueryBridge(entries=bad).transform("凯撒")
    assert result.retrieval_query == "凯撒" and result.failure == "TypeError"
    all_forms = [form for _, forms, _ in _registry_entries() for form in forms]
    assert not any("_" in form or len(form) == 32 for form in all_forms)
    assert not any(form in {"Appian", "Sallust"} for form in all_forms)


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
    assert embedding.values == ["query: 凯撒与庞培的内战 caesar"]
    assert not hasattr(collection, "add") and not hasattr(collection, "upsert")
