"""Production alias data supplies retrieval hints, never historical authority."""
import json
from pathlib import Path

from backend.app.rag import query_bridge
from backend.app.rag.query_bridge import HistoricalQueryBridge
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.app.models import AgentState


def test_standard_chinese_campaign_and_english_form():
    bridge = HistoricalQueryBridge()
    assert "Battle of Cannae" in bridge.transform("坎尼会战").retrieval_query
    assert bridge.transform("Second Punic War").retrieval_query == "Second Punic War"
    assert "Second Punic War" in bridge.transform("第二次布匿战争").retrieval_query
    assert bridge.transform("第二次布匏战争").retrieval_query == "第二次布匏战争"
    assert bridge.transform("坦尼").retrieval_query == "坦尼"


def test_out_of_fixture_aliases_and_transliteration():
    fixture = json.loads((Path(__file__).parent / "fixtures/roman_republic_query_bridge_lexicon.json").read_text(encoding="utf-8"))
    forms = {e["chinese_form"] for e in fixture}
    for alias, target in [("特雷比亚会战", "Battle of Trebia"), ("Massilia", "Massalia"), ("提契努斯会战", "Battle of Ticinus")]:
        assert alias not in forms
        assert target in HistoricalQueryBridge().transform(alias).retrieval_query
    assert HistoricalQueryBridge().transform("Massilian").retrieval_query == "Massilian"


def test_default_does_not_consume_legacy_entries_or_test_files(monkeypatch):
    query_bridge._registry_entries.cache_clear()
    monkeypatch.setattr(query_bridge, "V1_ENTRIES", (("测试陷阱", ("Fixture Trap",), "person"),))
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert "tests" not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    assert "Battle of Cannae" in HistoricalQueryBridge().transform("坎尼会战").retrieval_query
    assert HistoricalQueryBridge().transform("测试陷阱").retrieval_query == "测试陷阱"
    query_bridge._registry_entries.cache_clear()


def test_new_production_record_is_eligible_without_bridge_code_change(monkeypatch):
    from backend.app.geography import place_registry
    query_bridge._registry_entries.cache_clear()
    monkeypatch.setattr(place_registry, "records", lambda: [{"canonical_name": "SyntheticHarbor", "aliases": ["测试港口"]}])
    assert "SyntheticHarbor" in HistoricalQueryBridge().transform("测试港口").retrieval_query
    query_bridge._registry_entries.cache_clear()


def test_unknown_queries_reach_retrieval_and_expansion_creates_no_authority():
    class Store:
        def __init__(self): self.queries = []
        def query(self, query, *args):
            self.queries.append(query)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    store = Store()
    retriever = ChromaHistoricalRetriever(store, HistoricalQueryBridge())
    for query in ["未知人物泽林的行程", "Trace Unlisted Mariner", "坎尼会战"]:
        state = AgentState(session_id="bridge", user_query=query)
        assert retriever.retrieve(query, 1) == []
        assert state.user_query == query
        assert state.historical_events == [] and state.historical_route is None
        assert state.historical_evidence == []
    assert store.queries[:2] == ["未知人物泽林的行程", "Trace Unlisted Mariner"]
    assert "Battle of Cannae" in store.queries[2]


def test_expansion_is_bounded_and_normalized():
    entries = tuple(("名", (f"Alias {i}",), "entity") for i in range(30))
    result = HistoricalQueryBridge(entries=entries).transform("名")
    assert len(result.retrieval_query.split("Alias")) == 17
    assert "Massalia" in HistoricalQueryBridge().transform("Ｍａｓｓｉｌｉａ").retrieval_query
