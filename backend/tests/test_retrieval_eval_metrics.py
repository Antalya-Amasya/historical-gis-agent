from types import SimpleNamespace

from backend.app.rag.evaluation.retrieval_eval import evaluate


class FakeRetriever:
    def retrieve(self, query, top_k):
        documents = ["doc-a", "other", "doc-b"]
        return [SimpleNamespace(id=f"chunk-{index}", metadata={"document_id": document}, author="Author", work="Work", score=0.5, text="text") for index, document in enumerate(documents, start=1)]


def test_hit_and_document_recall_have_distinct_multi_document_semantics():
    item = {"id": "q", "query": "query", "language": "en", "category": "multi_source", "strict": True, "expected_document_ids": ["doc-a", "doc-b"], "expected_authors": ["A", "B"]}
    row = evaluate(FakeRetriever(), [item])["items"][0]
    assert row["hit_at"]["1"] is True
    assert row["document_recall_at"]["1"] == 0.5
    assert row["document_recall_at"]["3"] == 1.0
    assert row["first_relevant_rank"] == 1
