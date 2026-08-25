"""Offline retrieval evaluation for a manifest-declared Caesar corpus."""
from dataclasses import dataclass

from backend.app.rag.retriever import ChromaHistoricalRetriever


@dataclass(frozen=True)
class CaesarEvaluationCase:
    query: str
    expected_books: tuple[str, ...]


CAESAR_CASES = (
    CaesarEvaluationCase("凯撒在高卢战争第一年如何阻止赫尔维蒂人迁移？", ("1",)),
    CaesarEvaluationCase("阿莱西亚战役发生在哪里？", ("7",)),
    CaesarEvaluationCase("凯撒如何描述高卢地区的划分？", ("1",)),
    CaesarEvaluationCase("凯撒在不列颠进行了哪些军事行动？", ("4", "5")),
)


def evaluate_caesar_retrieval(retriever: ChromaHistoricalRetriever, top_k: int = 5) -> dict[str, object]:
    results = []
    for case in CAESAR_CASES:
        evidence = retriever.retrieve(case.query, top_k, filters={"corpus_id": "caesar_gallic_war"})
        results.append({
            "query": case.query,
            "expected_books": list(case.expected_books),
            "retrieved": [{"id": item.id, "corpus_id": item.metadata.get("corpus_id"), "book": item.book, "chapter": item.chapter, "author": item.author, "work": item.work} for item in evidence],
            "hit": any(item.book in case.expected_books for item in evidence),
        })
    return {"corpus_id": "caesar_gallic_war", "top_k": top_k, "cases": results}
