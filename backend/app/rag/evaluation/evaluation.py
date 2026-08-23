from dataclasses import dataclass
from pathlib import Path
import json

from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.app.rag.store import ChromaEvidenceStore


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    author: str
    book: str


CASES = [
    EvaluationCase("What difficulties did Hannibal face while crossing the Alps?", "Polybius", "3"),
    EvaluationCase("How does Polybius describe Hannibal's Alpine crossing?", "Polybius", "3"),
    EvaluationCase("What does Livy say about Hannibal's route into Italy?", "Livy", "21"),
    EvaluationCase("What role did local tribes play during the crossing?", "Polybius", "3"),
    EvaluationCase("What evidence is given about snow terrain or losses?", "Livy", "21"),
    EvaluationCase("汉尼拔翻越阿尔卑斯时遇到了哪些困难？", "Polybius", "3"),
    EvaluationCase("李维如何描述汉尼拔进入意大利？", "Livy", "21"),
]


def recall_at_k(retriever: ChromaHistoricalRetriever, top_k: int = 5) -> dict:
    hits = []
    for case in CASES:
        evidence = retriever.retrieve(case.query, top_k)
        hit = any(item.author == case.author and item.book == case.book for item in evidence)
        hits.append({"query": case.query, "expected": f"{case.author} Book {case.book}", "hit": hit})
    return {"top_k": top_k, "recall_at_k": sum(item["hit"] for item in hits) / len(hits), "cases": hits}


def write_report(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
