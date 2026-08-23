from abc import ABC, abstractmethod

from backend.app.models import Evidence
from .store import ChromaEvidenceStore


class HistoricalRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]: ...


class EmptyHistoricalRetriever(HistoricalRetriever):
    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        return []


class ChromaHistoricalRetriever(HistoricalRetriever):
    def __init__(self, store: ChromaEvidenceStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        result = self.store.query(query, top_k, filters)
        return [
            Evidence(id=id_, author=meta["author"], work=meta["work"], locator=f"Book {meta.get('book', 'unknown')}, p. {meta['page_start']}", excerpt=text[:500], text=text, book=meta.get("book"), chapter=meta.get("chapter"), section=meta.get("section"), page_start=int(meta["page_start"]), page_end=int(meta["page_end"]), source_file=meta["source_file"], source_type=meta["source_type"], score=round(1 / (1 + float(distance)), 4), metadata=meta)
            for id_, text, meta, distance in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]
