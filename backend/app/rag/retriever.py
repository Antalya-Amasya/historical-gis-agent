from abc import ABC, abstractmethod

from backend.app.models import Evidence


class HistoricalRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> list[Evidence]: ...


class EmptyHistoricalRetriever(HistoricalRetriever):
    """Phase 0 placeholder; Chroma-backed retrieval begins in Phase 2."""

    def retrieve(self, query: str) -> list[Evidence]:
        return []

