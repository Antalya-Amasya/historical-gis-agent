from abc import ABC, abstractmethod
import logging

from backend.app.models import Evidence
from .store import ChromaEvidenceStore

logger = logging.getLogger(__name__)


class HistoricalRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]: ...


class EmptyHistoricalRetriever(HistoricalRetriever):
    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        return []


class ChromaHistoricalRetriever(HistoricalRetriever):
    def __init__(self, store, query_bridge=None):
        self.store = store
        self.query_bridge = query_bridge

    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        bridge_result = self.query_bridge.transform(query) if self.query_bridge else None
        if bridge_result:
            logger.debug("historical_query_bridge applied=%s version=%s matched=%s failure=%s", bridge_result.applied, bridge_result.bridge_version, bridge_result.matched_entries, bridge_result.failure)
        result = self.store.query(bridge_result.retrieval_query if bridge_result else query, top_k, filters)
        rows = zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        evidence = []
        for rank, (id_, text, meta, distance) in enumerate(rows, start=1):
            if not isinstance(meta, dict) or not text or not meta.get("author") or not meta.get("work"):
                raise ValueError("malformed_chroma_result")
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            locator = f"Book {meta['book']}" if meta.get("book") else "section unavailable"
            if page_start is not None:
                locator += f", p. {page_start}"
            raw_metadata = dict(meta)
            raw_metadata.update({"document_id": meta.get("document_id"), "distance": float(distance), "rank": rank})
            evidence.append(Evidence(id=id_, author=meta["author"], work=meta["work"], locator=locator, excerpt=text[:500], text=text, book=meta.get("book"), chapter=meta.get("chapter"), section=meta.get("section"), page_start=int(page_start) if page_start is not None else None, page_end=int(page_end) if page_end is not None else None, source_file=meta.get("source_file"), source_type=meta.get("source_type"), score=round(1 / (1 + float(distance)), 4), metadata=raw_metadata))
        return evidence
