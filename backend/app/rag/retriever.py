from abc import ABC, abstractmethod
import logging

from backend.app.models import Evidence
from .coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    DEFAULT_PER_INTENT_K,
    DEFAULT_RAW_OBSERVATION_K,
    merge_coverage_results,
    passages_overlap,
)
from .store import ChromaEvidenceStore
from .evidence_ranking import diversify_route_evidence, rerank_evidence
from .lexical_index import derive_passages
from .retrieval_intents import RetrievalIntent, decompose_movement_query

logger = logging.getLogger(__name__)


class HistoricalRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]: ...

    def retrieve_candidates(
        self,
        query: str,
        observation_k: int = DEFAULT_RAW_OBSERVATION_K,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        return self.retrieve(query, observation_k, filters)

    def retrieve_with_coverage(
        self,
        query: str,
        budget: int = DEFAULT_COVERAGE_BUDGET,
        *,
        per_intent_k: int = DEFAULT_PER_INTENT_K,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        return self.retrieve(query, min(budget, 20), filters)


class EmptyHistoricalRetriever(HistoricalRetriever):
    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        return []


class ChromaHistoricalRetriever(HistoricalRetriever):
    def __init__(self, store, query_bridge=None):
        self.store = store
        self.query_bridge = query_bridge

    def _collect_candidates(
        self,
        query: str,
        *,
        semantic_k: int,
        lexical_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        bridge_result = self.query_bridge.transform(query) if self.query_bridge else None
        if bridge_result:
            logger.debug("historical_query_bridge applied=%s version=%s matched=%s failure=%s", bridge_result.applied, bridge_result.bridge_version, bridge_result.matched_entries, bridge_result.failure)
        retrieval_query = bridge_result.retrieval_query if bridge_result else query
        result = self.store.query(retrieval_query, semantic_k, filters)
        rows = zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        evidence_by_id = {}
        for rank, (id_, text, meta, distance) in enumerate(rows, start=1):
            if not isinstance(meta, dict) or not text or not meta.get("author") or not meta.get("work"):
                raise ValueError("malformed_chroma_result")
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            locator = f"Book {meta['book']}" if meta.get("book") else "section unavailable"
            if page_start is not None:
                locator += f", p. {page_start}"
            raw_metadata = {**meta, "document_id": meta.get("document_id"), "distance": float(distance), "rank": rank, "vector_rank": rank, "semantic_candidate": True}
            for passage in derive_passages(id_, text, raw_metadata):
                passage_meta = passage.metadata
                evidence_by_id[passage.id] = Evidence(id=passage.id, author=meta["author"], work=meta["work"], locator=locator, excerpt=passage.text[:500], text=passage.text, book=meta.get("book"), chapter=meta.get("chapter"), section=meta.get("section"), page_start=int(page_start) if page_start is not None else None, page_end=int(page_end) if page_end is not None else None, source_file=meta.get("source_file"), source_type=meta.get("source_type"), score=round(1 / (1 + float(distance)), 4), metadata=passage_meta)
        lexical_candidates = getattr(self.store, "lexical_candidates", lambda *_args, **_kwargs: [])(retrieval_query, lexical_k, filters)
        for lexical in lexical_candidates:
            if lexical.id in evidence_by_id:
                evidence_by_id[lexical.id] = evidence_by_id[lexical.id].model_copy(update={"metadata": {**evidence_by_id[lexical.id].metadata, "lexical_candidate": True, "lexical_score": lexical.lexical_score}})
                continue
            meta = lexical.metadata
            if not meta.get("author") or not meta.get("work"):
                continue
            page_start, page_end = meta.get("page_start"), meta.get("page_end")
            locator = f"Book {meta['book']}" if meta.get("book") else "section unavailable"
            if page_start is not None:
                locator += f", p. {page_start}"
            metadata = {**meta, "document_id": meta.get("document_id"), "lexical_candidate": True, "lexical_score": lexical.lexical_score, "source_chunk_id": meta.get("source_chunk_id", lexical.id.split(":", 1)[0])}
            evidence_by_id[lexical.id] = Evidence(id=lexical.id, author=meta["author"], work=meta["work"], locator=locator, excerpt=lexical.text[:500], text=lexical.text, book=meta.get("book"), chapter=meta.get("chapter"), section=meta.get("section"), page_start=int(page_start) if page_start is not None else None, page_end=int(page_end) if page_end is not None else None, source_file=meta.get("source_file"), source_type=meta.get("source_type"), score=0.0, metadata=metadata)
        return list(evidence_by_id.values())

    def retrieve_candidates(
        self,
        query: str,
        observation_k: int = DEFAULT_RAW_OBSERVATION_K,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        semantic_k = min(DEFAULT_RAW_OBSERVATION_K, max(20, observation_k))
        lexical_k = min(100, max(30, observation_k))
        candidates = self._collect_candidates(query, semantic_k=semantic_k, lexical_k=lexical_k, filters=filters)
        ranked = rerank_evidence(query, candidates)
        return ranked[:observation_k]

    def _finalize_selection(self, query: str, candidates: list[Evidence], top_k: int) -> list[Evidence]:
        selected: list[Evidence] = []
        ranked = rerank_evidence(query, candidates)
        for item in diversify_route_evidence(query, ranked):
            if not any(passages_overlap(item, prior) for prior in selected):
                selected.append(item)
            if len(selected) == top_k:
                break
        return [item.model_copy(update={"metadata": {**item.metadata, "rank": rank}}) for rank, item in enumerate(selected, 1)]

    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        candidate_k = min(DEFAULT_RAW_OBSERVATION_K, max(20, top_k * 3))
        lexical_k = min(100, max(30, top_k * 5))
        candidates = self._collect_candidates(query, semantic_k=candidate_k, lexical_k=lexical_k, filters=filters)
        return self._finalize_selection(query, candidates, top_k)

    def retrieve_with_coverage(
        self,
        query: str,
        budget: int = DEFAULT_COVERAGE_BUDGET,
        *,
        per_intent_k: int = DEFAULT_PER_INTENT_K,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        """Coverage-oriented retrieval for movement route queries."""
        del per_intent_k
        intents = decompose_movement_query(query)
        budget = min(budget, DEFAULT_COVERAGE_BUDGET)
        if len(intents) <= 1:
            return self.retrieve(query, budget, filters)
        channels = [RetrievalIntent("CANONICAL", query), *intents]
        intent_results = [
            (intent, self.retrieve_candidates(intent.query, DEFAULT_RAW_OBSERVATION_K, filters))
            for intent in channels
        ]
        return merge_coverage_results(query, intent_results, budget)
