from abc import ABC, abstractmethod
import logging

from backend.app.models import Evidence
from .coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    DEFAULT_PER_INTENT_K,
    merge_coverage_results,
)
from .store import ChromaEvidenceStore
from .evidence_ranking import diversify_route_evidence, rerank_evidence
from .lexical_index import derive_passages
from .retrieval_intents import decompose_movement_query

logger = logging.getLogger(__name__)


class HistoricalRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]: ...

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

    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        bridge_result = self.query_bridge.transform(query) if self.query_bridge else None
        if bridge_result:
            logger.debug("historical_query_bridge applied=%s version=%s matched=%s failure=%s", bridge_result.applied, bridge_result.bridge_version, bridge_result.matched_entries, bridge_result.failure)
        # The final API contract remains ``top_k``.  A modest, bounded pool
        # lets deterministic structure-aware ordering distinguish body prose
        # from explicit EPUB contents/navigation artefacts.
        candidate_k = min(60, max(20, top_k * 3))
        # Lexical recall needs a slightly wider, but still bounded, pool: it
        # compensates for ANN misses without asking the semantic index for 500
        # approximate neighbors.
        lexical_k = min(100, max(30, top_k * 5))
        result = self.store.query(bridge_result.retrieval_query if bridge_result else query, candidate_k, filters)
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
        lexical_candidates = getattr(self.store, "lexical_candidates", lambda *_args, **_kwargs: [])(bridge_result.retrieval_query if bridge_result else query, lexical_k, filters)
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
        selected = []
        ranked = rerank_evidence(query, list(evidence_by_id.values()))
        for item in diversify_route_evidence(query, ranked):
            source = item.metadata.get("source_chunk_id")
            start, end = item.metadata.get("passage_start"), item.metadata.get("passage_end")
            overlapping = False
            if source is not None and start is not None and end is not None:
                for prior in selected:
                    if prior.metadata.get("source_chunk_id") != source:
                        continue
                    a, b = prior.metadata.get("passage_start"), prior.metadata.get("passage_end")
                    if a is not None and b is not None and max(start, a) < min(end, b):
                        overlapping = True
                        break
            if not overlapping:
                selected.append(item)
            if len(selected) == top_k:
                break
        return [item.model_copy(update={"metadata": {**item.metadata, "rank": rank}}) for rank, item in enumerate(selected, 1)]

    def retrieve_with_coverage(
        self,
        query: str,
        budget: int = DEFAULT_COVERAGE_BUDGET,
        *,
        per_intent_k: int = DEFAULT_PER_INTENT_K,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        """Coverage-oriented retrieval for movement route queries."""
        intents = decompose_movement_query(query)
        if len(intents) <= 1:
            return self.retrieve(query, min(budget, 20), filters)
        intent_results: list[tuple] = []
        for intent in intents:
            items = self.retrieve(intent.query, per_intent_k, filters)
            intent_results.append((intent, items))
        return merge_coverage_results(query, intent_results, budget)
