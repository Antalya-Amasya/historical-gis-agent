import logging

from backend.app.core.config import settings
from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET, DEFAULT_PER_INTENT_K
from backend.app.rag.retriever import HistoricalRetriever, ChromaHistoricalRetriever
from backend.app.rag.http_store import build_production_retriever

logger = logging.getLogger(__name__)


class SemanticRouteEvidenceRetriever(HistoricalRetriever):
    """Lazy consumer of the existing semantic index; failures leave route extraction empty."""

    def __init__(self):
        self._retriever: ChromaHistoricalRetriever | None = None

    def _get(self) -> ChromaHistoricalRetriever:
        if self._retriever is None:
            self._retriever = build_production_retriever(settings)
        return self._retriever

    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        try:
            return self._get().retrieve(query, top_k, filters)
        except Exception as exc:
            logger.warning("route_evidence_retrieval_unavailable error_type=%s", type(exc).__name__)
            return []

    def retrieve_with_coverage(
        self,
        query: str,
        budget: int = DEFAULT_COVERAGE_BUDGET,
        *,
        per_intent_k: int = DEFAULT_PER_INTENT_K,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        try:
            return self._get().retrieve_with_coverage(
                query,
                budget,
                per_intent_k=per_intent_k,
                filters=filters,
            )
        except Exception as exc:
            logger.warning("route_evidence_retrieval_unavailable error_type=%s", type(exc).__name__)
            return []
