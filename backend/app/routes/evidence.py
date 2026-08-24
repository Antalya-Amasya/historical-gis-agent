from pathlib import Path
import logging

from backend.app.core.config import settings
from backend.app.models import Evidence
from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
from backend.app.rag.retriever import HistoricalRetriever, ChromaHistoricalRetriever
from backend.app.rag.store import ChromaEvidenceStore

logger = logging.getLogger(__name__)


class SemanticRouteEvidenceRetriever(HistoricalRetriever):
    """Lazy consumer of the existing semantic index; failures leave route extraction empty."""

    def __init__(self):
        self._retriever: ChromaHistoricalRetriever | None = None

    def _get(self) -> ChromaHistoricalRetriever:
        if self._retriever is None:
            provider = SentenceTransformerEmbeddingProvider(settings.rag_embedding_model, settings.rag_embedding_device, settings.rag_embedding_batch_size)
            store = ChromaEvidenceStore(Path("data/chroma_semantic"), "historical_primary_sources_semantic", provider)
            self._retriever = ChromaHistoricalRetriever(store)
        return self._retriever

    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        try:
            return self._get().retrieve(query, top_k, filters)
        except Exception as exc:
            logger.warning("route_evidence_retrieval_unavailable error_type=%s", type(exc).__name__)
            return []
