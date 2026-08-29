"""Read-only HTTP evidence store for the Roman Republic v2 corpus."""
from __future__ import annotations


def canonical_e5_query(text: str) -> str:
    """Apply the E5 query prefix exactly once."""
    normalized = text.strip()
    return normalized if normalized.startswith("query: ") else f"query: {normalized}"


class ChromaHttpEvidenceStore:
    """A deliberately read-only adapter over an injected Chroma collection."""

    def __init__(self, collection, embedding):
        self.collection = collection
        self.embedding = embedding
        self._lexical_index = None

    def query(self, query: str, top_k: int, filters: dict[str, str] | None = None) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        where = None
        if filters:
            predicates = [{key: {"$eq": value}} for key, value in filters.items() if value]
            where = predicates[0] if len(predicates) == 1 else ({"$and": predicates} if predicates else None)
        return self.collection.query(
            query_embeddings=self.embedding.embed([canonical_e5_query(query)]),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def lexical_candidates(self, query: str, top_k: int, filters: dict[str, str] | None = None):
        from .lexical_index import LexicalEvidenceIndex
        # Minimal test doubles and deliberately query-only collections remain
        # valid semantic stores; they simply have no corpus read API for the
        # supplementary lexical channel.
        if not hasattr(self.collection, "get"):
            return []
        if self._lexical_index is None:
            self._lexical_index = LexicalEvidenceIndex(self.collection)
        return self._lexical_index.query(query, top_k, filters)


def build_production_retriever(settings):
    """The single production dependency factory shared by API and Agent."""
    import chromadb
    from .embeddings.provider import SentenceTransformerEmbeddingProvider
    from .retriever import ChromaHistoricalRetriever
    from .query_bridge import HistoricalQueryBridge
    provider = SentenceTransformerEmbeddingProvider(
        settings.rag_embedding_model, settings.rag_embedding_device, settings.rag_embedding_batch_size
    )
    client = chromadb.HttpClient(host=settings.rag_chroma_host, port=settings.rag_chroma_port)
    return ChromaHistoricalRetriever(ChromaHttpEvidenceStore(client.get_collection(settings.rag_collection), provider), HistoricalQueryBridge(settings.rag_query_bridge_enabled))
