from pathlib import Path

import chromadb

from .embeddings.provider import DeterministicHashEmbedding, EmbeddingProvider
from .ingestion.models import TextChunk


class ChromaEvidenceStore:
    def __init__(self, path: Path, collection_name: str = "historical_primary_sources", embedding: EmbeddingProvider | None = None, collection_metadata: dict[str, object] | None = None):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection_name
        self.collection_metadata = dict(collection_metadata or {})
        self.collection = self.client.get_or_create_collection(collection_name, metadata=self.collection_metadata or None)
        self.embedding = embedding or DeterministicHashEmbedding()

    def rebuild(self, chunks: list[TextChunk]) -> None:
        if self.collection.count():
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name, metadata=self.collection_metadata or None)
        if chunks:
            self.collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                embeddings=self.embedding.embed([chunk.text for chunk in chunks]),
            )

    def query(self, query: str, top_k: int, filters: dict[str, str] | None = None) -> dict:
        where = None
        if filters:
            predicates = [{key: {"$eq": value}} for key, value in filters.items() if value]
            where = predicates[0] if len(predicates) == 1 else ({"$and": predicates} if predicates else None)
        return self.collection.query(
            query_embeddings=self.embedding.embed([query]),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
