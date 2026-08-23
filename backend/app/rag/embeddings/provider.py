from abc import ABC, abstractmethod
import hashlib
import math
import re


class EmbeddingProvider(ABC):
    """Provider boundary for local or remote semantic embeddings."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicHashEmbedding(EmbeddingProvider):
    """Offline fallback for tests and no-key demos; not the intended production default."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[\w']+", text.lower()):
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions] += 1
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            embeddings.append([value / norm for value in vector])
        return embeddings


class SemanticEmbeddingProvider(EmbeddingProvider):
    """Reserved integration point for a configured real embedding provider."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("No semantic embedding provider is configured.")
