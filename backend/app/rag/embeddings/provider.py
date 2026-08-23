from abc import ABC, abstractmethod
import hashlib
import math
import re


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicHashEmbedding(EmbeddingProvider):
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions
    def embed(self, texts):
        result = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[\w]+", text.lower()):
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions] += 1
            norm = math.sqrt(sum(x*x for x in vector)) or 1
            result.append([x/norm for x in vector])
        return result


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.batch_size = min(batch_size, 16)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        batch = self.batch_size
        while True:
            try:
                return self.model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False).tolist()
            except RuntimeError as error:
                if self.device == "cuda" and "out of memory" in str(error).lower() and batch > 1:
                    batch = max(1, batch // 2)
                    continue
                if self.device == "cuda" and "out of memory" in str(error).lower():
                    self.device = "cpu"
                    self.model.to("cpu")
                    batch = min(batch, 8)
                    continue
                raise
