import pytest
from backend.app.rag.ingestion.page_types import classify_page, should_include_in_index
from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider


def test_page_types_and_inclusion():
    assert classify_page("Project Gutenberg", 500, "3") == "gutenberg_footer"
    assert not should_include_in_index("gutenberg_footer")
    assert should_include_in_index("unknown")


def test_semantic_provider_cpu_embedding():
    provider = SentenceTransformerEmbeddingProvider("intfloat/multilingual-e5-small", "cpu", 16)
    vector = provider.embed(["query: Hannibal crossed the Alps"])[0]
    assert provider.dimensions == 384
    assert len(vector) == 384
