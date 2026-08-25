from pathlib import Path

from backend.app.rag.embeddings.provider import DeterministicHashEmbedding
from backend.app.rag.ingestion.models import TextChunk
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.app.rag.store import ChromaEvidenceStore


def caesar_chunk(identifier: str, book: str, chapter: str, text: str) -> TextChunk:
    return TextChunk(identifier, text, {"author": "Julius Caesar", "work": "De Bello Gallico", "book": book, "chapter": chapter, "page_start": 1, "page_end": 1, "source": "test corpus", "source_file": "caesar.epub", "source_type": "primary_source_epub", "corpus_id": "caesar_gallic_war", "provenance": "manifest_declared_epub", "chunk_strategy": "structure_aware"})


def test_caesar_collection_retains_metadata_and_does_not_mix_corpora(tmp_path: Path) -> None:
    metadata = {"corpus_id": "caesar_gallic_war", "embedding_provider": "deterministic_hash", "embedding_model": "test", "embedding_dimension": 256, "provenance": "manifest_declared_epub"}
    store = ChromaEvidenceStore(tmp_path / "caesar", "caesar_de_bello_gallico_semantic", DeterministicHashEmbedding(), metadata)
    store.rebuild([caesar_chunk("book-1", "1", "1", "Gaul is divided into three parts. Caesar opposed the Helvetii migration."), caesar_chunk("book-7", "7", "1", "Alesia was besieged by Caesar in Gaul.")])

    assert store.collection.metadata["corpus_id"] == "caesar_gallic_war"
    evidence = ChromaHistoricalRetriever(store).retrieve("Gaul divided", top_k=2, filters={"corpus_id": "caesar_gallic_war"})
    assert evidence and all(item.author == "Julius Caesar" and item.work == "De Bello Gallico" for item in evidence)
    assert all(item.metadata["corpus_id"] == "caesar_gallic_war" and item.metadata["provenance"] == "manifest_declared_epub" for item in evidence)
    assert evidence[0].book == "1" and evidence[0].chapter == "1"


def test_caesar_collection_filter_rejects_other_corpus_metadata(tmp_path: Path) -> None:
    store = ChromaEvidenceStore(tmp_path, "caesar_only", DeterministicHashEmbedding(), {"corpus_id": "caesar_gallic_war"})
    store.rebuild([caesar_chunk("caesar", "7", "1", "Alesia and Caesar."), TextChunk("polybius", "Hannibal crossed the Alps.", {"author": "Polybius", "work": "Histories", "book": "3", "chapter": "50", "page_start": 1, "page_end": 1, "source_file": "polybius.pdf", "source_type": "primary_source", "corpus_id": "polybius_histories", "provenance": "pdf"})])

    evidence = ChromaHistoricalRetriever(store).retrieve("Alesia", filters={"corpus_id": "caesar_gallic_war"})
    assert evidence and all(item.author not in {"Polybius", "Livy"} for item in evidence)
    assert all(item.book == "7" and item.chapter == "1" for item in evidence)
