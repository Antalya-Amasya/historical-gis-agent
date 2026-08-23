from pathlib import Path

from backend.app.rag.ingestion.models import TextChunk
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.app.rag.store import ChromaEvidenceStore


def test_chroma_query_maps_to_evidence(tmp_path: Path) -> None:
    metadata = {"author": "Polybius", "work": "Histories", "book": "3", "chapter": "50", "section": "unknown", "page_start": 123, "page_end": 123, "source_file": "polybius.pdf", "source_type": "primary_source", "language": "en", "topic": "Hannibal", "campaign": "Second Punic War", "extraction_quality": "good", "chunk_strategy": "structure_aware"}
    store = ChromaEvidenceStore(tmp_path)
    store.rebuild([TextChunk("one", "Hannibal faced snow and hostile tribes in the Alps.", metadata)])
    evidence = ChromaHistoricalRetriever(store).retrieve("Hannibal snow", filters={"book": "3"})
    assert evidence[0].author == "Polybius"
    assert evidence[0].page_start == 123
    assert 0 < evidence[0].score <= 1
