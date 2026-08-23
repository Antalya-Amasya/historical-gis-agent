from pathlib import Path

from backend.app.rag.ingestion.chunking import fixed_window, structure_aware
from backend.app.rag.ingestion.cleaners import clean_text
from backend.app.rag.ingestion.models import PageDocument
from backend.app.rag.ingestion.structure import annotate_structure


def test_cleaner_joins_hyphenated_line_breaks() -> None:
    assert clean_text("Han-\nnibal  crossed") == "Hannibal crossed"


def test_structure_detects_book_and_chapter() -> None:
    pages = [PageDocument("x.pdf", 1, "BOOK III\nCHAPTER 50\nText", "good", False, 0)]
    record = annotate_structure(pages)[0]
    assert record["book"] == "3"
    assert record["chapter"] == "50"


def test_structure_aware_chunk_has_required_metadata() -> None:
    pages = [PageDocument("x.pdf", 7, "BOOK III\nCHAPTER 50\n" + "Hannibal " * 80, "good", False, 0)]
    metadata = {"author": "Polybius", "work": "Histories", "source_file": "x.pdf", "source_type": "primary_source", "language": "en", "topic": "Hannibal", "campaign": "Second Punic War"}
    chunk = structure_aware(annotate_structure(pages), metadata, size=200)[0]
    for key in ("author", "work", "book", "chapter", "page_start", "page_end", "source_file", "chunk_strategy"):
        assert key in chunk.metadata
    assert chunk.metadata["chunk_strategy"] == "structure_aware"


def test_fixed_window_is_available_as_baseline() -> None:
    chunks = fixed_window("a" * 500, {"source_file": "x.pdf"}, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.metadata["chunk_strategy"] == "fixed_window" for chunk in chunks)
