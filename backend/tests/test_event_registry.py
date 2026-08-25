import json
from pathlib import Path

import pytest

from backend.app.rag.event_registry import CorpusEvidenceCatalog, EventRegistryValidationError, HistoricalEventRecord, HistoricalEventRegistry
from backend.app.rag.ingestion.models import TextChunk


def chunk(identifier="caesar-1", corpus="caesar_gallic_war", book="1", chapter="12"):
    return TextChunk(identifier, "Explicit corpus chunk", {"corpus_id": corpus, "book": book, "chapter": chapter})


def record(**changes):
    base = {"event_id": "helvetii-migration", "title": "Helvetii migration", "corpus_id": "caesar_gallic_war", "period": "58 BCE", "description": "Reviewed description", "involved_places": ["Helvetii"], "evidence_refs": ["caesar-1"], "source_book": "1", "source_chapter": "12", "confidence": 0.8}
    return HistoricalEventRecord(**(base | changes))


def test_registry_accepts_reviewed_record_with_matching_chunk_provenance():
    registry = HistoricalEventRegistry([record()], CorpusEvidenceCatalog.from_chunks([chunk()]))
    assert registry.get("helvetii-migration").involved_places == ["Helvetii"]


@pytest.mark.parametrize(("record_change", "chunks", "message"), [
    ({"evidence_refs": ["missing"]}, [chunk()], "unknown evidence_ref"),
    ({}, [chunk(corpus="polybius_histories")], "different corpus"),
    ({"source_chapter": "99"}, [chunk()], "source_book/source_chapter"),
])
def test_registry_rejects_invalid_evidence_provenance(record_change, chunks, message):
    with pytest.raises(EventRegistryValidationError, match=message):
        HistoricalEventRegistry([record(**record_change)], CorpusEvidenceCatalog.from_chunks(chunks))


def test_record_requires_nonempty_places_and_source_locator():
    with pytest.raises(Exception):
        record(involved_places=[])
    with pytest.raises(Exception):
        record(source_chapter="")


def test_caesar_registry_json_contains_the_reviewed_examples():
    data = json.loads((Path("backend/app/rag/registries/caesar_events.json")).read_text(encoding="utf-8"))
    assert {item["event_id"] for item in data["events"]} == {"helvetii-migration", "battle-of-bibracte", "alesia-siege", "britain-expedition"}
