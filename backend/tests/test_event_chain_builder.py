import pytest

from backend.app.rag.event_chain_builder import CampaignChainConfig, HistoricalEventChainBuilder
from backend.app.rag.event_registry import CorpusEvidenceCatalog, HistoricalEventRecord, HistoricalEventRegistry
from backend.app.rag.ingestion.models import TextChunk


def registry() -> HistoricalEventRegistry:
    chunks = [TextChunk("helvetii", "x", {"corpus_id": "caesar_gallic_war", "book": "1", "chapter": "12"}), TextChunk("bibracte", "x", {"corpus_id": "caesar_gallic_war", "book": "1", "chapter": "23"}), TextChunk("alesia", "x", {"corpus_id": "caesar_gallic_war", "book": "7", "chapter": "68"})]
    records = [HistoricalEventRecord(event_id=identifier, title=identifier.title(), corpus_id="caesar_gallic_war", period="58-51 BCE", description="Reviewed", involved_places=[identifier], evidence_refs=[identifier], source_book=book, source_chapter=chapter, confidence=0.8) for identifier, book, chapter in [("helvetii", "1", "12"), ("bibracte", "1", "23"), ("alesia", "7", "68")]]
    return HistoricalEventRegistry(records, CorpusEvidenceCatalog.from_chunks(chunks))


def config(events):
    return CampaignChainConfig(chain_id="caesar", title="Caesar campaign", corpus_id="caesar_gallic_war", period="58-51 BCE", description="Explicit order", events=events)


def test_builder_preserves_explicit_caesar_event_order_and_provenance():
    chain = HistoricalEventChainBuilder().build(registry(), config(["helvetii", "bibracte", "alesia"]))
    assert chain.event_ids == ["helvetii", "bibracte", "alesia"]
    assert [(step.order, step.source_book, step.source_chapter) for step in chain.steps] == [(1, "1", "12"), (2, "1", "23"), (3, "7", "68")]
    assert all(step.corpus_id == "caesar_gallic_war" and step.evidence_refs for step in chain.steps)


def test_builder_rejects_missing_event_id():
    with pytest.raises(KeyError, match="not registered"):
        HistoricalEventChainBuilder().build(registry(), config(["helvetii", "missing"]))


def test_builder_rejects_mixed_corpus_and_empty_chain():
    with pytest.raises(ValueError, match="belongs to corpus"):
        HistoricalEventChainBuilder().build(registry(), config(["helvetii"]).model_copy(update={"corpus_id": "polybius_histories"}))
    with pytest.raises(Exception):
        CampaignChainConfig(chain_id="empty", title="Empty", corpus_id="caesar_gallic_war", period="58 BCE", description="Empty", events=[])
