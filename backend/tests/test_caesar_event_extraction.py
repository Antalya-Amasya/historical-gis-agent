import pytest

from backend.app.rag.event_extraction import EventExtractionError, HistoricalEventExtractor, RetrievedEvidenceChunk


def evidence(event_id: str, book: str, chapter: str, description: str, places: list[str], sequence: int) -> RetrievedEvidenceChunk:
    return RetrievedEvidenceChunk(corpus_id="caesar_gallic_war", book=book, chapter=chapter, text="Explicit retrieved source text.", evidence_refs=[f"caesar-{book}-{chapter}"], event_id=event_id, period="58-51 BCE", description=description, involved_places=places, sequence=sequence)


@pytest.mark.parametrize(("chunk", "expected_book", "expected_place"), [
    (evidence("helvetii-migration", "1", "12", "Explicit Helvetii migration event.", ["Helvetii"], 1), "1", "Helvetii"),
    (evidence("alesia-campaign", "7", "68", "Explicit Alesia campaign event.", ["Alesia"], 1), "7", "Alesia"),
    (evidence("britain-expedition", "4", "20", "Explicit Britain expedition event.", ["Britain"], 1), "4", "Britain"),
])
def test_caesar_event_extraction_preserves_explicit_evidence_and_provenance(chunk, expected_book, expected_place):
    chain = HistoricalEventExtractor().extract([chunk])

    assert chain.event_id == chunk.event_id and chain.description == chunk.description
    assert chain.involved_places == [expected_place]
    assert chain.evidence_refs == chunk.evidence_refs
    assert chain.steps[0].provenance.model_dump() == {"source_corpus": "caesar_gallic_war", "book": expected_book, "chapter": chunk.chapter, "evidence_refs": chunk.evidence_refs}


def test_britain_book_iv_and_v_steps_keep_caller_supplied_sequence_without_route_inference():
    first = evidence("britain-expedition", "4", "20", "Explicit Britain expedition event.", ["Britain"], 1)
    second = evidence("britain-expedition", "5", "8", "Explicit Britain expedition event.", ["Britain"], 2)
    chain = HistoricalEventExtractor().extract([second, first])

    assert [step.provenance.book for step in chain.steps] == ["4", "5"]
    assert [step.sequence for step in chain.steps] == [1, 2]
    assert chain.involved_places == ["Britain"]


def test_extractor_rejects_inconsistent_explicit_event_inputs():
    first = evidence("helvetii-migration", "1", "12", "Explicit Helvetii migration event.", ["Helvetii"], 1)
    different_event = evidence("alesia-campaign", "7", "68", "Explicit Alesia campaign event.", ["Alesia"], 2)

    with pytest.raises(EventExtractionError, match="same explicit event_id"):
        HistoricalEventExtractor().extract([first, different_event])
