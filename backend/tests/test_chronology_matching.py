from backend.app.chronology.matching import ChronologyMatchStatus, ChronologyMatcher, ChronologyRecord
from backend.app.models import (
    HistoricalEvent, HistoricalEventPlaceMention, HistoricalEventTemporalGrounding,
    HistoricalEventType, TemporalGroundingStatus, TemporalPrecision,
)


def record(identifier: str, text: str, year: str, source: str = "modern-a") -> ChronologyRecord:
    return ChronologyRecord(record_id=identifier, canonical_event_label="Historical event", raw_event_text=text,
                            normalized_start=year, normalized_end=year, precision="YEAR", raw_temporal_expression=f"{-int(year)} B.C.",
                            source_id=source, source_locator={"epub_document": "chapter.xhtml", "paragraph_index": 1}, source_excerpt=text)


def event(text: str, *, year: str | None = None, places: list[str] | None = None) -> HistoricalEvent:
    grounding = HistoricalEventTemporalGrounding(
        raw_expression=f"{-int(year)} B.C." if year else None, normalized_start=year, normalized_end=year,
        precision=TemporalPrecision.YEAR if year else TemporalPrecision.UNKNOWN,
        status=TemporalGroundingStatus.EVIDENCE_GROUNDED if year else TemporalGroundingStatus.UNRESOLVED,
        evidence_refs=["primary-1"],
    )
    return HistoricalEvent(id="event-1", name="event", summary=text, event_type=HistoricalEventType.BATTLE,
                           evidence_refs=["primary-1"], temporal_grounding=grounding,
                           place_mentions=[HistoricalEventPlaceMention(raw_text=value, canonical_hint=value) for value in places or []])


def test_strong_same_event_match_preserves_separate_provenance_and_does_not_mutate_event():
    original = event("Roman forces defeated Hannibal at Cannae after encirclement.", places=["Cannae"])
    matcher = ChronologyMatcher([record("r1", "Roman forces defeated Hannibal at Cannae after encirclement.", "-216")])
    before = original.temporal_grounding.model_dump()

    result = matcher.match(original)

    assert result[0].match_status is ChronologyMatchStatus.MATCH
    assert result[0].chronology_source_id == "modern-a"
    assert result[0].event_evidence_refs == ["primary-1"]
    assert original.temporal_grounding.model_dump() == before


def test_generic_entity_or_place_overlap_cannot_become_match():
    matcher = ChronologyMatcher([
        record("r1", "Caesar appointed provincial governors in Rome.", "-46"),
        record("r2", "The senate debated tax collection at Rome.", "-55"),
    ])

    caesar = matcher.match(event("Caesar issued a decree concerning grain.", places=["Rome"]))
    rome = matcher.match(event("Roman delegates met at Rome over civic matters.", places=["Rome"]))

    assert all(item.match_status is not ChronologyMatchStatus.MATCH for item in caesar + rome)


def test_same_person_different_action_is_not_a_match_and_distinctive_action_plus_place_is():
    matcher = ChronologyMatcher([
        record("r1", "Caesar crossed the Rubicon and entered Italy.", "-49"),
        record("r2", "Caesar reformed the calendar in Rome.", "-46"),
    ])

    crossed = matcher.match(event("Caesar crossed the Rubicon and entered Italy.", places=["Rubicon", "Italy"]))
    reform = matcher.match(event("Caesar issued a grain decree in Rome.", places=["Rome"]))

    assert crossed[0].match_status is ChronologyMatchStatus.MATCH
    assert all(item.match_status is not ChronologyMatchStatus.MATCH for item in reform)


def test_multiple_sources_agree_without_dropping_provenance_and_disagreement_is_conflict():
    same = ChronologyMatcher([
        record("a", "Tiberius Gracchus proposed an agrarian law at Rome.", "-133", "greenidge"),
        record("b", "Tiberius Gracchus proposed an agrarian law at Rome.", "-133", "historians-history"),
    ]).match(event("Tiberius Gracchus proposed an agrarian law at Rome.", places=["Rome"]))
    conflict = ChronologyMatcher([
        record("a", "Tiberius Gracchus proposed an agrarian law at Rome.", "-133", "greenidge"),
        record("b", "Tiberius Gracchus proposed an agrarian law at Rome.", "-132", "historians-history"),
    ]).match(event("Tiberius Gracchus proposed an agrarian law at Rome.", places=["Rome"]))

    assert {item.chronology_source_id for item in same} == {"greenidge", "historians-history"}
    assert all(item.match_status is ChronologyMatchStatus.MATCH for item in same)
    assert all(item.match_status is ChronologyMatchStatus.CONFLICT for item in conflict)


def test_primary_explicit_date_agreement_and_conflict_are_preserved():
    agreed = ChronologyMatcher([record("a", "Roman forces defeated Hannibal at Cannae after encirclement.", "-216")]).match(
        event("Roman forces defeated Hannibal at Cannae after encirclement.", year="-216", places=["Cannae"]))
    conflict = ChronologyMatcher([record("a", "Roman forces defeated Hannibal at Cannae after encirclement.", "-215")]).match(
        event("Roman forces defeated Hannibal at Cannae after encirclement.", year="-216", places=["Cannae"]))

    assert "primary_temporal_agreement" in agreed[0].matched_signals
    assert conflict[0].match_status is ChronologyMatchStatus.CONFLICT
    assert "primary_temporal_conflict" in conflict[0].conflicts


def test_generic_military_vocabulary_and_multiple_weak_passages_fail_safely():
    matcher = ChronologyMatcher([
        record("a", "The army campaigned in Italy during a difficult war.", "-210"),
        record("b", "The army campaigned in Sicily during a difficult war.", "-209"),
    ])

    result = matcher.match(event("The army campaigned during a difficult war."))

    assert result
    assert all(item.match_status is not ChronologyMatchStatus.MATCH for item in result)
    assert all(item.match_status in {ChronologyMatchStatus.NO_MATCH, ChronologyMatchStatus.AMBIGUOUS} for item in result)
