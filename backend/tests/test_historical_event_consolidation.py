from backend.app.models import Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator


def evidence(identifier: str, text: str, period: str | None = None) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text, period=period)


def consolidate(*items: Evidence):
    candidates, _ = EvidenceGroundedHistoricalEventExtractor().extract(list(items))
    return HistoricalEventConsolidator().consolidate(candidates)


def test_same_battle_at_same_place_merges_and_retains_all_evidence():
    events, diagnostics = consolidate(
        evidence("a", "Army A fought the battle at Place X."),
        evidence("b", "Later in the account, the battle at Place X ended in victory."),
    )
    assert len(events) == 1 and events[0].event_type is HistoricalEventType.BATTLE
    assert events[0].evidence_refs == ["a", "b"] and len(events[0].source_statements) == 2
    assert diagnostics["merged_candidate_count"] == 1


def test_different_battles_same_place_with_different_raw_time_remain_separate():
    events, diagnostics = consolidate(
        evidence("a", "Army A fought the battle at Place X in 133 BCE."),
        evidence("b", "Army B fought the battle at Place X in 132 BCE."),
    )
    assert len(events) == 2 and diagnostics["merged_candidate_count"] == 0
    assert "TEMPORAL_CONFLICT" in diagnostics["reason_codes"]


def test_same_type_at_different_places_and_type_conflicts_never_merge():
    events, diagnostics = consolidate(
        evidence("x", "Army A fought the battle at Place X."),
        evidence("y", "Army B fought the battle at Place Y."),
        evidence("r", "A reform was proposed at Place X."),
    )
    assert len(events) == 3 and diagnostics["merged_candidate_count"] == 0
    assert "EVENT_TYPE_CONFLICT" in diagnostics["reason_codes"]


def test_ambiguous_place_free_candidates_remain_separate_and_keep_unknown_time():
    events, diagnostics = consolidate(
        evidence("a", "The senate issued a decree."),
        evidence("b", "The senate issued a decree."),
    )
    assert len(events) == 2 and diagnostics["ambiguous_candidate_count"] == 2
    assert all(item.temporal_grounding.raw_expression is None for item in events)


def test_movement_consolidation_never_constructs_route_data():
    events, _ = consolidate(
        evidence("a", "The army marched from Place X to Place Y."),
        evidence("b", "The army marched from Place X to Place Y."),
    )
    assert len(events) == 1 and events[0].event_type is HistoricalEventType.MOVEMENT
    assert events[0].places == [] and any("route inference" in item for item in events[0].limitations)
