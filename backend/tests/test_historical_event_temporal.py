from backend.app.models import Evidence, TemporalGroundingStatus, TemporalPrecision
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.temporal import EvidenceTemporalResolver


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def resolve(text: str):
    return EvidenceTemporalResolver().resolve(text, "e")


def test_explicit_bce_spellings_and_chinese_normalize_to_one_signed_historical_year():
    readings = [resolve(value)[0][0] for value in ("133 BCE", "133 BC", "B.C. 133", "130 B.C.", "31 B.C.", "公元前133年")]
    assert {(item.normalized_start, item.normalized_end) for item in readings} == {("-133", "-133"), ("-130", "-130"), ("-31", "-31")}
    assert all(item.precision is TemporalPrecision.YEAR for item in readings)


def test_explicit_ce_spellings_normalize_without_year_zero():
    readings = [resolve(value)[0][0] for value in ("14 CE", "14 AD", "A.D. 14", "公元14年")]
    assert {(item.normalized_start, item.normalized_end) for item in readings} == {("14", "14")}
    assert resolve("0 BCE")[0] == [] and resolve("0 CE")[0] == []


def test_bce_ranges_are_chronologically_ordered_in_signed_historical_years():
    for text in ("133–121 BCE", "133 BC to 121 BC", "公元前133年至公元前121年"):
        reading = resolve(text)[0][0]
        assert (reading.normalized_start, reading.normalized_end, reading.precision) == ("-133", "-121", TemporalPrecision.YEAR_RANGE)


def test_approximate_year_keeps_its_precision_and_explicit_century_is_safe_unresolved():
    reading = resolve("circa 133 BC")[0][0]
    assert (reading.normalized_start, reading.precision) == ("-133", TemporalPrecision.APPROXIMATE)
    century, codes = resolve("公元前2世纪")
    assert century[0].normalized_start is None and codes == ["TEMPORAL_CENTURY_UNRESOLVED"]


def test_relative_dates_and_numeric_false_positives_are_never_normalized():
    for text in ("during his tribunate", "the following year", "500 iugera", "Book XXVIII", "chapter 12"):
        readings, codes = resolve(text)
        assert readings == [] and codes == ["TEMPORAL_UNRESOLVED"]


def test_conflicting_dates_are_not_silently_selected_and_all_readings_keep_provenance():
    readings, codes = resolve("The battle at Place X occurred in 133 BCE and 132 BC.")
    primary = EvidenceTemporalResolver().primary(readings, "e")
    assert codes == ["TEMPORAL_CONFLICT"] and primary.status is TemporalGroundingStatus.CONFLICT
    assert {item.normalized_start for item in readings} == {"-133", "-132"}
    assert all(item.evidence_refs == ["e"] for item in readings)


def test_equivalent_expressions_merge_using_normalized_identity_and_preserve_both_provenance():
    extractor = EvidenceGroundedHistoricalEventExtractor()
    candidates, _ = extractor.extract([
        evidence("a", "The army fought the battle at Place X in 133 BCE."),
        evidence("b", "The battle at Place X ended in 133 BC."),
    ])
    events, diagnostics = HistoricalEventConsolidator().consolidate(candidates)
    assert len(events) == 1 and diagnostics["merged_candidate_count"] == 1
    assert events[0].evidence_refs == ["a", "b"]
    assert {item.normalized_start for item in events[0].temporal_groundings} == {"-133"}


def test_event_extraction_resolves_statement_time_not_document_metadata_and_never_creates_route_data():
    events, diagnostics = EvidenceGroundedHistoricalEventExtractor().extract([
        evidence("movement", "The army marched from Place X to Place Y in 133 BCE."),
    ])
    assert events[0].temporal_grounding.normalized_start == "-133"
    assert events[0].temporal_grounding.evidence_refs == ["movement"]
    assert events[0].places == [] and "route inference" in events[0].limitations[0]
    assert diagnostics["reason_codes"] == ["EVENT_EXTRACTED"]
