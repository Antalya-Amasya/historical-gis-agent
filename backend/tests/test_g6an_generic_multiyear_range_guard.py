"""G6AN: generic multi-year range-like spans must not collapse to false exact years."""

from __future__ import annotations

from backend.app.models import TemporalGroundingStatus, TemporalPrecision
from backend.app.routes.episode_relevance import _explicit_temporal_contradiction, _explicit_temporal_intervals
from backend.app.routes.temporal import EvidenceTemporalResolver
from backend.tests.test_g6ag_query_subject_scope_normalization import classify_route

S5_03_TEXT = "Ariston marched from Rome to Capua from 200 through 210 CE."
BCE_THROUGH = "from 210 through 200 BCE Ariston marched."
EXACT_MOVEMENT = "Ariston marched from Rome to Capua in 210 CE."
INDEPENDENT_DATES = "In 200 BCE Ariston was consul; in 100 BCE he marched."
BETWEEN_RANGE = "Between 200 and 210 CE Ariston marched from Rome to Capua."
SLASH_FORM = "200/210 CE"
UNSEEN_THRU = "200 thru 210 CE"


def _resolve(text: str, *, query: str | None = None):
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve(text, "probe")
    return resolver.primary(readings, "probe"), readings, codes


def _assert_not_exact_year(reading, *, text: str) -> None:
    assert reading.precision is not TemporalPrecision.YEAR, text
    if reading.normalized_start is not None and reading.normalized_end is not None:
        assert (
            reading.normalized_start != reading.normalized_end
            or reading.precision is not TemporalPrecision.YEAR
        ), text


def test_a_exact_gpt6_s5_03_ce_through_range_not_exact_year():
    reading, readings, codes = _resolve(S5_03_TEXT)
    assert reading.precision is not TemporalPrecision.YEAR
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "210"
    assert reading.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    assert codes == ["TEMPORAL_RESOLVED"]
    assert readings


def test_b_bce_through_range_not_exact_endpoint():
    reading, readings, codes = _resolve(BCE_THROUGH)
    _assert_not_exact_year(reading, text=BCE_THROUGH)
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "-210"
    assert reading.normalized_end == "-200"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_c_ordinary_exact_year_movement_remains_year():
    reading, _, codes = _resolve(EXACT_MOVEMENT)
    assert reading.precision is TemporalPrecision.YEAR
    assert reading.normalized_start == "210"
    assert reading.normalized_end == "210"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_d_independent_dated_propositions_do_not_fabricate_range():
    reading, _, codes = _resolve(INDEPENDENT_DATES)
    assert reading.precision is not TemporalPrecision.YEAR_RANGE
    assert codes == ["TEMPORAL_CONFLICT"]


def test_e_between_range_preserved():
    reading, _, codes = _resolve(BETWEEN_RANGE)
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "210"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_f_slash_form_stays_unresolved():
    reading, readings, codes = _resolve(SLASH_FORM)
    _assert_not_exact_year(reading, text=SLASH_FORM)
    assert readings == []
    assert codes == ["TEMPORAL_UNRESOLVED"]


def test_g_unseen_thru_form_not_exact_endpoint():
    reading, readings, codes = _resolve(UNSEEN_THRU)
    _assert_not_exact_year(reading, text=UNSEEN_THRU)
    assert reading.precision in {TemporalPrecision.YEAR_RANGE, TemporalPrecision.UNKNOWN} or not readings
    if reading.precision is TemporalPrecision.YEAR_RANGE:
        assert (reading.normalized_start, reading.normalized_end) == ("200", "210")
    else:
        assert readings == []
        assert codes == ["TEMPORAL_UNRESOLVED"]


def test_h_consumer_safety_preserves_range_not_exact_authority():
    reading, readings, codes = _resolve(S5_03_TEXT)
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "210"
    assert codes == ["TEMPORAL_RESOLVED"]

    assert _explicit_temporal_intervals(S5_03_TEXT) == [(200, 210)]
    assert _explicit_temporal_contradiction(S5_03_TEXT, ("Trace Ariston's route in 205 CE.",)) is False
    assert _explicit_temporal_contradiction(S5_03_TEXT, ("Trace Ariston's route in 210 CE.",)) is False
    assert _explicit_temporal_contradiction(S5_03_TEXT, ("Trace Ariston's route in 300 CE.",)) is True

    for query in (
        "Trace Ariston's route in 205 CE.",
        "Trace Ariston's route in 210 CE.",
        "Trace Ariston's route in 300 CE.",
    ):
        resolved, _, _ = _resolve(S5_03_TEXT)
        assert resolved.precision is TemporalPrecision.YEAR_RANGE
        assert resolved.normalized_start == "200"
        assert resolved.normalized_end == "210"

    _, _, detail_210, admitted_210, route_points_210 = classify_route(
        "Trace Ariston's route in 210 CE.",
        S5_03_TEXT,
    )
    assert detail_210["admitted"] is True
    assert admitted_210 is True
    assert route_points_210 >= 2

    _, _, detail_300, admitted_300, route_points_300 = classify_route(
        "Trace Ariston's route in 300 CE.",
        S5_03_TEXT,
    )
    assert detail_300["admitted"] is False
    assert admitted_300 is False
    assert route_points_300 < 2
