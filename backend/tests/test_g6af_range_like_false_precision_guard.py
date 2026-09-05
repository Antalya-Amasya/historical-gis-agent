"""G6AF: range-like evidence must not collapse to false exact single years."""

from __future__ import annotations

from backend.app.models import TemporalGroundingStatus, TemporalPrecision
from backend.app.routes.episode_relevance import _explicit_temporal_contradiction, _explicit_temporal_intervals
from backend.app.routes.temporal import EvidenceTemporalResolver


def _resolve(text: str):
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve(text, "probe")
    return resolver.primary(readings, "probe"), readings, codes


def _assert_not_exact_year(reading, *, text: str) -> None:
    assert reading.precision is not TemporalPrecision.YEAR, text
    if reading.normalized_start is not None and reading.normalized_end is not None:
        assert reading.normalized_start != reading.normalized_end or reading.precision is TemporalPrecision.YEAR, text


def test_a_to_ce_range_must_not_steal_endpoint_year():
    reading, readings, codes = _resolve("200 to 210 CE")
    _assert_not_exact_year(reading, text="200 to 210 CE")
    assert reading.precision in {TemporalPrecision.YEAR_RANGE, TemporalPrecision.UNKNOWN} or not readings
    if reading.precision is TemporalPrecision.YEAR_RANGE:
        assert reading.normalized_start == "200"
        assert reading.normalized_end == "210"
    else:
        assert codes == ["TEMPORAL_UNRESOLVED"]


def test_b_bce_to_range_must_not_steal_endpoint_year():
    reading, readings, codes = _resolve("BCE 210 to 200")
    _assert_not_exact_year(reading, text="BCE 210 to 200")
    assert reading.precision in {TemporalPrecision.YEAR_RANGE, TemporalPrecision.UNKNOWN} or not readings
    if reading.precision is TemporalPrecision.YEAR_RANGE:
        assert reading.normalized_start == "-210"
        assert reading.normalized_end == "-200"
    else:
        assert codes == ["TEMPORAL_UNRESOLVED"]


def test_c_slash_form_stays_conservative():
    reading, readings, codes = _resolve("200/210 CE")
    _assert_not_exact_year(reading, text="200/210 CE")
    assert codes == ["TEMPORAL_UNRESOLVED"]
    assert readings == []


def test_d_double_hyphen_must_not_steal_endpoint_year():
    reading, readings, codes = _resolve("200--210 CE")
    _assert_not_exact_year(reading, text="200--210 CE")
    assert reading.precision in {TemporalPrecision.YEAR_RANGE, TemporalPrecision.UNKNOWN} or not readings
    if reading.precision is TemporalPrecision.YEAR_RANGE:
        assert (reading.normalized_start, reading.normalized_end) == ("200", "210")
    else:
        assert codes == ["TEMPORAL_UNRESOLVED"]


def test_e_supported_en_dash_range_remains_year_range():
    reading, _, codes = _resolve("200–210 CE")
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "210"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_f_supported_bce_en_dash_range_remains_year_range():
    reading, _, codes = _resolve("BCE 210–200")
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "-210"
    assert reading.normalized_end == "-200"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_g_exact_year_regression():
    cases = {
        "200 BCE": "-200",
        "BCE 200": "-200",
        "200 BC": "-200",
        "200 CE": "200",
        "AD 200": "200",
    }
    for text, expected in cases.items():
        reading, _, codes = _resolve(text)
        assert reading.precision is TemporalPrecision.YEAR, text
        assert reading.normalized_start == expected, text
        assert reading.normalized_end == expected, text
        assert codes == ["TEMPORAL_RESOLVED"], text


def test_h_ordinary_punctuation_preserves_exact_year():
    reading, _, codes = _resolve("In 200 CE, Ariston marched.")
    assert reading.precision is TemporalPrecision.YEAR
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "200"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_i_unseen_until_separator_is_not_false_exact_year():
    reading, readings, codes = _resolve("200 until 210 CE")
    _assert_not_exact_year(reading, text="200 until 210 CE")
    assert codes == ["TEMPORAL_UNRESOLVED"] or reading.precision is TemporalPrecision.YEAR_RANGE
    if reading.precision is not TemporalPrecision.YEAR_RANGE:
        assert readings == []


def test_j_consumer_slash_range_does_not_gain_exact_compatibility():
    reading, readings, codes = _resolve("200/210 CE")
    assert reading.precision is not TemporalPrecision.YEAR
    assert readings == []
    assert codes == ["TEMPORAL_UNRESOLVED"]
    statement = "Commander marched in 200/210 CE."
    assert _explicit_temporal_intervals(statement) == []
    assert _explicit_temporal_contradiction(
        statement,
        ("Trace the route in 210 CE.",),
    ) is False
    assert reading.status is not TemporalGroundingStatus.EVIDENCE_GROUNDED or reading.normalized_start is None
