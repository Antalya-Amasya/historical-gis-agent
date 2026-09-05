"""G6AB: temporal ranges must not collapse into false exact single years."""

from __future__ import annotations

from backend.app.models import TemporalGroundingStatus, TemporalPrecision
from backend.app.routes.episode_relevance import _explicit_temporal_contradiction, _explicit_temporal_intervals
from backend.app.routes.temporal import EvidenceTemporalResolver


def _primary(text: str):
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve(text, "probe")
    return resolver.primary(readings, "probe"), codes


def test_a_ce_en_dash_range_is_not_exact_single_year():
    reading, codes = _primary("200–210 CE")
    assert reading.normalized_start != "210" or reading.normalized_end != "210"
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "210"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_b_bce_prefix_range_is_not_exact_single_year():
    reading, codes = _primary("BCE 210–200")
    assert reading.normalized_start != "-210" or reading.normalized_end != "-210"
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "-210"
    assert reading.normalized_end == "-200"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_c_ascii_hyphen_ce_range_matches_en_dash_semantics():
    en_dash, _ = _primary("200–210 CE")
    ascii_dash, _ = _primary("200-210 CE")
    assert ascii_dash.precision is TemporalPrecision.YEAR_RANGE
    assert (ascii_dash.normalized_start, ascii_dash.normalized_end) == (
        en_dash.normalized_start,
        en_dash.normalized_end,
    )


def test_d_single_bce_year_regression():
    for text in ("200 BCE", "BCE 200", "200 BC", "BC 200"):
        reading, codes = _primary(text)
        assert reading.normalized_start == "-200"
        assert reading.normalized_end == "-200"
        assert reading.precision is TemporalPrecision.YEAR
        assert codes == ["TEMPORAL_RESOLVED"]


def test_e_single_ce_year_regression():
    for text in ("200 CE", "AD 200"):
        reading, codes = _primary(text)
        assert reading.normalized_start == "200"
        assert reading.normalized_end == "200"
        assert reading.precision is TemporalPrecision.YEAR
        assert codes == ["TEMPORAL_RESOLVED"]


def test_f_unsupported_range_without_era_stays_unresolved():
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve("200–210", "probe")
    assert readings == []
    assert codes == ["TEMPORAL_UNRESOLVED"]


def test_consumer_inside_range_query_is_compatible():
    intervals = _explicit_temporal_intervals("Commander marched in 200–210 CE.")
    assert intervals == [(200, 210)]
    assert _explicit_temporal_contradiction(
        "Commander marched in 200–210 CE.",
        ("Trace the route in 205 CE.",),
    ) is False


def test_consumer_outside_range_query_is_incompatible():
    assert _explicit_temporal_contradiction(
        "Commander marched in 200–210 CE.",
        ("Trace the route in 300 CE.",),
    ) is True


def test_consumer_bce_range_overlap_uses_signed_interval():
    intervals = _explicit_temporal_intervals("The army moved during BCE 210–200.")
    assert intervals == [(-210, -200)]
    assert _explicit_temporal_contradiction(
        "The army moved during BCE 210–200.",
        ("Trace the route in 205 BCE.",),
    ) is False
    assert _explicit_temporal_contradiction(
        "The army moved during BCE 210–200.",
        ("Trace the route in 100 BCE.",),
    ) is True
