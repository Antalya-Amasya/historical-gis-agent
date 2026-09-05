"""G6AJ: between-range evidence must not collapse to false exact single years."""

from __future__ import annotations

from backend.app.models import TemporalGroundingStatus, TemporalPrecision
from backend.app.routes.episode_relevance import _explicit_temporal_contradiction, _explicit_temporal_intervals
from backend.app.routes.temporal import EvidenceTemporalResolver
from backend.tests.test_g6ag_query_subject_scope_normalization import classify_route

S4_03_TEXT = "Between 200 and 210 CE Ariston marched from Rome to Capua."


def _resolve(text: str):
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve(text, "probe")
    return resolver.primary(readings, "probe"), readings, codes


def test_a_exact_gpt6_between_ce_range_is_not_exact_year():
    reading, readings, codes = _resolve(S4_03_TEXT)
    assert reading.precision is not TemporalPrecision.YEAR
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "210"
    assert reading.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    assert codes == ["TEMPORAL_RESOLVED"]
    assert readings


def test_b_bce_between_range_is_not_exact_endpoint():
    reading, readings, codes = _resolve("Between 210 and 200 BCE Ariston marched.")
    assert reading.precision is not TemporalPrecision.YEAR
    if reading.precision is TemporalPrecision.YEAR_RANGE:
        assert reading.normalized_start == "-210"
        assert reading.normalized_end == "-200"
        assert codes == ["TEMPORAL_RESOLVED"]
    else:
        assert readings == []
        assert codes == ["TEMPORAL_UNRESOLVED"]


def test_c_exact_year_ordinary_sentence_remains_year():
    reading, _, codes = _resolve("In 210 CE Ariston marched.")
    assert reading.precision is TemporalPrecision.YEAR
    assert reading.normalized_start == "210"
    assert reading.normalized_end == "210"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_d_ordinary_conjunction_does_not_fabricate_range():
    text = "In 200 CE Ariston marched and in 210 CE Bion sailed."
    reading, readings, codes = _resolve(text)
    assert reading.precision is not TemporalPrecision.YEAR_RANGE
    assert codes == ["TEMPORAL_CONFLICT"]


def test_e_geographic_between_preserves_exact_year():
    reading, _, codes = _resolve("Ariston marched between Rome and Capua in 200 CE.")
    assert reading.precision is TemporalPrecision.YEAR
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "200"
    assert codes == ["TEMPORAL_RESOLVED"]


def test_f_consumer_precision_safety_no_false_exact_year_authority():
    reading, readings, codes = _resolve(S4_03_TEXT)
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert reading.normalized_start == "200"
    assert reading.normalized_end == "210"
    assert codes == ["TEMPORAL_RESOLVED"]

    intervals = _explicit_temporal_intervals(S4_03_TEXT)
    assert intervals == [(200, 210)]

    assert _explicit_temporal_contradiction(S4_03_TEXT, ("Trace Ariston's route in 205 CE.",)) is False
    assert _explicit_temporal_contradiction(S4_03_TEXT, ("Trace Ariston's route in 210 CE.",)) is False
    assert _explicit_temporal_contradiction(S4_03_TEXT, ("Trace Ariston's route in 300 CE.",)) is True

    _, episode, detail, admitted, route_points = classify_route(
        "Trace Ariston's route in 210 CE.",
        S4_03_TEXT,
    )
    assert reading.precision is TemporalPrecision.YEAR_RANGE
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2
    assert episode.value == "DIRECT_QUERY_EPISODE"
