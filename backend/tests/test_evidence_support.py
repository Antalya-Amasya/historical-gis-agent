from backend.app.agent.evidence_support import assess_evidence_support, extract_subject_terms, has_unsupported_route_pattern
from backend.app.models import Evidence


def evidence(identifier, text):
    return Evidence(id=identifier, author="Polybius", work="Histories", locator="Book III", excerpt=text, text=text)


def test_hannibal_route_evidence_is_sufficient():
    result = assess_evidence_support("展示汉尼拔翻越阿尔卑斯的路线", "historical_route", [evidence("one", "Hannibal crossed the Alps")])
    assert result.status == "sufficient" and set(result.matched_subject_terms) == {"hannibal", "alps"}


def test_caesar_gaul_request_with_hannibal_evidence_is_irrelevant():
    result = assess_evidence_support("展示凯撒征服高卢的路线", "historical_route", [evidence("one", "Hannibal crossed the Alps during the Punic War")])
    assert result.status == "irrelevant" and result.relevant_count == 0
    assert set(result.missing_subject_terms) == {"caesar", "gaul"}


def test_generic_words_do_not_make_route_support_sufficient():
    result = assess_evidence_support("show Caesar route in Gaul", "historical_route", [evidence("one", "The army marched on a route to Italy")])
    assert result.status == "irrelevant"


def test_bilingual_aliases_normalize_to_maintainable_subject_terms():
    assert set(extract_subject_terms("凯撒在高卢")) == {"caesar", "gaul"}
    assert set(extract_subject_terms("Hannibal crossed the 阿尔卑斯")) == {"hannibal", "alps"}


def test_partial_and_aggregated_subject_coverage():
    partial = assess_evidence_support("Caesar in Gaul route", "historical_route", [evidence("one", "Caesar led an army")])
    assert partial.status == "insufficient" and partial.matched_subject_terms == ("caesar",)
    full = assess_evidence_support("Caesar in Gaul route", "historical_route", [evidence("one", "Caesar led an army"), evidence("two", "War in Gaul")])
    assert full.status == "sufficient" and set(full.matched_subject_terms) == {"caesar", "gaul"}


def test_route_pattern_validator_is_generic():
    assert has_unsupported_route_pattern("The route goes A → B → C")
    assert has_unsupported_route_pattern("路线经过甲、乙、丙")
    assert not has_unsupported_route_pattern("The current evidence is insufficient to support a route.")
