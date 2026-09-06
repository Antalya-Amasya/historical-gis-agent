"""G6AU: noun-phrase route queries carry explicit person identity."""

from __future__ import annotations

from backend.app.routes.evidence_relevance import (
    normalized_query_person_identities,
    person_identities_match,
)
from backend.tests.test_g6aq_directed_fragment_chain_authority import execute_route


QUERY = "The route of Marcus Valerius from Rome to Capua."


def test_a_wrong_person_noun_phrase_has_no_production_route():
    _state, result, _diagnostics = execute_route(
        "In 200 BCE, during Campaign Alpha, Lucius Valerius marched from Rome to Capua.",
        query=QUERY,
    )
    assert normalized_query_person_identities((QUERY,)) == [("marcus", "valerius")]
    assert result["result"]["route"] is None


def test_b_same_person_noun_phrase_remains_viable():
    _state, result, _diagnostics = execute_route(
        "Marcus Valerius marched from Rome to Capua.", query=QUERY,
    )
    assert result["result"]["route"] is not None


def test_c_noun_phrase_stops_before_time_scope():
    assert normalized_query_person_identities(("The route of Marcus Valerius in 200 BCE.",)) == [
        ("marcus", "valerius"),
    ]


def test_d_noun_phrase_stops_before_campaign_scope():
    assert normalized_query_person_identities(("The route of Marcus Valerius during Campaign Alpha.",)) == [
        ("marcus", "valerius"),
    ]


def test_e_single_token_noun_phrase_remains_supported():
    assert normalized_query_person_identities(("The route of Ariston from Rome to Capua.",)) == [("ariston",)]


def test_f_surname_only_evidence_is_not_a_full_name_match():
    assert not person_identities_match("Valerius marched from Rome to Capua.", (QUERY,))


def test_g_supported_query_forms_have_equal_identity():
    forms = (
        "Trace Marcus Valerius's route.",
        "Trace Marcus Valerius from Rome to Capua.",
        "Follow Marcus Valerius from Rome to Capua.",
        "Reconstruct Marcus Valerius in 200 BCE.",
        QUERY,
        "The route of Marcus Valerius during Campaign Alpha.",
    )
    assert [normalized_query_person_identities((form,)) for form in forms] == [[("marcus", "valerius")]] * len(forms)


def test_h_legacy_fallback_does_not_restore_wrong_person_route():
    _state, result, _diagnostics = execute_route(
        "Lucius Valerius marched from Rome to Capua, then sailed from Corcyra to Brundisium.",
        query=QUERY,
    )
    assert result["result"]["route"] is None
