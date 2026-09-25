"""R4-C3: possessive route-subject parsing without location misclassification."""
from __future__ import annotations

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import decompose_movement_query, primary_route_subject


def test_synthetic_possessive_journey_from_to():
    roles = analyze_query("Marcus's journey from Alpha to Beta")
    assert "marcus" in roles.person_terms
    assert "s" not in roles.person_terms
    assert "s" not in roles.context_terms
    assert {"alpha", "beta"} <= roles.location_terms
    assert "marcus" not in roles.location_terms
    assert "journey" in roles.action_terms or "journey" in roles.context_terms
    assert primary_route_subject("Marcus's journey from Alpha to Beta", roles) == "Marcus"


def test_multiword_possessive_march():
    query = "Julius Caesar's march from Rome to Capua"
    roles = analyze_query(query)
    assert {"julius", "caesar"} <= roles.person_terms
    assert "s" not in roles.person_terms
    assert {"rome", "capua"} <= roles.location_terms
    assert "caesar" not in roles.location_terms
    assert primary_route_subject(query, roles) == "Julius Caesar"


def test_multiword_possessive_campaign_into():
    query = "Alexander the Great's campaign into Asia"
    roles = analyze_query(query)
    assert "alexander" in roles.person_terms
    assert "s" not in roles.person_terms
    assert "asia" in roles.location_terms
    assert "alexander" not in roles.location_terms
    assert primary_route_subject(query, roles) == "Alexander the Great"


def test_non_possessive_route_unchanged():
    roles = analyze_query("Marcus marched from Alpha to Beta")
    assert "marcus" in roles.person_terms
    assert {"alpha", "beta"} <= roles.location_terms
    assert "marched" in roles.action_terms
    assert "s" not in roles.person_terms


def test_non_person_possessive_does_not_emit_standalone_s():
    roles = analyze_query("the army's march to Syria")
    assert "s" not in roles.person_terms
    assert "s" not in roles.context_terms
    assert "syria" in roles.location_terms


def test_preposition_to_does_not_make_possessive_subject_a_location():
    query = "What route is attributed to Commander's principal expedition?"
    roles = analyze_query(query)
    assert "commander" in roles.person_terms
    assert "commander" not in roles.location_terms
    assert "s" not in roles.person_terms


def _intent_text(query: str) -> str:
    return " ".join(intent.query for intent in decompose_movement_query(query))


def test_attributed_to_possessive_subject_is_owner_only():
    query = "What route is attributed to Marcus's expedition?"
    assert primary_route_subject(query) == "Marcus"
    blob = _intent_text(query)
    assert "attributed to Marcus" not in blob.casefold()
    assert " s " not in f" {blob} "
    assert not blob.casefold().split() or "s" not in {token.casefold() for token in blob.split()}


def test_associated_with_multiword_possessive_subject():
    query = "Trace the journey associated with Julius Caesar's campaign."
    assert primary_route_subject(query) == "Julius Caesar"
    blob = _intent_text(query)
    assert "associated with" not in blob.casefold()
    assert "s" not in {token.casefold() for token in blob.split()}


def test_map_multiword_the_great_possessive_subject():
    query = "Map the route of Alexander the Great's expedition into Asia."
    assert primary_route_subject(query) == "Alexander the Great"
    blob = _intent_text(query)
    assert "s" not in {token.casefold() for token in blob.split()}


def test_ordinary_possessive_controls_remain():
    assert primary_route_subject("Trace Marcus's route.") == "Marcus"
    assert primary_route_subject("Julius Caesar's march from Rome to Capua") == "Julius Caesar"
    assert primary_route_subject("Alexander the Great's campaign into Asia") == "Alexander the Great"


def test_non_possessive_controls_do_not_invent_primary_subject():
    assert primary_route_subject("Trace Marcus during the campaign.") is None
    roles = analyze_query("Caesar marched from Rome to Capua.")
    assert "caesar" in roles.person_terms
    assert {"rome", "capua"} <= roles.location_terms
