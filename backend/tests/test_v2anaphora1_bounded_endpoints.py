"""Adjacent completed destinations may bind location anaphora, never actors."""
import pytest

from backend.app.models import EventActorStatus, EventPlaceRole
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.tests.test_g4a_movement_semantics import evidence
from backend.tests.test_v2auth1_strict_endpoint_admission import build, labels

LOCAL = HistoricalPlaceMentionExtractor(tuple(HistoricalPlaceAlias(n, (n.lower(),))
    for n in ("A", "B", "C", "D", "Neralon", "Darsena", "Galveth", "Veloria")))


def extract(text, extractor=LOCAL):
    item = evidence("local-source", text)
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor).extract([item])
    claims = extractor.movement_claims([item], event_id="local")
    return events, claims


def pairs(claims):
    return [(c.source_place, c.destination_place) for c in claims]


def origins(event):
    return [p.canonical_hint or p.raw_text for p in event.place_mentions if p.role is EventPlaceRole.ORIGIN]


@pytest.mark.parametrize("prefix", ["From there", "Thence", "From that place"])
@pytest.mark.parametrize("mover", ["he", "Marcus"])
def test_adjacent_destination_is_origin_and_actor_remains_independent(prefix, mover):
    first = "Marcus marched from A to B."
    second = f"{prefix} {mover} proceeded to C."
    events, claims = extract(first + " " + second)
    assert pairs(claims) == [("A", "B"), ("B", "C")]
    assert origins(events[-1]) == ["B"]
    assert events[-1].actor.actor_status is (EventActorStatus.UNKNOWN if mover == "he" else EventActorStatus.EXPLICIT)
    assert events[-1].source_statements == [second, first]
    assert claims[-1].textual_basis == first + " " + second
    assert claims[-1].supporting_evidence_ids == ["local-source"]
    assert len({c.id for c in claims}) == len(claims)


def test_three_legs_resume_each_completed_destination():
    text = "Damon Philostratos marched from Neralon to Darsena. From there he proceeded to Galveth. Thence he sailed to Veloria."
    events, claims = extract(text)
    assert pairs(claims) == [("Neralon", "Darsena"), ("Darsena", "Galveth"), ("Galveth", "Veloria")]
    assert [origins(e) for e in events] == [["Neralon"], ["Darsena"], ["Galveth"]]
    assert all(e.actor.actor_status is EventActorStatus.UNKNOWN for e in events[1:])


@pytest.mark.parametrize("first", [
    "Marcus advanced from A toward B but stopped before reaching it.",
    "Marcus was prevented from entering B.",
    "Marcus planned to march to B.",
    "Marcus intended to go to B.",
    "Marcus did not reach B.",
    "Marcus could march to B.",
    "Marcus attempted to enter B.",
])
def test_non_completed_destination_is_not_an_antecedent(first):
    events, claims = extract(first + " From there he marched to C.")
    assert not any(c.source_place == "B" for c in claims)
    assert all(not origins(e) for e in events if e.summary.startswith("From there"))


@pytest.mark.parametrize("first", [
    "Marcus discussed the campaign at A.",
    "Marcus met Pompey between A and B.",
    "Marcus marched from A to B and C.",
    "Marcus marched from A to B, while Pompey marched from C to D.",
])
def test_no_unique_completed_endpoint_means_no_inheritance(first):
    events, claims = extract(first + " From there he marched to D.")
    assert all(not origins(e) for e in events if e.summary.startswith("From there"))
    assert not any(c.movement_relation == "discourse_continuation" for c in claims)


@pytest.mark.parametrize("interruption", [
    "Pompey then arrived.", "The army waited for several days.",
    "Years later Marcus began a new campaign at D.",
    "The army waited for several days. Pompey arrived from D.",
])
def test_intervening_sentence_breaks_continuity(interruption):
    events, claims = extract("Marcus marched from A to B. " + interruption + " From there he marched to C.")
    assert all(not origins(e) for e in events if e.summary.startswith("From there"))
    assert not any(c.movement_relation == "discourse_continuation" for c in claims)


def test_explicit_different_actor_cannot_use_predecessor_endpoint():
    events, claims = extract("Marcus marched from A to B. From there Pompey proceeded to C.")
    assert pairs(claims) == [("A", "B")]
    assert not origins(events[-1])
    assert events[-1].actor.actor_text == "Pompey"


def test_explicit_new_actor_establishes_own_endpoint():
    x = HistoricalPlaceMentionExtractor()
    events, claims = extract("Marcus marched to Capua. Pompey marched to Beneventum. From there he moved to Brundisium.", x)
    assert pairs(claims) == [("Beneventum", "Brundisium")]
    assert origins(events[-1]) == ["Beneventum"]
    assert events[-1].actor.actor_status is EventActorStatus.UNKNOWN


@pytest.mark.parametrize("continuation", [
    "Years later from there Marcus marched to C.",
    "From there Marcus marched to C in a new campaign.",
    "From there Marcus marched to C during another war.",
])
def test_occurrence_boundary_breaks_continuity(continuation):
    events, claims = extract("Marcus marched from A to B. " + continuation)
    assert not any(c.movement_relation == "discourse_continuation" for c in claims)
    assert all(not origins(e) for e in events if e.summary == continuation)


def test_paragraph_boundary_breaks_continuity():
    events, claims = extract("Marcus marched from A to B.\n\nFrom there he proceeded to C.")
    assert pairs(claims) == [("A", "B")]
    assert not origins(events[-1])


def test_evidence_boundary_breaks_continuity():
    items = [evidence("one", "Marcus marched from A to B."), evidence("two", "From there he proceeded to C.")]
    claims = LOCAL.movement_claims(items, event_id="boundary")
    events, _ = EvidenceGroundedHistoricalEventExtractor(LOCAL).extract(items)
    assert pairs(claims) == [("A", "B")]
    assert not origins(events[-1])


def test_query_and_parent_metadata_cannot_supply_predecessor():
    item = evidence("isolated", "From there he proceeded to C.", parent_text="Marcus marched from A to B.")
    events, _ = EvidenceGroundedHistoricalEventExtractor(LOCAL).extract([item], query="Marcus marched from A to B")
    assert not LOCAL.movement_claims([item], event_id="isolated")
    assert all(not origins(e) for e in events)


@pytest.mark.parametrize("destination", ["Rome", "Roma", "Oricum", "Orikon", "Brundisium", "Brundusium"])
def test_antecedent_keeps_semantic_canonical_identity(destination):
    x = HistoricalPlaceMentionExtractor()
    first = f"Marcus marched from Capua to {destination}."
    semantic = analyze_sentence(first, x.aliases_in(first))
    endpoint = next(p for p in semantic.endpoints if p.role == "destination")
    events, claims = extract(first + " From there he proceeded to Corcyra.", x)
    assert claims[-1].source_place == endpoint.place_name
    assert origins(events[-1]) == [endpoint.place_name]


def test_place_never_becomes_inherited_actor():
    events, claims = extract("Marcus marched from Rome to Cilicia. From there he proceeded to Greece.", HistoricalPlaceMentionExtractor())
    assert pairs(claims)[-1] == ("Cilicia", "Greece")
    assert origins(events[-1]) == ["Cilicia"]
    assert events[-1].actor.actor_status is EventActorStatus.UNKNOWN


def test_destination_there_is_not_treated_as_origin_anaphora():
    events, claims = extract("Marcus reached B. He went there from A.")
    assert not claims
    assert all("B" not in origins(e) for e in events[1:])


def test_claim_existence_does_not_bypass_strict_endpoint_admission():
    text = "Marcus marched from Rome to Capua. From there Marcus marched to Brundisium."
    result, _, _ = build([text], query="Trace Marcus's route from Rome to Capua.")
    assert labels(result) == ["Roma", "Capua"]


def test_pronoun_does_not_gain_query_actor_authority_or_full_route():
    text = "Marcus marched from Rome to Capua. From there he proceeded to Brundisium."
    result, events, _ = build([text], query="Trace Marcus's route from Rome to Brundisium.")
    assert events[-1].actor.actor_status is EventActorStatus.UNKNOWN
    assert result.route is None
