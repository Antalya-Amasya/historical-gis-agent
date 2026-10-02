"""Evidence-local coordination preserves actors without supplying movement authority."""
import pytest
from backend.app.models import EventActorStatus, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor as Extractor
from backend.tests.test_agent_loop import ev
from backend.tests.test_v2auth1_strict_endpoint_admission import build, labels

QUERY = "Trace Ariston's route."


def extract(text):
    return Extractor().extract([ev("coordination", text)])


def assert_grounded(event, name):
    actor = event.actor
    assert actor.actor_status is EventActorStatus.EXPLICIT and actor.actor_text == name
    statement = event.source_statements[0]
    start, end = actor.actor_span
    assert statement[start:end] == name
    a, b = actor.actor_clause_span
    assert a <= start < end <= b <= len(statement)


@pytest.mark.parametrize("text", [
    "Ariston did not hesitate and marched from Rome to Capua.",
    "Ariston marched from Rome to Capua.",
    "Ariston refused to wait and advanced from Rome to Capua.",
    "Ariston was not delayed and travelled from Rome to Capua.",
    "Ariston spoke to the envoys and then marched from Rome to Capua.",
    "Ariston, without hesitation, marched from Rome to Capua.",
])
def test_supported_positive_actor_and_route(text):
    events, _ = extract(text)
    assert len(events) == 1
    assert_grounded(events[0], "Ariston")
    result, _, _ = build([text], QUERY)
    assert labels(result) == ["Roma", "Capua"]


@pytest.mark.parametrize("text", [
    "Ariston did not remain in Rome but marched to Capua.",
    "Ariston entered Rome, rested, and marched to Capua.",
    "Ariston marched from Rome to Beneventum and continued to Capua.",
])
def test_actor_continuity_does_not_invent_endpoints(text):
    events, _ = extract(text)
    assert_grounded(events[0], "Ariston")
    if "did not remain" in text:
        assert all(m.role.value != "ORIGIN" for m in events[0].place_mentions)


@pytest.mark.parametrize("text,actor", [
    ("Ariston spoke to Pompey, and Pompey marched from Rome to Capua.", "Pompey"),
    ("Ariston and Pompey argued; Pompey marched from Rome to Capua.", "Pompey"),
    ("Ariston arrived. Caesar marched from Rome to Capua.", "Caesar"),
])
def test_competing_explicit_actor_is_local_actor(text, actor):
    events, _ = extract(text)
    movement = [e for e in events if e.event_type is HistoricalEventType.MOVEMENT]
    assert_grounded(movement[-1], actor)


@pytest.mark.parametrize("text", [
    "Ariston spoke to Pompey, and he marched from Rome to Capua.",
    "Ariston did not hesitate. He marched from Rome to Capua.",
    "Ariston did not hesitate; instead, he marched from Rome to Capua.",
    "Ariston spoke to Pompey and marched from Rome to Capua.",
    "Ariston watched while Pompey waited and marched from Rome to Capua.",
    "Ariston spoke; marched from Rome to Capua.",
    "Ariston and Pompey did not hesitate and marched from Rome to Capua.",
    "Ariston said that he was ready and marched from Rome to Capua.",
    "Ariston spoke, a messenger waited and marched from Rome to Capua.",
    "Ariston spoke although a guide waited and marched from Rome to Capua.",
    "Ariston and marched from Rome to Capua.",
])
def test_no_unbounded_or_ambiguous_actor_carryover(text):
    events, _ = extract(text)
    assert all(e.actor.actor_text != "Ariston" for e in events if e.event_type is HistoricalEventType.MOVEMENT)


@pytest.mark.parametrize("subject", ["The army", "The consul", "The commander"])
def test_roles_and_collectives_are_not_people(subject):
    events, _ = extract(f"{subject} did not hesitate and marched from Rome to Capua.")
    assert events and all(e.actor.actor_status is EventActorStatus.UNKNOWN for e in events)


@pytest.mark.parametrize("text,outcome", [
    ("Ariston did not march from Rome to Capua.", "NEGATED"),
    ("Ariston never reached Capua.", None),
    ("Ariston was unable to enter Capua.", None),
    ("Ariston did not hesitate and advanced from Rome toward Capua but stopped before reaching it.", "ABORTED"),
    ("Ariston spoke confidently and attempted to enter Capua but was prevented.", "PREVENTED"),
    ("Ariston did not hesitate and planned to march to Capua.", "PLANNED"),
    ("Ariston spoke and intended to travel to Capua.", None),
])
def test_actor_continuity_never_upgrades_non_completion(text, outcome):
    events, diagnostics = extract(text)
    assert not any(e.event_type is HistoricalEventType.MOVEMENT for e in events)
    if outcome:
        fact = diagnostics["incomplete_movement_facts"][0]
        assert fact["outcome"] == outcome and fact["actor"]["actor_text"] == "Ariston"
    result, _, _ = build([text], QUERY)
    assert result.route is None


def test_negative_clause_does_not_erase_subject_of_later_completed_movement():
    text = "Ariston did not march to Capua but later travelled from Rome to Beneventum."
    events, diagnostics = extract(text)
    assert len(events) == 1
    assert_grounded(events[0], "Ariston")
    assert diagnostics["incomplete_movement_facts"][0]["outcome"] == "NEGATED"
    assert "Capua" not in [p.raw_text for p in events[0].place_mentions]
    result, _, _ = build([text], QUERY)
    assert labels(result) == ["Roma", "Beneventum"]


@pytest.mark.parametrize("name", ["Damon Philostratos", "Lucius Aemilius Varro", "Titus Calpurnius Severus"])
def test_fresh_evidence_names_need_no_registry(name):
    text = f"{name} did not hesitate and marched from Rome to Capua."
    events, _ = extract(text)
    assert_grounded(events[0], name)
    result, _, _ = build([text], f"Trace {name}'s route.")
    assert labels(result) == ["Roma", "Capua"]


def test_query_cannot_supply_actor_for_elided_or_pronoun_statement():
    result, events, _ = build(["He did not hesitate and marched from Rome to Capua."], QUERY)
    assert result.route is None
    assert all(e.actor.actor_text != "Ariston" for e in events)


@pytest.mark.parametrize("text", [
    "Ariston did not stop and continued from Rome to Capua.",
    "Before marching from Rome to Capua, Ariston did not hesitate.",
    "Ariston camped outside Rome before marching to Capua.",
])
def test_unsupported_grammar_is_not_repaired_by_query(text):
    events, _ = extract(text)
    assert all(e.actor.actor_status is EventActorStatus.UNKNOWN for e in events)
