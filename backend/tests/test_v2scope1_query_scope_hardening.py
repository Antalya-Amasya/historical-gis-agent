"""Query syntax constrains admission; presentation text supplies no authority."""
import pytest

from backend.app.agent.loop import infer_requested_output
from backend.app.models import EventActorStatus
from backend.app.routes.query_route_admission import AuthorityState, parse_query_route_scope
from backend.app.routes.query_scope_parser import QuerySpanRole, parse_query_scope
from backend.tests.test_v2auth1_strict_endpoint_admission import build, labels
from backend.tests.test_v2gen9_query_subject_authority import control
from backend.tests.test_v2gen12_query_scope import PICTOR_QUERY

BASE = "Reconstruct Pictor's route from Delphi to Rome."
NOISY = "Reconstruct Pictor's route from Delphi to Rome, then divide the answer into evidence, simulation and uncertainty."
INSTRUCTIONS = (
    NOISY,
    "Trace Pictor's route from Delphi to Rome and explain the evidence separately.",
    "Show Pictor's route from Delphi to Rome. Return the answer in three sections.",
    "Explain Pictor's movement from Delphi to Rome and then summarize the uncertainty.",
    "Trace the route from Delphi to Rome; format the answer as a table.",
    "Explain Pictor route from Delphi to Rome, then format the answer as three sections.",
    "Trace Pictor's route from Delphi to Rome; summarize in a table.",
    "Trace Pictor's route from Delphi to Rome; return three sections.",
    "Trace Pictor's route from Delphi to Rome; show evidence, simulation, uncertainty.",
)


def scope(query):
    return parse_query_route_scope((query,))


@pytest.mark.parametrize("query", (BASE, *INSTRUCTIONS))
def test_presentation_preserves_strict_endpoints_and_route_phase(query):
    s = scope(query)
    assert infer_requested_output(query) == "historical_route"
    assert (s.origin, s.destination, s.endpoint_strict) == ("Delphi", "Rome", True)
    assert s.has_endpoint_constraint and not s.subject_ambiguous
    result, _, admissions = build(["Pictor returned from Delphi to Rome."], query)
    assert labels(result) == ["Delphi", "Roma"]
    assert admissions and all(a.admitted and a.route_phase_match is AuthorityState.MATCH for a in admissions)


@pytest.mark.parametrize("tail", (
    "then format the answer using Pompey's movement from Capua to Rome during Campaign Beta in 80 BCE.",
    "explain the evidence described by Polybius from Capua to Rome.",
))
def test_instruction_tail_cannot_supply_subject_endpoints_episode_or_time(tail):
    query = BASE[:-1] + ", " + tail
    s = scope(query)
    assert (s.subject, s.origin, s.destination) == ("Pictor", "Delphi", "Rome")
    assert not s.has_episode_constraint and not s.has_temporal_constraint and not s.subject_ambiguous
    parsed = parse_query_scope(query)
    spans = [x for x in parsed.spans if x.role is QuerySpanRole.CONTEXT]
    assert any(tail in x.text for x in spans)
    assert all(query[x.start:x.end] == x.text for x in spans)


@pytest.mark.parametrize("query", (
    "Trace Pictor's route from Delphi to Rome and then to Capua.",
    "Trace Pictor's route from Delphi to Rome and then from Rome to Capua.",
    "Trace Pictor's route from Delphi to Rome and then to Capua; format the answer as a table.",
))
def test_real_continuation_is_not_trimmed_to_a_false_strict_pair(query):
    s = scope(query)
    assert s.origin is None and s.destination is None and s.has_endpoint_constraint


def test_via_and_historical_episode_are_retained():
    query = "Trace the route from Delphi to Rome via Brundisium."
    parsed = parse_query_scope(query)
    assert (parsed.origin, parsed.destination) == ("Delphi", "Rome")
    assert any(s.role is QuerySpanRole.VIA and s.text == "Brundisium" for s in parsed.spans)
    s = scope("Trace Pictor's route from Delphi to Rome during Campaign Alpha; format the answer as a table.")
    assert s.has_episode_constraint and "alpha" in s.episode.casefold()


@pytest.mark.parametrize("command", ("Show", "Display", "Explain", "Reconstruct", "Trace", "Describe"))
@pytest.mark.parametrize("name", ("Pictor", "Dorieus Melanthos", "Cassius Varro Minor", "Tiberius Aelianus"))
def test_command_subject_route_noun(command, name):
    for noun in ("route", "movement"):
        s = scope(f"{command} {name} {noun} from Rome to Capua.")
        assert s.subject == name and not s.subject_ambiguous
        assert (s.origin, s.destination, s.endpoint_strict) == ("Rome", "Capua", True)


@pytest.mark.parametrize("query", (
    "Explain Aurelius Nestor and Pompey routes from Rome.",
    "Compare Pictor route with Pompey.",
    "Trace Aurelius Nestor and Caesar movement.",
    "Explain Dorieus Melanthos and Cassius Varro Minor routes from Rome to Capua.",
))
def test_multiple_requested_people_remain_ambiguous(query):
    s = scope(query)
    assert s.subject is None and s.subject_ambiguous
    _, events, result, admissions = control(query, "Pompey")
    assert events[0].actor.actor_text == "Pompey" and result.route is None
    assert admissions and all(a.subject_match is AuthorityState.UNKNOWN and not a.admitted for a in admissions)


@pytest.mark.parametrize("query", (
    "Explain Rome route to Capua.",
    "Show Battle of Cannae route.",
    "Explain Second Punic War route.",
    "Trace the consul route.",
    "Show the army route.",
    "Explain the commander route.",
    "Explain Consul route from Rome to Capua.",
    "According to Livy, explain the route from Rome to Capua.",
    "Explain the route described by Polybius.",
    "Explain Dorieus Melanthos wrote a book.",
))
def test_places_events_roles_sources_and_nonroute_names_are_not_movers(query):
    s = scope(query)
    assert s.subject is None and not s.subject_ambiguous


@pytest.mark.parametrize("actor,match", (("Aurelius Nestor", AuthorityState.MATCH), ("Pompey", AuthorityState.WRONG), ("He", AuthorityState.UNKNOWN)))
def test_query_subject_does_not_supply_an_evidence_actor(actor, match):
    query = "Explain Aurelius Nestor route from Rome to Capua."
    _, events, result, admissions = control(query, actor)
    assert events[0].actor.actor_text != "Aurelius Nestor" or actor == "Aurelius Nestor"
    if actor == "He":
        assert events[0].actor.actor_status is not EventActorStatus.EXPLICIT
    if actor != "He":
        assert admissions
    assert all(a.subject_match is match for a in admissions)
    assert (result.route is not None) is (match is AuthorityState.MATCH)


def test_instruction_does_not_weaken_strict_endpoint_rejection():
    result, _, admissions = build(["Pictor marched from Rome to Capua."], NOISY)
    assert result.route is None and admissions
    assert all(a.route_phase_match is not AuthorityState.MATCH and not a.admitted for a in admissions)


def test_existing_full_name_pictor_remains_admitted_without_new_surname_authority():
    text = "Quintus Fabius Pictor returned from Delphi to Rome."
    result, _, admissions = build([text], PICTOR_QUERY)
    assert labels(result) == ["Delphi", "Roma"] and all(a.admitted for a in admissions)
    # Existing identity is exact/alias-based; query parsing must not widen it.
    result, _, admissions = build([text], "Explain Pictor route from Delphi to Rome.")
    assert scope("Explain Pictor route from Delphi to Rome.").subject == "Pictor"
    assert result.route is None and all(a.subject_match is AuthorityState.WRONG for a in admissions)


@pytest.mark.parametrize("query,subject", (
    ("展示凯撒路线", "caesar"),
    ("追踪汉尼拔翻越阿尔卑斯的路线", "hannibal"),
))
def test_known_cjk_aliases_unchanged(query, subject):
    assert scope(query).subject == subject


def test_front_presentation_clause_does_not_obscure_route():
    s = scope("In three sections, explain Pictor route from Delphi to Rome.")
    assert (s.subject, s.origin, s.destination) == ("Pictor", "Delphi", "Rome")


@pytest.mark.parametrize("query", (
    "Explain why Caesar retreated from Rome to Capua.",
    "Describe the political consequences of Caesar's movement from Rome to Capua.",
    "Explain Caesarea route.",
    "Display Delphi route to Rome.",
))
def test_unbounded_capitalized_words_and_analytical_questions_stay_safe(query):
    if "route" not in query:
        assert infer_requested_output(query) == "answer"
    else:
        assert scope(query).subject is None


def test_unseen_name_with_unseen_endpoint_labels():
    s = scope("Explain Dorieus Melanthos route from Alpha to Beta.")
    assert (s.subject, s.origin, s.destination) == ("Dorieus Melanthos", "Alpha", "Beta")


def test_presentation_section_count_is_not_a_fixed_prompt_template():
    s = scope("Trace Pictor's route from Delphi to Rome; return seven sections.")
    assert (s.origin, s.destination) == ("Delphi", "Rome")
    assert any(x.role is QuerySpanRole.CONTEXT for x in parse_query_scope(
        "Trace Pictor's route from Delphi to Rome; return seven sections."
    ).spans)


def test_explicit_source_authors_own_route_is_a_subject():
    assert scope("Explain Polybius route from Rome to Capua.").subject.casefold() == "polybius"
