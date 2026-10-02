"""Unseen non-possessive route subjects constrain admission without proving actors."""
import pytest

from backend.app.agent.evidence_support import _subject_alias_registry, extract_subject_terms
from backend.app.models import EventActorStatus
from backend.app.rag.query_bridge import V1_ENTRIES
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.query_route_admission import AuthorityState, parse_query_route_scope
from backend.tests.test_agent_loop import ev
from backend.tests.test_v2gen9_query_subject_authority import control

AURELIUS = "Aurelius Nestor"
FRESH = ("Dorieus Melanthos", "Cassius Varro Minor", "Tiberius Aelianus")
BARE_FORMS = (
    "Show {name} route from Rome to Capua.",
    "Display {name} route from Rome to Capua.",
    "Route for {name} from Rome to Capua.",
    "Trace the movement of {name} from Rome to Capua.",
    "Reconstruct {name} movement from Rome to Capua.",
    "{name} route: Rome to Capua.",
)


def _scope(query: str):
    return parse_query_route_scope((query,))


@pytest.mark.parametrize("template", BARE_FORMS)
def test_unseen_nonpossessive_subject_is_preserved(template):
    query = template.format(name=AURELIUS)
    scope = _scope(query)
    assert scope.subject == AURELIUS and not scope.subject_ambiguous
    assert extract_subject_terms(query) == ()


@pytest.mark.parametrize("query", [
    "Aurelius Nestor's route from Rome to Capua.",
    "Trace Aurelius Nestor's movement from Rome to Capua.",
    "Show Neralis Vexon's route from Rome to Capua.",
])
def test_unseen_possessive_subject_remains(query):
    assert _scope(query).subject in {AURELIUS, "Neralis Vexon"}
    assert not _scope(query).subject_ambiguous


@pytest.mark.parametrize("template", BARE_FORMS)
@pytest.mark.parametrize("actor,allowed", [(AURELIUS, True), ("Pompey", False)])
def test_unseen_subject_checks_evidence_actor_without_relabeling(template, actor, allowed):
    query = template.format(name=AURELIUS)
    evidence, events, result, admissions = control(query, actor)
    assert events[0].actor.actor_text == actor
    assert events[0].actor.actor_status is EventActorStatus.EXPLICIT
    assert evidence.text.startswith(actor)
    assert admissions
    assert all(a.subject_match is (AuthorityState.MATCH if allowed else AuthorityState.WRONG) for a in admissions)
    assert all(a.admitted is allowed for a in admissions)
    assert (result.route is not None) is allowed


def test_query_name_does_not_prove_actor():
    query = "Show Aurelius Nestor route from Rome to Capua."
    evidence = ev("pronoun", "He marched from Rome to Capua.")
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([evidence])
    assert all(event.actor.actor_text != AURELIUS for event in events)
    assert all(event.actor.actor_status is not EventActorStatus.EXPLICIT or event.actor.actor_text != AURELIUS for event in events)
    _, events, result, admissions = control(query, "He")
    assert all(event.actor.actor_text != AURELIUS for event in events)
    assert result.route is None
    if admissions:
        assert all(a.subject_match is not AuthorityState.MATCH for a in admissions)


@pytest.mark.parametrize("query,subject", [
    ("Show Caesar route from Rome to Capua.", "caesar"),
    ("展示凯撒路线", "caesar"),
    ("Show Hannibal route across the Alps.", "hannibal"),
    ("追踪汉尼拔翻越阿尔卑斯的路线", "hannibal"),
])
def test_known_alias_normalization_remains(query, subject):
    scope = _scope(query)
    assert scope.subject == subject and not scope.subject_ambiguous


def test_gen9_wrong_caesar_actor_still_rejects_pompey():
    _, events, result, admissions = control("Show Caesar route from Rome to Capua.", "Pompey", "Caesar remained in camp. ")
    assert events[0].actor.actor_text == "Pompey"
    assert result.route is None
    assert admissions and all(a.subject_match is AuthorityState.WRONG and not a.admitted for a in admissions)


@pytest.mark.parametrize("query", [
    "Show Rome route to Capua.",
    "Display Delphi route to Rome.",
    "Route from Oricum to Brundisium.",
])
def test_place_only_routes_do_not_invent_a_person(query):
    scope = _scope(query)
    assert scope.subject is None and not scope.subject_ambiguous


@pytest.mark.parametrize("query", [
    "Show Battle of Cannae route.",
    "Trace Second Punic War route.",
])
def test_event_names_do_not_become_people(query):
    scope = _scope(query)
    assert scope.subject is None and not scope.subject_ambiguous


@pytest.mark.parametrize("query", [
    "Show the army route from Rome to Capua.",
    "Trace the consul movement from Rome to Capua.",
    "Show the commander route from Rome to Capua.",
    "Trace the troops movement from Rome to Capua.",
    "Show the forces route from Rome to Capua.",
])
def test_roles_and_collectives_stay_unidentified(query):
    scope = _scope(query)
    assert scope.subject is None and not scope.subject_ambiguous


@pytest.mark.parametrize("query", [
    "Show the route described by Polybius.",
    "According to Livy, show the route from Rome to Capua.",
])
def test_documentary_sources_are_not_requested_movers(query):
    scope = _scope(query)
    assert scope.subject is None and not scope.subject_ambiguous
    _, events, result, admissions = control(query, "Pompey")
    assert events[0].actor.actor_text == "Pompey"
    assert result.route is not None
    assert admissions and all(a.subject_match is AuthorityState.MATCH for a in admissions)


@pytest.mark.parametrize("query", [
    "Show Aurelius Nestor and Pompey routes from Rome.",
    "Compare Aurelius Nestor route with Pompey.",
    "Trace the movement of Aurelius Nestor and Caesar.",
    "Show Dorieus Melanthos and Cassius Varro Minor routes from Rome.",
])
def test_multi_person_queries_do_not_select_one_subject(query):
    scope = _scope(query)
    assert scope.subject is None and scope.subject_ambiguous
    _, events, result, admissions = control(query, "Pompey")
    assert events[0].actor.actor_text == "Pompey"
    assert result.route is None
    assert admissions and all(a.subject_match is AuthorityState.UNKNOWN and not a.admitted for a in admissions)


@pytest.mark.parametrize("query,forbidden", [
    ("Explain Aurelius Nestor's movement from Rome to Capua.", "explain"),
    ("Show Aurelius Nestor route from Rome to Capua.", "show"),
    ("Trace the movement of Aurelius Nestor.", "trace"),
    ("Display Aurelius Nestor route from Rome to Capua.", "display"),
    ("Reconstruct Aurelius Nestor movement from Rome to Capua.", "reconstruct"),
])
def test_command_verb_is_not_part_of_subject(query, forbidden):
    subject = _scope(query).subject
    assert subject is not None
    assert forbidden not in subject.casefold().split()
    assert subject.casefold() == AURELIUS.casefold()


@pytest.mark.parametrize("name", FRESH)
@pytest.mark.parametrize("template", [
    "Show {name} route from Rome to Capua.",
    "Trace the movement of {name} from Rome to Capua.",
    "Route for {name} from Rome to Capua.",
])
def test_fresh_unseen_names_stay_out_of_registries(name, template):
    query = template.format(name=name)
    assert extract_subject_terms(query) == ()
    registry = str(_subject_alias_registry()).lower()
    assert name.lower() not in registry
    assert name.lower() not in str(V1_ENTRIES).lower()
    assert _scope(query).subject == name


@pytest.mark.parametrize("name", FRESH)
def test_fresh_names_match_only_their_own_evidence_actor(name):
    query = f"Show {name} route from Rome to Capua."
    _, events, result, admissions = control(query, name)
    assert events[0].actor.actor_text == name
    assert result.route is not None
    assert all(a.subject_match is AuthorityState.MATCH and a.admitted for a in admissions)
    _, wrong_events, wrong_result, wrong_admissions = control(query, "Pompey")
    assert wrong_events[0].actor.actor_text == "Pompey"
    assert wrong_result.route is None
    assert all(a.subject_match is AuthorityState.WRONG and not a.admitted for a in wrong_admissions)


def test_gen1_registry_independence_for_nonpossessive_form():
    name = "Dorieus Melanthos"
    query = f"Show {name} route from Rome to Capua."
    assert extract_subject_terms(query) == ()
    assert name.lower() not in str(V1_ENTRIES).lower()
    assert "dorieus" not in str(_subject_alias_registry()).lower()
    scope = _scope(query)
    assert scope.subject == name and not scope.subject_ambiguous
