"""Boundary query punctuation normalizes before unchanged route authority checks."""
import socket
import pytest
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentState
from backend.app.routes.query_scope_parser import _clean, parse_query_scope
from backend.app.routes.query_route_admission import parse_query_route_scope, AuthorityState
from test_agent_loop import ev, Retriever, call, terminal
from test_v2auth1_strict_endpoint_admission import build, labels, CanonicalPlaces


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Punctuation tests must not contact providers")
    monkeypatch.setattr(socket.socket, "connect", blocked)


SURFACES = ("Capua", "Capua?", "Capua!", "Capua.", "Capua,", "Capua;", "Capua:",
            '"Capua"', "'Capua'", "“Capua”", "‘Capua’", "(Capua)", "Capua)",
            "Capua？", "Capua。", "Capua！", "Capua，", '"(Capua?)"!')


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("prefix", ["Trace Marcus", "How did Marcus move"])
def test_endpoint_syntax_is_semantically_equivalent(surface, prefix):
    q = f"{prefix} from Rome to {surface}"
    scope = parse_query_route_scope((q,))
    assert (scope.subject, scope.origin, scope.destination) == ("Marcus", "Rome", "Capua")
    assert scope.endpoint_strict and scope.has_endpoint_constraint
    result, _, admissions = build(["Marcus marched from Rome to Capua."], q)
    assert labels(result) == ["Roma", "Capua"]
    assert admissions and all(a.subject_match is AuthorityState.MATCH
        and a.route_phase_match is AuthorityState.MATCH and a.admitted for a in admissions)


@pytest.mark.parametrize("surface", ['"Rome"', "'Rome'", '“Rome”', '‘Rome’', '(Rome)', 'Rome)'])
def test_origin_wrappers_are_normalized(surface):
    q = f"Trace Marcus from {surface} to Capua?"
    assert parse_query_route_scope((q,)).origin == "Rome"
    assert labels(build(["Marcus marched from Rome to Capua."], q)[0]) == ["Roma", "Capua"]


@pytest.mark.parametrize("value", [*SURFACES, " Saint-Jean? ", "Aix-en-Provence!", "O'Connor?",
    "Aelius O'Connor", "Capua (Campania)", "Harbor (Old) North?", "(Capua) or (Neapolis)",
    "?", '""', '()', "O’Connor", "Harbor (Old))"])
def test_cleanup_is_idempotent(value):
    assert _clean(_clean(value)) == _clean(value)


@pytest.mark.parametrize("name", ["Saint-Jean", "Aix-en-Provence", "O'Connor", "O’Connor",
    "Capua (Campania)", "Harbor (Old) North", "(Capua) or (Neapolis)"])
def test_internal_punctuation_is_preserved(name):
    assert _clean(name) == name
    assert _clean(f'"{name}"?') == name
    if not name.startswith('('):
        assert parse_query_scope(f"Trace Marcus from Rome to {name}?").destination == name


@pytest.mark.parametrize("tail", [", then summarize it.", "; provide citations.",
    " — briefly explain why.", ", then summarize the evidence."])
def test_presentation_suffix_does_not_become_destination(tail):
    q = "Trace Marcus from Rome to Capua" + tail
    scope = parse_query_route_scope((q,))
    assert (scope.subject, scope.origin, scope.destination) == ("Marcus", "Rome", "Capua")
    assert labels(build(["Marcus marched from Rome to Capua."], q)[0]) == ["Roma", "Capua"]


@pytest.mark.parametrize("text", ["Lucius marched from Rome to Capua.",
    "Marcus marched from Rome to Capua or Neapolis.", "Marcus marched from Rome to Neapolis.",
    "Marcus marched from Capua to Rome.", "He marched from Rome to Capua."])
def test_normalization_does_not_weaken_authority(text):
    q = "How did Marcus move from Rome to Capua?"
    result, events, admissions = build([text], q)
    assert not labels(result)
    assert all(not a.admitted for a in admissions)
    if text.startswith("Lucius"):
        assert events[0].actor.actor_text == "Lucius"
        assert all(a.subject_match is AuthorityState.WRONG for a in admissions)


@pytest.mark.parametrize("origin,destination", [("Rome", "?"), ("Rome", '""'), ("Rome", "()"),
    ("?", "Capua"), ('""', "Capua"), ("()", "Capua")])
@pytest.mark.parametrize("prefix", ["Trace Marcus", "Show the route"])
def test_empty_surface_remains_requested_constraint(origin, destination, prefix):
    q = f"{prefix} from {origin} to {destination}"
    scope = parse_query_route_scope((q,))
    assert scope.has_endpoint_constraint and scope.origin is None and scope.destination is None
    result, _, admissions = build(["Marcus marched from Rome to Capua."], q)
    assert not labels(result)
    assert admissions and all(a.route_phase_match is AuthorityState.UNKNOWN and not a.admitted for a in admissions)


@pytest.mark.parametrize("origin", ["Rome", "Roma"])
def test_rome_canonical_alias_is_unchanged(origin):
    q = f"Trace Marcus from {origin} to Capua?"
    assert labels(build(["Marcus marched from Rome to Capua."], q)[0]) == ["Roma", "Capua"]


def test_oricum_source_to_orikon_resolved_alias_control():
    q = "Trace Marcus from Orikon to Capua?"
    assert _clean("Oricum?") == "Oricum" and _clean("Orikon?") == "Orikon"
    result, _, _ = build(["Marcus marched from Oricum to Capua."], q)
    assert labels(result) == ["Orikon", "Capua"]


@pytest.mark.parametrize("punctuation", ["?", "!", "."])
def test_fresh_places_and_multiword_endpoint(punctuation):
    for endpoint in ("Neralon", "Darsena", "Harbor Neralon"):
        q = f"Trace Aelius Dorion from Veloria to {endpoint}{punctuation}"
        scope = parse_query_route_scope((q,))
        assert (scope.subject, scope.origin, scope.destination) == ("Aelius Dorion", "Veloria", endpoint)
        result, _, _ = build([f"Aelius Dorion marched from Veloria to {endpoint}."], q)
        assert labels(result) == ["Veloria", endpoint]


@pytest.mark.parametrize("punctuation", ["？", "。", "！", "，"])
def test_cjk_surfaces_in_existing_from_to_frame(punctuation):
    s = parse_query_scope(f"Trace Marcus from 罗马 to 卡普阿{punctuation}")
    assert (s.subject, s.origin, s.destination) == ("Marcus", "罗马", "卡普阿")


def test_generic_query_and_colon_separator_stay_usable():
    q = "Show the route from Rome to Capua?"
    s = parse_query_route_scope((q,))
    assert s.subject is None and not s.subject_unresolved
    assert labels(build(["Lucius marched from Rome to Capua."], q)[0]) == ["Roma", "Capua"]
    assert parse_query_route_scope(("Aurelius Nestor route: Rome to Capua?",)).subject == "Aurelius Nestor"


@pytest.mark.parametrize("query", ["Trace Marcus from Rome to Capua?", "How did Marcus move from Rome to Capua?"])
@pytest.mark.parametrize("text,expected", [("Marcus marched from Rome to Capua.", ["Roma", "Capua"]),
    ("Lucius marched from Rome to Capua.", []), ("Marcus marched from Rome to Capua or Neapolis.", [])])
def test_ordinary_agent_path(query, text, expected):
    a = HistoricalGisAgent(ScriptedLLMProvider([
        call("search_historical_evidence", {"query": query}),
        call("build_historical_route", {"event_id": "punct", "name": "punct", "period": "unspecified"}),
        terminal("The source describes movement.", ["p"]),
    ]), Retriever([ev("p", text)]), CanonicalPlaces(), max_grounding_corrections=0)
    _, state = a.respond(query, AgentState(session_id="punctuation"))
    actual = [p.historical_place.canonical_name for p in state.historical_route.ordered_points] if state.historical_route else []
    assert actual == expected
    assert any(t.tool_name == "build_historical_route" for t in state.tool_history)
