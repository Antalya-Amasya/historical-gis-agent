"""Source alternatives stay unresolved through the ordinary route tool path."""
import socket
import pytest

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentState, EventPlaceRole
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias, HISTORICAL_PLACE_ALIASES
from backend.app.routes.route_observations import (
    RouteObservation, RouteObservationKind, _typed_same_event_relations,
)
from test_agent_loop import ev, Retriever, call, terminal
from test_v2auth1_strict_endpoint_admission import build, labels, CanonicalPlaces
from test_v1_1g2_waypoint_observation_contract import _event, ALPHA


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Endpoint tests must not contact external providers")
    monkeypatch.setattr(socket.socket, "connect", blocked)


M = HistoricalPlaceMentionExtractor()


@pytest.mark.parametrize("source", [
    "from Rome to Capua or Neapolis",
    "from Rome or Capua to Neapolis",
    "from Rome or Capua to Neapolis or Brundisium",
    "from Rome to Capua/Neapolis",
    "from Rome to either Capua or Neapolis",
    "from Rome or perhaps Capua to Neapolis",
    "from Neapolis or Capua to Rome",
    "from Rome to Capua or Neapolis or Brundisium",
    "from Rome/Capua to Neapolis/Brundisium",
    "from either Rome or Capua to Neapolis",
])
@pytest.mark.parametrize("query", ["Trace Marcus's route.", "Trace Marcus from Rome to Capua.",
                                  "Trace Marcus from Rome to Neapolis."])
def test_alternatives_never_become_unique_edges(source, query):
    text = f"Marcus marched {source}."
    semantics = analyze_sentence(text, M.aliases_in(text))
    assert semantics.should_abstain and semantics.abstain_reason == "ambiguous_endpoint_alternatives"
    assert semantics.endpoints and not semantics.edges and not semantics.route_orderings
    assert not M.movement_claims([ev("alt", text)], event_id="alt")
    result, events, _ = build([text], query)
    assert not labels(result) and not result.observation_relations
    assert events and all(not e.route_orderings for e in events)


def test_candidate_sets_preserve_unique_opposite_role_and_mentions():
    text = "Marcus marched from Rome to Capua or Neapolis."
    semantics = analyze_sentence(text, M.aliases_in(text))
    assert {p.place_name for p in semantics.endpoints if p.role == "destination"} == {"Capua", "Neapolis"}
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([ev("alt", text)])
    roles = {p.raw_text: p.role for p in events[0].place_mentions}
    assert roles["Rome"] is EventPlaceRole.ORIGIN
    assert roles["Capua"] is EventPlaceRole.RELATED_PLACE
    assert roles["Neapolis"] is EventPlaceRole.RELATED_PLACE


@pytest.mark.parametrize("source", ["from Rome to Capua", "from Rome, the capital, to Capua",
    "from Rome to Capua in Campania", "from Rome to Brundisium, a port",
    "from Rome/Roma to Capua", "from Rome or Roma to Capua"])
def test_unique_and_contextual_controls(source):
    text = f"Marcus marched {source}."
    semantics = analyze_sentence(text, M.aliases_in(text))
    assert not semantics.should_abstain and len(semantics.edges) == 1
    result, _, _ = build([text], "Trace Marcus's route.")
    assert labels(result) == ["Roma", "Brundisium" if "Brundisium" in source else "Capua"]


def test_canonical_equivalent_custom_aliases_are_unique():
    aliases = HISTORICAL_PLACE_ALIASES + (HistoricalPlaceAlias("Orikon", ("oricum", "orikon")),)
    m = HistoricalPlaceMentionExtractor(aliases)
    for join in ("/", " or "):
        text = f"Marcus marched from Oricum{join}Orikon to Capua."
        sem = analyze_sentence(text, m.aliases_in(text))
        assert not sem.should_abstain and len(sem.edges) == 1
        assert sem.edges[0].origin.canonical == "Orikon"
        assert len(m.movement_claims([ev("alias", text)], event_id="alias")) == 1


def test_and_is_not_or_and_does_not_invent_sequence():
    text = "Marcus marched from Rome to Capua and Neapolis."
    sem = analyze_sentence(text, M.aliases_in(text))
    assert sem.should_abstain and sem.abstain_reason == "ambiguous_endpoint_conjunction"
    assert not sem.edges
    assert not labels(build([text], "Trace Marcus's route.")[0])


@pytest.mark.parametrize("suffix", ["From there he marched to Brundisium.",
    "He later marched from Capua to Brundisium."])
def test_later_sentence_cannot_resolve_prior_alternative(suffix):
    text = f"Marcus marched from Rome to Capua or Neapolis. {suffix}"
    claims = M.movement_claims([ev("later", text)], event_id="later")
    assert all(c.source_place != "Roma" for c in claims)
    assert all(c.movement_relation != "discourse_continuation" for c in claims)
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([ev("later", text)])
    assert not events[0].route_orderings
    assert all(p.role is not EventPlaceRole.DESTINATION for p in events[0].place_mentions)
    if suffix.startswith("From there"):
        assert all(p.role is not EventPlaceRole.ORIGIN for e in events[1:] for p in e.place_mentions)


@pytest.mark.parametrize("text", ["Marcus attempted to reach Capua or Neapolis.",
    "Marcus was prevented from entering Capua or Neapolis.",
    "Pompey marched from Rome to Capua or Neapolis."])
def test_noncompletion_and_wrong_actor(text):
    assert not labels(build([text], "Trace Marcus's route.")[0])


@pytest.mark.parametrize("role", [EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION])
def test_same_event_fallback_requires_canonical_uniqueness(role):
    e = _event("alt", "Commander Alpha marched from Rome to Capua.", [], ["e"])
    def obs(label, r):
        return RouteObservation(observation_id=label, kind=RouteObservationKind.PLACE,
            event_id=e.id, label=label, actor_text=ALPHA.actor_text,
            actor_status=ALPHA.actor_status, evidence_refs=("e",), place_role=r)
    observations = [obs("Roma", EventPlaceRole.ORIGIN), obs("Capua", EventPlaceRole.DESTINATION)]
    assert len(_typed_same_event_relations(e, observations, [])) == 1
    observations.append(obs("Neapolis", role))
    assert not _typed_same_event_relations(e, observations, [])
    assert not _typed_same_event_relations(e, list(reversed(observations)), [])


@pytest.mark.parametrize("source", ["from Veloria to Neralon or Darsena",
    "from Veloria or Galveth to Darsena", "from Veloria/Galveth to Darsena/Neralon"])
def test_fresh_test_only_places(source):
    aliases = tuple(HistoricalPlaceAlias(x, (x.lower(),), "test-only")
                    for x in ("Veloria", "Neralon", "Darsena", "Galveth"))
    m = HistoricalPlaceMentionExtractor(aliases)
    text = f"Marcus marched {source}."
    sem = analyze_sentence(text, m.aliases_in(text))
    assert sem.should_abstain and not sem.edges and len(sem.endpoints) >= 3
    assert not m.movement_claims([ev("fresh", text)], event_id="fresh")


@pytest.mark.parametrize("source,expected", [
    ("from Rome to Capua or Neapolis", []),
    ("from Rome or Capua to Neapolis", []),
    ("from Rome to Capua", ["Roma", "Capua"]),
])
@pytest.mark.parametrize("query", ["Trace Marcus's route.", "Trace Marcus from Rome to Capua."])
def test_ordinary_agent_route_tool(source, expected, query):
    text = f"Marcus marched {source}."
    a = HistoricalGisAgent(ScriptedLLMProvider([
        call("search_historical_evidence", {"query": query}),
        call("build_historical_route", {"event_id": "alt", "name": "alt", "period": "unspecified"}),
        terminal("The source describes Marcus.", ["alt"]),
    ]), Retriever([ev("alt", text)]), CanonicalPlaces(), max_grounding_corrections=0)
    _, state = a.respond(query, AgentState(session_id="endpoint-alternatives"))
    actual = [p.historical_place.canonical_name for p in state.historical_route.ordered_points] if state.historical_route else []
    assert actual == expected
    assert any(t.tool_name == "build_historical_route" for t in state.tool_history)


def test_bare_departure_alternatives_cannot_reenter_fallback():
    text = "Marcus left Rome or Capua and reached Neapolis."
    sem = analyze_sentence(text, M.aliases_in(text))
    assert sem.should_abstain and not sem.edges
    assert not labels(build([text], "Trace Marcus's route.")[0])


def test_duplicate_canonical_observations_are_not_ambiguity():
    e = _event("same", "Commander Alpha marched from Rome to Capua.", [], ["e"])
    observations = [RouteObservation(observation_id=label, kind=RouteObservationKind.PLACE,
        event_id=e.id, label=label, actor_text=ALPHA.actor_text,
        actor_status=ALPHA.actor_status, evidence_refs=("e",), place_role=role)
        for label, role in (("Roma", EventPlaceRole.ORIGIN), ("Roma", EventPlaceRole.ORIGIN),
                            ("Capua", EventPlaceRole.DESTINATION))]
    assert len(_typed_same_event_relations(e, observations, [])) == 1
