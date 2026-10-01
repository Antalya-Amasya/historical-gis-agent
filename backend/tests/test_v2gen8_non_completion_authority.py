"""Non-completion blocks arrival authority independently of diagnostic endpoints."""
import pytest

from backend.app.agent.loop import infer_requested_output
from backend.app.models import EventActorStatus, EventPlaceRole
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_agent_loop import ev


class SyntheticPlaces:
    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        return {"found": True, "id": "synthetic-" + name, "canonical_name": name,
                "latitude": 40., "longitude": 15. if name.endswith("Alpha") else 16.,
                "source": "synthetic", "confidence": .99, "uncertain": False,
                "coordinate_role": "exact_site"}


def extract(text):
    evidence = ev("non-completion", text)
    events, diagnostics = EvidenceGroundedHistoricalEventExtractor().extract([evidence])
    return evidence, events, diagnostics["incomplete_movement_facts"]


def build(evidence, events, query="Trace Marcus's route from City Beta to City Alpha."):
    events, _ = HistoricalEventPlaceResolver(SyntheticPlaces()).resolve(events)
    return EventAnchorRouteBuilder().build_with_diagnostics(
        events, [evidence], event_id="control", name="control", period="unspecified",
        query_contexts=(query,),
    )


@pytest.mark.parametrize("verb", ["marched", "moved", "advanced", "proceeded", "travelled"])
def test_aborted_assertion_never_emits_arrival(verb):
    evidence, events, facts = extract(
        f"Marcus {verb} toward City Alpha but stopped before reaching it.")
    assert facts[0]["outcome"] == "ABORTED"
    assert facts[0]["actor"]["actor_text"] == "Marcus"
    assert all(p["role"] == "UNKNOWN" for p in facts[0]["destination_mentions"])
    assert events == []
    result = build(evidence, events)
    assert result.route is None
    assert result.diagnostics["observation_count"] == 0
    assert result.diagnostics["transition_constraint_count"] == 0


@pytest.mark.parametrize("verb", ["advanced", "proceeded", "travelled"])
def test_explicit_origin_cannot_make_unreached_destination_route(verb):
    evidence, events, facts = extract(
        f"Marcus {verb} from City Beta toward City Alpha but stopped before reaching it.")
    assert facts[0]["outcome"] == "ABORTED" and events == []
    # These verbs still have no recognized diagnostic destination: safety must
    # therefore follow outcome, not a richer endpoint list.
    assert facts[0]["destination_mentions"] == []
    result = build(evidence, events)
    assert result.route is None
    assert result.diagnostics["observation_count"] == 0
    assert result.diagnostics["transition_constraint_count"] == 0
    assert result.diagnostics["observation_component_count"] == 0


@pytest.mark.parametrize("verb", ["sailed", "departed", "returned"])
def test_other_actor_place_and_verb_vocabulary_is_not_a_special_case(verb):
    _, events, facts = extract(
        f"Dorieus {verb} toward Port Delta but stopped before reaching it.")
    assert facts[0]["outcome"] == "ABORTED"
    assert facts[0]["actor"]["actor_text"] == "Dorieus"
    assert events == []


@pytest.mark.parametrize("text,outcome", [
    ("Marcus attempted to enter City Alpha but was prevented.", "PREVENTED"),
    ("Marcus did not enter City Alpha.", "NEGATED"),
    ("Marcus advanced toward City Alpha but abandoned the advance before reaching it.", "ABORTED"),
    ("Marcus planned to march to City Alpha.", "PLANNED"),
    ("Marcus advanced from City Beta to City Alpha but aborted the advance.", "ABORTED"),
])
def test_other_non_completion_outcomes_remain_non_authoritative(text, outcome):
    evidence, events, facts = extract(text)
    assert facts[0]["outcome"] == outcome and events == []
    result = build(evidence, events)
    assert result.route is None
    assert result.diagnostics["observation_count"] == 0
    assert result.diagnostics["transition_constraint_count"] == 0


def test_prevented_clause_does_not_pollute_later_completed_endpoint():
    _, events, facts = extract(
        "Marcus was prevented from entering City Alpha, but later entered City Beta.")
    assert facts[0]["outcome"] == "PREVENTED"
    assert "City Beta" not in facts[0]["source_statement"]
    assert len(events) == 1
    assert [(p.raw_text, p.role) for p in events[0].place_mentions] == [
        ("City Beta", EventPlaceRole.DESTINATION)]
    assert events[0].actor.actor_status is EventActorStatus.UNKNOWN
    assert "City Alpha" not in events[0].summary


def test_same_place_prevented_first_completed_later_remains_positive():
    _, events, facts = extract(
        "Marcus was prevented from entering City Alpha, but later entered City Alpha.")
    assert facts[0]["outcome"] == "PREVENTED" and len(events) == 1
    assert events[0].place_mentions[0].role is EventPlaceRole.DESTINATION
    assert events[0].actor.actor_status is EventActorStatus.UNKNOWN


def test_aborted_sentence_does_not_cancel_later_completed_sentence():
    _, events, facts = extract(
        "Marcus advanced toward City Alpha but stopped before reaching it. Later he entered City Alpha.")
    assert facts[0]["outcome"] == "ABORTED" and len(events) == 1
    assert events[0].summary == "Later he entered City Alpha."
    assert events[0].actor.actor_status is EventActorStatus.UNKNOWN


def test_mixed_actor_does_not_transfer_marcus_identity():
    _, events, facts = extract(
        "Marcus advanced toward City Alpha but stopped before reaching it. Pompey later entered City Alpha.")
    assert facts[0]["actor"]["actor_text"] == "Marcus" and len(events) == 1
    assert events[0].actor.actor_text != "Marcus"
    # Current actor grounding does not recognize this adverb placement.
    assert events[0].actor.actor_status is EventActorStatus.UNKNOWN


def test_explicit_other_actor_completion_is_retained():
    _, events, facts = extract(
        "Dorieus advanced toward Port Delta but stopped before reaching it. Bion entered Port Delta.")
    assert facts[0]["actor"]["actor_text"] == "Dorieus" and len(events) == 1
    assert events[0].actor.actor_text == "Bion"
    assert events[0].actor.actor_status is EventActorStatus.EXPLICIT


def test_completed_contrast_can_still_form_its_own_route():
    evidence, events, facts = extract(
        "Marcus advanced toward City Alpha but stopped before reaching it but Bion marched from City Beta to City Gamma.")
    assert facts[0]["outcome"] == "ABORTED" and len(events) == 1
    assert events[0].actor.actor_text == "Bion"
    assert "City Alpha" not in events[0].summary
    result = build(evidence, events, "Trace Bion's route from City Beta to City Gamma.")
    assert result.route is not None
    assert [p.historical_place.canonical_name for p in result.route.ordered_points] == ["City Beta", "City Gamma"]


def test_other_actor_abandonment_preserves_completed_clause_only():
    _, events, facts = extract(
        "Marcus marched to City Beta but Bion abandoned the advance before reaching City Alpha.")
    assert facts[0]["actor"]["actor_text"] == "Bion" and len(events) == 1
    assert events[0].actor.actor_text == "Marcus"
    assert [p.raw_text for p in events[0].place_mentions] == ["City Beta"]


@pytest.mark.parametrize("verb", ["marched", "advanced", "proceeded", "travelled"])
def test_completed_movements_keep_endpoints_and_route(verb):
    evidence, events, facts = extract(f"Marcus {verb} from City Beta to City Alpha.")
    assert facts == [] and len(events) == 1
    assert events[0].actor.actor_text == "Marcus"
    assert [p.role for p in events[0].place_mentions] == [EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION]
    assert build(evidence, events).route is not None


@pytest.mark.parametrize("text", ["Marcus discussed travelling to City Alpha.",
                                  "Marcus described the advance toward City Alpha."])
def test_non_movement_prose_gains_no_authority(text):
    _, events, facts = extract(text)
    assert events == [] and facts == []


def test_gen7_explicit_movement_explanation_remains_route_request():
    assert infer_requested_output("Trace Marcus's movement and explain the coordinates.") == "historical_route"


def test_completed_crossing_without_named_endpoint_keeps_existing_eligibility():
    _, events, facts = extract("Ariston was prevented from marching, but later crossed the river.")
    assert facts[0]["outcome"] == "PREVENTED" and len(events) == 1
    assert events[0].summary == "later crossed the river."
    assert events[0].actor.actor_status is EventActorStatus.UNKNOWN
