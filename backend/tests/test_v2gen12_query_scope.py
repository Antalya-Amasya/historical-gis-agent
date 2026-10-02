"""Route grammar remains separate from instructions and actor authority."""
import pytest
from backend.app.agent.loop import infer_requested_output, BoundedAgentLoop
from backend.app.models import EventPlaceRole, HistoricalTravelMode
from backend.app.routes.query_route_admission import parse_query_route_scope, classify_observation_relation_admission, AuthorityState
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.tests.test_agent_loop import ev
from backend.tests.test_v2gen8_non_completion_authority import SyntheticPlaces
from backend.tests.test_v2gen9_query_subject_authority import control

PICTOR_QUERY = (
 "Reconstruct Quintus Fabius Pictor's return from Delphi to Rome, but separate every part "
 "of the result into historically supported facts, representative geographic anchors, "
 "simulated road geometry, and uncertain elements. Do not treat Roman-road vertices or "
 "representative coordinates as historical waypoints. If the historical travel mode is "
 "not explicitly supported, preserve it as unknown while still constructing a "
 "geographically plausible route.")

def scope(text):
    return parse_query_route_scope((text,))

@pytest.mark.parametrize("actor,noun", [
 ("Quintus Fabius Pictor", "return"), ("Caesar", "march"), ("Pompey", "return"),
 ("Hannibal", "advance"), ("Libo", "movement"), ("Neralis Vexon", "journey"),
 ("Neralis Vexon", "travel"), ("Neralis Vexon", "voyage")])
def test_possessor_subject(actor, noun):
    text = f"Reconstruct {actor}'s {noun} from Rome to Capua."
    parsed = scope(text)
    assert (parsed.subject, parsed.origin, parsed.destination) == (actor, "Rome", "Capua")
    assert scope(text.removeprefix("Reconstruct ")).subject == actor

def test_unknown_return_and_existing_forms():
    assert scope("Reconstruct Neralis Vexon's return from Rome to Capua.").subject == "Neralis Vexon"
    assert scope("Trace Caesar's route from Rome to Capua.").subject == "Caesar"
    assert scope("Reconstruct Pompey's route from Rome to Brundisium.").subject == "Pompey"
    assert scope("Show Hannibal's movement across the Alps.").subject == "Hannibal"

@pytest.mark.parametrize("instruction", [
 "separate the result into historical facts", "divide the answer into sections",
 "group the evidence into categories", "turn the output into a table",
 "convert coordinates into GeoJSON", "split the response into historical and simulated parts",
 "convert the result through a geographic projection", "convert the result to GeoJSON"])
def test_instruction_direction(instruction):
    parsed = scope("Reconstruct Neralis Vexon's return from Rome to Capua and " + instruction + ".")
    assert (parsed.subject, parsed.origin, parsed.destination) == ("Neralis Vexon", "Rome", "Capua")

@pytest.mark.parametrize("text,origin,destination", [
 (PICTOR_QUERY, "Delphi", "Rome"),
 ("Reconstruct Caesar's route from Rome to Capua, and separate the result into historical facts and simulated geometry.", "Rome", "Capua"),
 ("Reconstruct Pompey's return from Brundisium to Rome and divide the answer into sections.", "Brundisium", "Rome"),
 ("Trace Hannibal's march from Hispania to Italy and convert the result into GeoJSON.", "Hispania", "Italy"),
 ("Reconstruct Libo's movement from Oricum to Brundisium, splitting the result into historical and simulated components.", "Oricum", "Brundisium")])
def test_instructional_endpoints(text, origin, destination):
    parsed = scope(text)
    assert (parsed.origin, parsed.destination) == (origin, destination)
    assert parsed.has_endpoint_constraint

@pytest.mark.parametrize("text", [
 "Trace Marcus from Rome to Capua and then to Beneventum.",
 "Trace Marcus from Oricum into the Adriatic and onward to Brundisium.",
 "Reconstruct Caesar's march from Rome to Capua, then his later movement to Beneventum."])
def test_real_continuation_is_conservative(text):
    parsed = scope(text)
    assert parsed.origin is None and parsed.destination is None and parsed.has_endpoint_constraint

def test_real_into_and_via():
    assert scope("Trace Marcus from Rome into Capua.").destination == "Capua"
    assert scope("Trace Marcus from Delphi through Brundisium to Rome.").destination == "Rome"
    assert scope("Trace Marcus from Delphi via Brundisium to Rome.").origin == "Delphi"
    for text, place in [("Marcus marched into Capua.", "Capua"), ("The army advanced into Gaul.", "Gaul")]:
        events, _ = EvidenceGroundedHistoricalEventExtractor().extract([ev("direction", text)])
        assert any(m.raw_text == place and m.role == EventPlaceRole.DESTINATION for e in events for m in e.place_mentions)
    for text in ["Caesar entered into the city.", "Libo sailed into the harbor."]:
        assert not scope(text).has_endpoint_constraint

def test_exact_pictor_admission():
    source = ev("return", "Quintus Fabius Pictor returned from Delphi to Rome.")
    events, diagnostics = EvidenceGroundedHistoricalEventExtractor().extract([source], query_contexts=(PICTOR_QUERY,))
    assert not diagnostics["incomplete_movement_facts"]
    events, _ = HistoricalEventPlaceResolver(SyntheticPlaces()).resolve(events)
    result = EventAnchorRouteBuilder().build_with_diagnostics(events, [source], event_id="a", name="a", period="unspecified", query_contexts=(PICTOR_QUERY,))
    assert infer_requested_output(PICTOR_QUERY) == "historical_route"
    assert scope(PICTOR_QUERY).subject == "Quintus Fabius Pictor" and result.route is not None
    admissions = [classify_observation_relation_admission(r, {o.observation_id:o for o in result.observations}, {e.id:e for e in events}, {source.id:source}, (PICTOR_QUERY,)) for r in result.observation_relations]
    assert admissions and all(a.admitted and a.subject_match is AuthorityState.MATCH and a.route_phase_match is AuthorityState.MATCH and a.movement_assertion is AuthorityState.MATCH and a.event_episode_compatibility is AuthorityState.MATCH for a in admissions)
    assert all(c.travel_mode is HistoricalTravelMode.UNKNOWN for c in result.route.claims)

def test_wrong_actor_and_aborted_authority():
    _, _, result, admissions = control("Show Caesar route from Rome to Capua.", "Pompey", "Caesar remained in camp. ")
    assert result.route is None and all(a.subject_match is AuthorityState.WRONG for a in admissions)
    events, diagnostics = EvidenceGroundedHistoricalEventExtractor().extract([ev("aborted", "Marcus advanced from City Beta toward City Alpha but stopped before reaching it.")])
    assert events == [] and diagnostics["incomplete_movement_facts"][0]["outcome"] == "ABORTED"

@pytest.mark.parametrize("text", ["Rome's road network", "Italy's geography", "the army's movement", "the consul's route"])
def test_nonperson_possessives(text):
    assert scope(text).subject is None

def test_negative_terminal():
    assert not BoundedAgentLoop._final_answer_asserts_unsupported_route("The movement was prevented, so no completed route should be reconstructed.")
    assert BoundedAgentLoop._final_answer_asserts_unsupported_route("No route is established, but it probably went from A to B.")
