"""Strict requested-route membership is separate from movement occurrence."""
import pytest

from backend.app.models import EventActorStatus, HistoricalTravelMode
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.query_route_admission import (
    AuthorityState, classify_observation_relation_admission, parse_query_route_scope,
)
from backend.app.routes.soft_phase_membership import build_soft_phase_membership_index
from backend.tests.test_agent_loop import ev
from backend.tests.test_v2gen8_non_completion_authority import SyntheticPlaces
from backend.tests.test_v2gen12_query_scope import PICTOR_QUERY
from backend.tests.test_v2gen7_movement_terminalization import LIBO_QUERY

class CanonicalPlaces(SyntheticPlaces):
    def call(self, tool, arguments):
        result = super().call(tool, arguments)
        if arguments["name"] == "Oricum":
            result["canonical_name"] = "Orikon"
        return result


QUERY = "Trace Marcus's route from Rome to Capua."


def build(texts, query=QUERY):
    evidence = [ev(f"strict-{i}", text) for i, text in enumerate(texts)]
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract(evidence)
    events, _ = HistoricalEventPlaceResolver(CanonicalPlaces()).resolve(events)
    result = EventAnchorRouteBuilder().build_with_diagnostics(
        events, evidence, event_id="strict", name="strict", period="unspecified",
        query_contexts=(query,),
    )
    obs = {o.observation_id: o for o in result.observations}
    event_map = {e.id: e for e in events}
    evidence_map = {e.id: e for e in evidence}
    index = build_soft_phase_membership_index(
        list(result.observation_relations), obs, event_map, evidence_map, (query,),
    )
    admissions = [classify_observation_relation_admission(
        r, obs, event_map, evidence_map, (query,), soft_phase_membership_index=index,
    ) for r in result.observation_relations]
    return result, events, admissions


def labels(result):
    return [p.historical_place.canonical_name for p in result.route.ordered_points] if result.route else []


def dated(edges, actor="Marcus"):
    # Dates belong to evidence, independently of retrieval/list order.
    return [f"In {80-i} BCE, {actor} marched from {a} to {b}." for i, (a, b) in enumerate(edges)]


@pytest.mark.parametrize("origin,destination,phase", [
    ("Capua", "Rome", AuthorityState.WRONG),
    ("Capua", "Brundisium", AuthorityState.WRONG),
    ("Brundisium", "Rome", AuthorityState.WRONG),
    ("Tarentum", "Brundisium", AuthorityState.UNKNOWN),
    ("Rome", "Brundisium", AuthorityState.UNKNOWN),
    ("Brundisium", "Capua", AuthorityState.UNKNOWN),
])
def test_isolated_incompatible_movement_cannot_be_a_partial_requested_route(origin, destination, phase):
    result, events, admissions = build([f"Marcus marched from {origin} to {destination}."])
    assert events[0].actor.actor_status is EventActorStatus.EXPLICIT
    assert admissions and admissions[0].subject_match is AuthorityState.MATCH
    assert admissions[0].movement_assertion is AuthorityState.MATCH
    assert admissions[0].route_phase_match is phase and not admissions[0].admitted
    assert result.observation_relations[0].authority == "SAME_MOVEMENT_EVENT"
    assert result.route is None and not result.observation_components
    assert result.diagnostics["rejected_observation_component_edges"]
    scope = parse_query_route_scope((QUERY,))
    assert (scope.origin, scope.destination, scope.endpoint_strict) == ("Rome", "Capua", True)


def test_direct_route_and_resolved_rome_alias_remain_admitted():
    result, _, admissions = build(["Marcus marched from Rome to Capua."])
    assert labels(result) == ["Roma", "Capua"]
    assert admissions[0].route_phase_match is AuthorityState.MATCH and admissions[0].admitted
    # Existing Rome/Roma completeness debt is allowed, not fixed here.
    assert result.diagnostics["canonical_completeness"] == "PARTIAL"


@pytest.mark.parametrize("edges,expected", [
    ([("Rome", "Beneventum"), ("Beneventum", "Capua")], ["Roma", "Beneventum", "Capua"]),
    ([("Rome", "Beneventum"), ("Beneventum", "Venusia"), ("Venusia", "Capua")],
     ["Roma", "Beneventum", "Venusia", "Capua"]),
])
def test_supported_intermediate_chains(edges, expected):
    result, _, _ = build(dated(edges))
    assert labels(result) == expected
    assert len(result.observation_components) == 1


def test_evidence_dates_not_list_order_establish_chain():
    texts = dated([("Rome", "Beneventum"), ("Beneventum", "Capua")])
    result, _, _ = build(list(reversed(texts)))
    assert labels(result) == ["Roma", "Beneventum", "Capua"]


@pytest.mark.parametrize("edges", [
    [("Rome", "Beneventum"), ("Tarentum", "Capua")],
    [("Rome", "Beneventum"), ("Venusia", "Capua"), ("Tarentum", "Brundisium")],
])
def test_chronology_does_not_join_disconnected_movements(edges):
    result, _, _ = build(dated(edges))
    assert result.route is None and not result.observation_components


def test_unrelated_extra_movement_is_excluded():
    result, _, _ = build(dated([
        ("Rome", "Beneventum"), ("Beneventum", "Capua"), ("Tarentum", "Brundisium"),
    ]))
    assert labels(result) == ["Roma", "Beneventum", "Capua"]
    assert len(result.observation_components) == 1
    assert result.diagnostics["rejected_observation_component_edges"]


def test_movements_before_origin_and_after_destination_are_excluded():
    result, _, _ = build(dated([
        ("Tarentum", "Rome"), ("Rome", "Beneventum"), ("Beneventum", "Capua"), ("Capua", "Brundisium"),
    ]))
    assert labels(result) == ["Roma", "Beneventum", "Capua"]


def test_non_strict_movement_query_keeps_broader_routes():
    result, _, _ = build(dated([("Rome", "Capua"), ("Capua", "Brundisium")]), "Trace Marcus's movements.")
    assert labels(result) == ["Roma", "Capua", "Brundisium"]


def test_gen8_endpoint_match_cannot_override_non_completion():
    result, _, _ = build(["Marcus advanced from Rome toward Capua but stopped before reaching it."])
    assert result.route is None


@pytest.mark.parametrize("query,actor,allowed", [
    ("Show Caesar route from Rome to Capua.", "Pompey", False),
    ("Show Aurelius Nestor route from Rome to Capua.", "Pompey", False),
    ("Show Aurelius Nestor route from Rome to Capua.", "Aurelius Nestor", True),
    ("Show Aurelius Nestor route from Rome to Capua.", "He", False),
])
def test_requested_actor_authority_remains(query, actor, allowed):
    result, events, admissions = build([f"{actor} marched from Rome to Capua."], query)
    assert bool(result.route) is allowed
    if actor == "He":
        assert all(e.actor.actor_text != "Aurelius Nestor" for e in events)
    else:
        assert admissions[0].subject_match is (AuthorityState.MATCH if allowed else AuthorityState.WRONG)


@pytest.mark.parametrize("query,text,expected,mode", [
    (PICTOR_QUERY,
     "While these things were carrying on, Quintus Fabius Pictor, the ambassador, returned from Delphi to Rome, and read the response of the oracle from a written copy.",
     ["Delphi", "Roma"], HistoricalTravelMode.UNKNOWN),
    (LIBO_QUERY,
     "Libo having sailed from Oricum, with a fleet of fifty ships, which he commanded, came to Brundisium, and seized an island, which lies opposite to the harbour; judging it better to guard that place, which was our only pass to sea, than to keep all the shores and ports blocked up by a fleet.",
     ["Orikon", "Brundisium"], HistoricalTravelMode.SEA),
])
def test_pictor_and_libo_evidence_controls(query, text, expected, mode):
    result, _, _ = build([text], query)
    assert labels(result) == expected
    assert result.route.claims[0].travel_mode is mode


FRESH_QUERY = "Trace Dorieus Melanthos's route from Port Alpha to City Delta."
@pytest.mark.parametrize("edges,expected", [
    ([("Port Alpha", "City Delta")], ["Port Alpha", "City Delta"]),
    ([("Port Alpha", "Town Beta"), ("Town Beta", "City Delta")], ["Port Alpha", "Town Beta", "City Delta"]),
    ([("Port Alpha", "Town Beta"), ("Town Beta", "Hill Gamma"), ("Hill Gamma", "City Delta")],
     ["Port Alpha", "Town Beta", "Hill Gamma", "City Delta"]),
    ([("City Delta", "Town Beta")], []),
    ([("Town Beta", "Port Alpha")], []),
    ([("Hill Gamma", "Fort Epsilon")], []),
    ([("Port Alpha", "Town Beta"), ("Hill Gamma", "City Delta")], []),
    ([("Port Alpha", "Town Beta"), ("Town Beta", "City Delta"), ("Hill Gamma", "Fort Epsilon")],
     ["Port Alpha", "Town Beta", "City Delta"]),
])
def test_fresh_topology(edges, expected):
    result, _, _ = build(dated(edges, "Dorieus Melanthos"), FRESH_QUERY)
    assert labels(result) == expected


def test_location_overlap_without_ordering_cannot_prove_chain():
    result, _, _ = build([
        "Marcus marched from Rome to Beneventum.", "Marcus marched from Beneventum to Capua.",
    ])
    assert result.route is None


def test_wrong_actor_cannot_complete_requested_subject_chain():
    result, _, _ = build([
        "In 80 BCE, Marcus marched from Rome to Beneventum.",
        "In 79 BCE, Pompey marched from Beneventum to Capua.",
    ])
    assert result.route is None


def test_different_campaign_cannot_complete_chain():
    result, _, _ = build([
        "In 80 BCE during Campaign Red, Marcus marched from Rome to Beneventum.",
        "In 79 BCE during Campaign Blue, Marcus marched from Beneventum to Capua.",
    ])
    assert result.route is None


def test_strict_ordering_pair_preserves_claim_without_out_of_scope_auxiliary_legs():
    from backend.tests.test_g6ds_route_assembly_acceptance import POSITIVE_FIRST, POSITIVE_SECOND, QUERY as PAIR_QUERY, _pair
    e1, e2, evidence_map = _pair(POSITIVE_FIRST, POSITIVE_SECOND)
    result = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_map.values()), event_id="pair", name="pair", period="200 BCE",
        query_contexts=PAIR_QUERY,
    )
    assert labels(result) == ["Capua", "Brundisium"]
    claim = result.route.claims[0]
    assert claim.claim_type == "WAYPOINT_ORDERING"
    assert claim.movement_relation is None
    assert "no direct movement is asserted" in claim.text


def test_other_same_role_binding_cannot_supply_observation_endpoint_identity():
    from backend.app.models import EventPlaceRole
    from backend.app.routes.query_route_admission import _classify_route_phase_match
    from backend.app.routes.route_observations import ObservationOrderingAuthority
    from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import _event, _binding, _observation, _relation
    event = _event("m", "Dorieus Melanthos marched from Port Alpha to Town Beta before reaching City Delta.",
                   origin="Port Alpha", destination="Town Beta", evidence_id="ev")
    event = event.model_copy(update={"place_bindings": [*event.place_bindings, _binding("City Delta", EventPlaceRole.DESTINATION, eid="ev")]})
    origin = _observation("a", label="Port Alpha", event_id="m", evidence_id="ev", place_role=EventPlaceRole.ORIGIN)
    unrelated = _observation("x", label="Town Beta", event_id="m", evidence_id="ev", place_role=EventPlaceRole.DESTINATION)
    relation = _relation("a", "x", event_ids=("m",), evidence_refs=("ev",), authority=ObservationOrderingAuthority.AFTER_SUBORDINATE)
    phase = _classify_route_phase_match("Port Alpha", "Town Beta", parse_query_route_scope((FRESH_QUERY,)), event.summary,
        relation=relation, observations_by_id={"a": origin, "x": unrelated}, events_by_id={"m": event})
    assert phase is AuthorityState.UNKNOWN


def test_soft_membership_proof_cannot_be_reused_as_strict_membership():
    from backend.app.routes.soft_phase_membership import SoftPhaseMembershipIndex, SoftPhaseAnchorType
    from backend.app.routes.route_observations import ObservationOrderingRelation, ObservationOrderingAuthority
    soft = parse_query_route_scope(("Trace Marcus's route from Rome toward Capua.",))
    index = SoftPhaseMembershipIndex._from_builder(soft, {("a", "b"): SoftPhaseAnchorType.COMPLETE})
    relation = ObservationOrderingRelation("a", "b", ObservationOrderingAuthority.AFTER_SUBORDINATE, ("m",), ("ev",), "SAME_MOVEMENT_EVENT")
    assert index.proof_for(relation, parse_query_route_scope((QUERY,))) is None


@pytest.mark.parametrize("extra", [False, True])
def test_typed_same_event_intermediate_chain_respects_strict_boundaries(extra):
    from backend.app.models import (EventPlaceRole, EventRouteOrdering, EventRouteOrderingRef,
                                    EventRouteOrderingEndpointKind, EventRouteOrderingAuthority)
    from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import _event, _binding
    from backend.tests.test_g6dq_explicit_connector_source_chronology import explicit_actor
    text = "Dorieus Melanthos first marched from Port Alpha to Town Beta, then marched from Town Beta to City Delta."
    if extra:
        text += " Then Dorieus Melanthos marched from City Delta to Fort Epsilon."
    evidence = ev("typed", text)
    event = _event("typed", text, origin="Port Alpha", destination="Town Beta", evidence_id=evidence.id)
    bindings = [*event.place_bindings, _binding("City Delta", EventPlaceRole.DESTINATION, eid=evidence.id)]
    path = ["Port Alpha", "Town Beta", "City Delta"]
    if extra:
        path.append("Fort Epsilon")
        bindings.append(_binding("Fort Epsilon", EventPlaceRole.DESTINATION, eid=evidence.id))
    refs = [EventRouteOrderingRef(endpoint_kind=(EventRouteOrderingEndpointKind.ORIGIN if i == 0
            else EventRouteOrderingEndpointKind.DESTINATION), surface=name, canonical=name)
            for i, name in enumerate(path)]
    orderings = [EventRouteOrdering(earlier=a, later=b, authority=EventRouteOrderingAuthority.FIRST_THEN,
                 source_statement=text, evidence_refs=[evidence.id]) for a, b in zip(refs, refs[1:])]
    event = event.model_copy(update={"actor": explicit_actor("Dorieus Melanthos"),
                                    "place_bindings": bindings, "route_orderings": orderings})
    result = EventAnchorRouteBuilder().build_with_diagnostics([event], [evidence], event_id="typed", name="typed",
        period="unspecified", query_contexts=(FRESH_QUERY,))
    assert labels(result) == ["Port Alpha", "Town Beta", "City Delta"]
    assert len(result.observation_components) == 1
    if extra:
        assert result.diagnostics["rejected_observation_component_edges"]
