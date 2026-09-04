"""G5N: event-anchor episode relevance gate for route admission."""
from __future__ import annotations

from backend.app.models import (
    Evidence,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalClaim,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_event_anchor_episode,
    classify_legacy_claim_episode,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule, _filter_relations_for_query
from backend.app.routes.evidence_relevance import EvidenceRelevance, relation_admission_allowed
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g5e2_relation_admission import _event, _evidence, _place
from backend.tests.test_g5l_movement_parity_episode_safety import CAESAR_QUERY, POMPEY_QUERY, ev

XENOPHON_QUERY = (
    "Trace the route of Xenophon and the Ten Thousand from Cunaxa back to the "
    "Greek world after the Battle of Cunaxa in 401 BCE."
)


def _xenophon_athens_peloponnesus_event() -> tuple[HistoricalEvent, dict[str, Evidence]]:
    movement = "It was therefore a very hazardous journey to travel by land from Athens to Peloponnesus."
    chunk = (
        "Xenophon and the army deliberated after Cunaxa in 401 BCE. "
        + movement
        + " Commander Z later fought elsewhere."
    )
    bindings = [
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(raw_text="Athens", role=EventPlaceRole.ORIGIN, evidence_refs=["ev1"]),
            place=_place("Athenae"),
            role=EventPlaceRole.ORIGIN,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(
                raw_text="Peloponnesus", role=EventPlaceRole.DESTINATION, evidence_refs=["ev1"],
            ),
            place=_place("Peloponnesus/Peloponnesos/Peloponnese"),
            role=EventPlaceRole.DESTINATION,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
    ]
    event = _event("xen", movement, source_statements=[movement], refs=["ev1"], bindings=bindings)
    evidence_by_id = {"ev1": _evidence("ev1", chunk)}
    return event, evidence_by_id


def test_pompey_query_rejects_xenophon_athenae_peloponnesus_same_movement():
    event, evidence_by_id = _xenophon_athens_peloponnesus_event()
    rel = relation(
        "Athenae",
        "Peloponnesus/Peloponnesos/Peloponnese",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("xen",),
    )
    kept, rejected = _filter_relations_for_query(
        [rel], {"xen": event}, evidence_by_id, (POMPEY_QUERY,),
    )
    assert kept == []
    assert rejected
    assert rejected[0]["reason"] == "EPISODE_RELEVANCE_REJECTED"
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        list(evidence_by_id.values()),
        event_id="pompey-48",
        name="Pompey flight",
        period="48 BCE",
        query_contexts=(POMPEY_QUERY,),
    )
    assert outcome.route is None or not any(
        point.historical_place.canonical_name == "Athenae"
        for point in (outcome.route.ordered_points if outcome.route else [])
    )


def test_xenophon_query_admits_athenae_peloponnesus_same_movement():
    event, evidence_by_id = _xenophon_athens_peloponnesus_event()
    rel = relation(
        "Athenae",
        "Peloponnesus/Peloponnesos/Peloponnese",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("xen",),
    )
    kept, rejected = _filter_relations_for_query(
        [rel], {"xen": event}, evidence_by_id, (XENOPHON_QUERY,),
    )
    assert kept == [rel]
    assert rejected == []
    episode, detail = classify_event_anchor_episode(
        rel,
        {"xen": event},
        evidence_by_id,
        (XENOPHON_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_same_subject_wrong_episode_same_movement_rejected():
    movement = "Commander Alpha marched from Beta Province to Gamma Region during the earlier war."
    chunk = (
        "Commander Alpha fought in the earlier war. "
        + movement
        + " Later he campaigned near Delta Bay in the current conflict."
    )
    event = _event("alpha-e2", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", chunk)}
    rel = relation("Beta Province", "Gamma Region", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("alpha-e2",))
    contexts = ("Trace Commander Alpha route through Delta Bay in the current conflict",)
    assert not relation_admission_allowed(
        rel, {"alpha-e2": event}, evidence_by_id, contexts, rule=rel.rule,
    )
    episode, detail = classify_event_anchor_episode(
        rel, {"alpha-e2": event}, evidence_by_id, contexts,
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_other_subject_same_movement_rejected():
    movement = "Commander Beta crossed from Port One into Port Two."
    event = _event("beta", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation("Port One", "Port Two", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("beta",))
    contexts = ("Trace Commander Alpha campaign route",)
    assert not relation_admission_allowed(rel, {"beta": event}, evidence_by_id, contexts, rule=rel.rule)
    episode, _ = classify_event_anchor_episode(
        rel, {"beta": event}, evidence_by_id, contexts,
        subject_relevance=EvidenceRelevance.OTHER_CAMPAIGN,
    )
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN


def test_current_episode_same_movement_admitted():
    movement = "Commander Alpha crossed the Adriatic and landed in Epirus near Apollonia."
    event = _event("alpha-e1", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation("Epirus", "Apollonia", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("alpha-e1",))
    contexts = (
        "Trace Commander Alpha route from Italy across the Adriatic into Epirus "
        "and through the campaign leading to Pharsalus in 48 BCE.",
    )
    assert relation_admission_allowed(
        rel, {"alpha-e1": event}, evidence_by_id, contexts, rule=rel.rule,
    )
    episode, detail = classify_event_anchor_episode(
        rel, {"alpha-e1": event}, evidence_by_id, contexts,
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_unknown_episode_does_not_fabricate_conflict():
    movement = "From there they marched toward Beta Province."
    event = _event("sparse", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation("Alpha", "Beta Province", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("sparse",))
    contexts = ("Trace Commander Alpha retreat",)
    episode, detail = classify_event_anchor_episode(
        rel, {"sparse": event}, evidence_by_id, contexts,
        subject_relevance=EvidenceRelevance.UNKNOWN,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is True


def test_caesar_legacy_hispania_italia_still_blocked():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text=(
            "He did not think that Caesar had yet arrived in Italy from Spain, "
            "and even if he were there he did not suspect that his rival would cross the Ionian sea."
        ),
        textual_basis=(
            "He did not think that Caesar had yet arrived in Italy from Spain, "
            "and even if he were there he did not suspect that his rival would cross the Ionian sea."
        ),
        source_place="Hispania",
        destination_place="Italia",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Plutarch", work="Lives")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False
