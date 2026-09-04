"""G5N-5: event-anchor episode classifier semantics."""
from __future__ import annotations

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
)
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_event_anchor_episode,
    classify_legacy_claim_episode,
)
from backend.app.routes.event_route_orchestration import OrderingRule
from backend.app.routes.evidence_relevance import EvidenceRelevance, relation_admission_allowed
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g5e2_relation_admission import _event, _evidence, _place
from backend.tests.test_g5l_movement_parity_episode_safety import CAESAR_QUERY, POMPEY_QUERY, ev
from backend.tests.test_g5n_event_anchor_episode_gate import (
    XENOPHON_QUERY,
    _xenophon_athens_peloponnesus_event,
)


def _theseus_athens_peloponnesus_live_shape() -> tuple:
    movement = "It was therefore a very hazardous journey to travel by land from Athens to Peloponnesus;"
    chunk = (
        "Some of these, Hercules destroyed and cut off in his passage through these countries, but some, "
        "escaping his notice while he was passing by, fled and hid themselves, or else were spared by him "
        "in contempt of their abject submission; and after that Hercules fell into misfortune, and, having "
        "slain Iphitus, retired to Lydia, and for a long time was there slave to Omphale, a punishment "
        "which he had imposed upon himself for the murder, then, indeed, Lydia enjoyed high peace and "
        "security, but in Greece things were brought back into confusion. "
        + movement
        + " and Pittheus, giving him an exact account of each of these robbers and villains, tried to "
        "persuade Theseus to go by sea."
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
    event = _event("theseus", movement, source_statements=[movement], refs=["ev1"], bindings=bindings)
    evidence_by_id = {"ev1": _evidence("ev1", chunk)}
    rel = relation(
        "Athenae",
        "Peloponnesus/Peloponnesos/Peloponnese",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("theseus",),
    )
    return event, evidence_by_id, rel


def test_pompey_rejects_theseus_athenae_peloponnesus_live_shape():
    event, evidence_by_id, rel = _theseus_athens_peloponnesus_live_shape()
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        evidence_by_id,
        (POMPEY_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False
    assert not relation_admission_allowed(
        rel, {event.id: event}, evidence_by_id, (POMPEY_QUERY,), rule=rel.rule,
    )


def test_xenophon_admits_athenae_peloponnesus():
    event, evidence_by_id = _xenophon_athens_peloponnesus_event()
    rel = relation(
        "Athenae",
        "Peloponnesus/Peloponnesos/Peloponnese",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("xen",),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {"xen": event},
        evidence_by_id,
        (XENOPHON_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_weak_one_anchor_overlap_not_direct():
    movement = "Commander Alpha marched from Gamma Coast into Beta Province."
    chunk = (
        "Commander Alpha fought in the earlier war. "
        + movement
        + " Later he campaigned near Beta Province during Operation Summit."
    )
    event = _event("alpha", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", chunk)}
    rel = relation("Gamma Coast", "Beta Province", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("alpha",))
    contexts = ("Trace Commander Alpha route through Beta Province during Operation Summit",)
    episode, detail = classify_event_anchor_episode(
        rel, {"alpha": event}, evidence_by_id, contexts,
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is not EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is False


def test_strong_episode_alignment_is_direct():
    movement = "Commander Alpha crossed from Gamma Coast into Beta Province during Operation Summit."
    event = _event("alpha", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation("Gamma Coast", "Beta Province", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("alpha",))
    contexts = ("Trace Commander Alpha route through Beta Province and Operation Summit",)
    episode, detail = classify_event_anchor_episode(
        rel, {"alpha": event}, evidence_by_id, contexts,
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_unknown_stays_unknown():
    movement = "From there they marched toward Beta Province."
    event = _event("sparse", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation("Alpha", "Beta Province", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("sparse",))
    episode, detail = classify_event_anchor_episode(
        rel, {"sparse": event}, evidence_by_id, ("Trace Commander Alpha retreat",),
        subject_relevance=EvidenceRelevance.UNKNOWN,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is True


def test_chunk_noise_cannot_override_local_statement():
    event, evidence_by_id, rel = _theseus_athens_peloponnesus_live_shape()
    episode, _ = classify_event_anchor_episode(
        rel,
        {event.id: event},
        evidence_by_id,
        (POMPEY_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE


def test_caesar_legacy_controls_still_hold():
    from backend.app.models import HistoricalClaim

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
