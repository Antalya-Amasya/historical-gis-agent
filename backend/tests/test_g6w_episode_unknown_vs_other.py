"""G6W: insufficient episode support must not become affirmative OTHER_EPISODE."""
from __future__ import annotations

from backend.app.models import Evidence, HistoricalClaim
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_event_anchor_episode,
    classify_legacy_claim_episode,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule, _filter_relations_for_query
from backend.app.routes.evidence_relevance import EvidenceRelevance, relation_admission_allowed
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g5e2_relation_admission import _event, _evidence
from backend.tests.test_g5l_movement_parity_episode_safety import CAESAR_QUERY, POMPEY_QUERY, ev
from backend.tests.test_g6r_temporal_authority import _movement_claim, _movement_event


ARISTON_CAMPAIGN_A_QUERY = "Trace Ariston route through Campaign Alpha."


def test_a_same_subject_without_episode_support_is_unknown_not_other():
    movement = "Ariston marched from Port Helios to Port Selene."
    event = _event("ariston", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {"ariston": event},
        evidence_by_id,
        (ARISTON_CAMPAIGN_A_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is not EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is False
    assert detail["admission_reason"] == "REJECT_UNKNOWN"


def test_b_explicit_temporal_contradiction_stays_other_episode():
    query = "Trace the route from Alpha City to Beta Province in 200 BCE."
    statement = "Commander marched from Alpha City to Beta Province in 100 BCE."
    claim = _movement_claim(statement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=statement, text=statement,
    )
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False
    assert detail["admission_reason"] == "REJECT_SAME_SUBJECT_OTHER_EPISODE"


def test_c_explicit_endpoint_contradiction_stays_other_episode():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Caesar marched from Corfinium to Sicily.",
        textual_basis="Caesar marched from Corfinium to Sicily.",
        source_place="Corfinium",
        destination_place="Sicily",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = ev("ev1", claim.text, author="Julius Caesar", work="Civil War")
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_d_direct_positive_episode_match_preserved():
    query = "Trace Ariston route from Port Helios to Port Selene during Campaign Alpha."
    movement = "Ariston marched from Port Helios to Port Selene during Campaign Alpha."
    event = _event("ariston", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {"ariston": event},
        evidence_by_id,
        (query,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_e_subject_only_evidence_is_unknown_not_other():
    movement = "Ariston marched from Port Helios to Port Selene."
    event = _event("ariston", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {"ariston": event},
        evidence_by_id,
        (ARISTON_CAMPAIGN_A_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert episode is not EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_f_different_subject_stays_other_campaign():
    movement = "Commander Beta crossed from Port One into Port Two."
    event = _event("beta", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation("Port One", "Port Two", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("beta",))
    contexts = ("Trace Commander Alpha campaign route",)
    episode, detail = classify_event_anchor_episode(
        rel,
        {"beta": event},
        evidence_by_id,
        contexts,
        subject_relevance=EvidenceRelevance.OTHER_CAMPAIGN,
    )
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is False


def test_admission_reasons_distinguish_contradiction_from_unknown():
    unknown_movement = "Ariston marched from Port Helios to Port Selene."
    unknown_event = _event("ariston", unknown_movement, source_statements=[unknown_movement], refs=["ev1"])
    unknown_rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    _, unknown_detail = classify_event_anchor_episode(
        unknown_rel,
        {"ariston": unknown_event},
        {"ev1": _evidence("ev1", unknown_movement)},
        (ARISTON_CAMPAIGN_A_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )

    contradiction_query = "Trace the route from Alpha City to Beta Province in 200 BCE."
    contradiction_statement = "Commander marched from Alpha City to Beta Province in 100 BCE."
    contradiction_claim = _movement_claim(contradiction_statement)
    contradiction_evidence = Evidence(
        id="ev1",
        author="Source",
        work="Work",
        locator="1",
        excerpt=contradiction_statement,
        text=contradiction_statement,
    )
    _, contradiction_detail = classify_legacy_claim_episode(
        contradiction_claim,
        {contradiction_evidence.id: contradiction_evidence},
        (contradiction_query,),
    )

    assert unknown_detail["admission_reason"] == "REJECT_UNKNOWN"
    assert contradiction_detail["admission_reason"] == "REJECT_SAME_SUBJECT_OTHER_EPISODE"
    assert unknown_detail["admission_reason"] != contradiction_detail["admission_reason"]


def test_route_level_unresolved_episode_does_not_fabricate_other_episode():
    movement = "Ariston marched from Port Helios to Port Selene."
    event = _event("ariston", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    kept, rejected = _filter_relations_for_query(
        [rel],
        {"ariston": event},
        evidence_by_id,
        (ARISTON_CAMPAIGN_A_QUERY,),
    )
    assert kept == []
    assert rejected
    assert rejected[0]["reason"] == "EPISODE_RELEVANCE_REJECTED"
    episode, detail = classify_event_anchor_episode(
        rel,
        {"ariston": event},
        evidence_by_id,
        (ARISTON_CAMPAIGN_A_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert episode is not EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert not relation_admission_allowed(
        rel,
        {"ariston": event},
        evidence_by_id,
        (ARISTON_CAMPAIGN_A_QUERY,),
        rule=rel.rule,
    )
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        list(evidence_by_id.values()),
        event_id="ariston-route",
        name="Ariston",
        period="unknown",
        query_contexts=(ARISTON_CAMPAIGN_A_QUERY,),
    )
    assert outcome.route is None
