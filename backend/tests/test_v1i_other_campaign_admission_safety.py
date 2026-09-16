"""V1I: OTHER_CAMPAIGN must not be retried as DIRECT_SUBJECT by default."""
from __future__ import annotations

from backend.app.routes.episode_relevance import EpisodeRelevance, classify_event_anchor_episode
from backend.app.routes.event_route_orchestration import OrderingRule
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    relation_admission_allowed,
    relation_admission_diagnostic,
    same_movement_relation_relevance,
)
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g5e2_relation_admission import _event, _evidence
from backend.tests.test_g6w_episode_unknown_vs_other import ARISTON_CAMPAIGN_A_QUERY


QUERY = "Subject Alpha Hispania campaign"
MOVEMENT = "Commander Z crossed from Alpha Port into Beta Region."


def test_other_campaign_same_movement_is_not_retried_as_direct_subject():
    event = _event("foreign", MOVEMENT, source_statements=[MOVEMENT], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", MOVEMENT)}
    rel = relation(
        "Alpha Port",
        "Beta Region",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("foreign",),
    )
    relevance = same_movement_relation_relevance(event, evidence_by_id, (QUERY,))
    assert relevance is EvidenceRelevance.OTHER_CAMPAIGN

    episode, detail = classify_event_anchor_episode(
        rel,
        {"foreign": event},
        evidence_by_id,
        (QUERY,),
        subject_relevance=relevance,
    )
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is False

    diagnostic = relation_admission_diagnostic(
        rel,
        {"foreign": event},
        evidence_by_id,
        (QUERY,),
        rule=rel.rule,
    )
    assert diagnostic["admission"] == "REJECT"
    assert not relation_admission_allowed(
        rel,
        {"foreign": event},
        evidence_by_id,
        (QUERY,),
        rule=rel.rule,
    )


def test_other_campaign_with_explicit_actor_mismatch_stays_rejected():
    event = _event(
        "foreign",
        "Commander Z crossed from Alpha Port into Beta Region.",
        source_statements=["Commander Z crossed from Alpha Port into Beta Region."],
        refs=["ev1"],
    )
    evidence_by_id = {"ev1": _evidence("ev1", event.summary)}
    rel = relation(
        "Alpha Port",
        "Beta Region",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("foreign",),
    )
    assert not relation_admission_allowed(
        rel,
        {"foreign": event},
        evidence_by_id,
        ("Subject Alpha Hispania campaign",),
        rule=rel.rule,
    )


def test_other_campaign_with_endpoint_overlap_only_stays_rejected():
    event = _event(
        "foreign",
        MOVEMENT,
        source_statements=[MOVEMENT],
        refs=["ev1"],
    )
    evidence_by_id = {"ev1": _evidence("ev1", MOVEMENT)}
    rel = relation(
        "Alpha Port",
        "Beta Region",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("foreign",),
    )
    assert same_movement_relation_relevance(event, evidence_by_id, (QUERY,)) is EvidenceRelevance.OTHER_CAMPAIGN
    assert not relation_admission_allowed(
        rel,
        {"foreign": event},
        evidence_by_id,
        (QUERY,),
        rule=rel.rule,
    )


def test_other_campaign_with_unknown_actor_stays_rejected():
    movement = "They crossed from Alpha Port into Beta Region."
    event = _event("foreign", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation(
        "Alpha Port",
        "Beta Region",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("foreign",),
    )
    assert same_movement_relation_relevance(event, evidence_by_id, (QUERY,)) is EvidenceRelevance.OTHER_CAMPAIGN
    assert not relation_admission_allowed(
        rel,
        {"foreign": event},
        evidence_by_id,
        (QUERY,),
        rule=rel.rule,
    )


def test_cross_event_unknown_campaign_control_unchanged():
    movement = "Ariston marched from Port Helios to Port Selene."
    event = _event("ariston", movement, source_statements=[movement], refs=["ev1"])
    evidence = _evidence("ev1", movement)
    rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    _, detail = classify_event_anchor_episode(
        rel,
        {"ariston": event},
        {evidence.id: evidence},
        (ARISTON_CAMPAIGN_A_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert detail["admitted"] is False
    assert not relation_admission_allowed(
        rel,
        {"ariston": event},
        {evidence.id: evidence},
        (ARISTON_CAMPAIGN_A_QUERY,),
        rule=rel.rule,
    )


def test_structural_order_unknown_behavior_unchanged():
    first = _event("e1", "Subject Alpha advanced from Alpha City toward Beta Port.", refs=["a"])
    second = _event("e2", "Commander Z crossed from Gamma Port into Delta Region.", refs=["b"])
    evidence_by_id = {
        "a": _evidence("a", first.summary),
        "b": _evidence("b", second.summary),
    }
    rel = relation(
        "Beta Port",
        "Gamma Port",
        OrderingRule.SOURCE_STRUCTURAL_ORDER,
        refs=("a", "b"),
        event_ids=("e1", "e2"),
    )
    assert not relation_admission_allowed(
        rel,
        {"e1": first, "e2": second},
        evidence_by_id,
        ("Subject Alpha campaign",),
        rule=rel.rule,
    )
