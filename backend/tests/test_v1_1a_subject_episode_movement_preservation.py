"""V1.1A: preserve bounded person+episode+movement evidence in coverage merge."""
from __future__ import annotations

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    _qualifies_subject_episode_movement,
    merge_coverage_results,
)
import pytest
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent

QUERY = (
    "Trace Commander Alpha's route after the Battle of Gamma Harbor to Delta Bay."
)
QUALIFIED_ID = "qualified:0:120"
QUALIFIED_TEXT = (
    "Commander Alpha sailed from Gamma Harbor toward Delta Bay after the battle."
)


def _item(
    identifier: str,
    text: str,
    *,
    family: str,
    score: float = 0.5,
    lexical_score: float | None = None,
) -> Evidence:
    parts = identifier.split(":")
    start = int(parts[1]) if len(parts) > 1 else 0
    end = int(parts[2]) if len(parts) > 2 else start + 80
    metadata = {
        "parent_id": family,
        "source_chunk_id": family,
        "document_id": family,
        "passage_start": start,
        "passage_end": end,
        "passage_index": 0,
        "semantic_candidate": True,
        "vector_rank": start + 1,
    }
    if lexical_score is not None:
        metadata["lexical_candidate"] = True
        metadata["lexical_score"] = lexical_score
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="1",
        excerpt=text[:500],
        text=text,
        score=score,
        metadata=metadata,
    )


def _ranked(items: list[Evidence]) -> list[Evidence]:
    return rerank_evidence(QUERY, items, pool_relative=False)


MOVEMENT_CHANNEL_QUERY = "route journey travel by land march"
SUBJECT_CHANNEL_QUERY = "Commander Alpha"
EPISODE_CHANNEL_QUERY = "Commander Alpha battle Gamma Harbor Delta Bay"


def _merge_channels(*channel_items: tuple[str, str, list[Evidence]]) -> list[Evidence]:
    intent_results = [
        (RetrievalIntent(kind, query), _ranked(items))
        for kind, query, items in channel_items
    ]
    return merge_coverage_results(QUERY, intent_results, budget=DEFAULT_COVERAGE_BUDGET)


def _stress_merge(qualified: Evidence) -> list[Evidence]:
    decoys = [
        _item(
            f"decoy-{index}:{index * 100}:{index * 100 + 80}",
            f"Commander Alpha marched from Port Helios to Port Selene by land in segment {index}.",
            family=f"decoy-{index % 8}",
            score=0.99 - index * 0.001,
            lexical_score=50.0,
        )
        for index in range(24)
    ]
    movement_generics = [
        _item(
            f"move-{index}:{index * 100}:{index * 100 + 80}",
            f"The route required a long march from Port {index} to Port {index + 1} by land.",
            family=f"move-{index % 5}",
            score=0.95 - index * 0.005,
            lexical_score=40.0,
        )
        for index in range(8)
    ]
    return _merge_channels(
        ("CANONICAL", QUERY, [*decoys, qualified]),
        ("MOVEMENT", MOVEMENT_CHANNEL_QUERY, movement_generics),
    )


def _generic_flood(count: int = 22) -> list[Evidence]:
    return [
        _item(
            f"generic-{index}:{index * 100}:{index * 100 + 80}",
            f"The route required a long march from Port {index} to Port {index + 1} by land.",
            family=f"generic-{index % 6}",
            score=0.95 - index * 0.01,
        )
        for index in range(count)
    ]


def _preservation_ids(items: list[Evidence]) -> set[str]:
    return {
        item.id
        for item in items
        if (item.metadata.get("retrieval_provenance") or {}).get("final_merge_reason")
        == "subject_episode_movement"
    }


def test_qualified_subject_episode_movement_survives_final_selection():
    qualified = _item(QUALIFIED_ID, QUALIFIED_TEXT, family="qualified-family", score=0.2)
    merged = _stress_merge(qualified)
    assert QUALIFIED_ID in {item.id for item in merged}


def test_person_only_without_movement_not_preserved():
    person_only = _item(
        "person:0:80",
        "Commander Alpha remained in camp while the council debated supplies.",
        family="person-family",
        score=0.9,
    )
    merged = _merge_channels(("CANONICAL", QUERY, [person_only, *_generic_flood(count=12)]))
    assert person_only.id not in _preservation_ids(merged)


def test_movement_only_without_person_or_episode_not_preserved():
    movement_only = _item(
        "movement:0:80",
        "The army marched from Port Helios to Port Selene by land.",
        family="movement-family",
        score=0.9,
    )
    merged = _merge_channels(("CANONICAL", QUERY, [movement_only, *_generic_flood(count=12)]))
    assert movement_only.id not in _preservation_ids(merged)


def test_endpoint_overlap_only_not_preserved():
    endpoint_only = _item(
        "endpoint:0:80",
        "Gamma Harbor lies near Delta Bay along the coast.",
        family="endpoint-family",
        score=0.9,
    )
    merged = _merge_channels(("CANONICAL", QUERY, [endpoint_only, *_generic_flood(count=12)]))
    assert endpoint_only.id not in _preservation_ids(merged)


def test_wrong_explicit_person_not_preserved():
    wrong_person = _item(
        "wrong:0:80",
        "Commander Beta sailed from Gamma Harbor to Delta Bay after the battle.",
        family="wrong-family",
        score=0.9,
    )
    merged = _merge_channels(("CANONICAL", QUERY, [wrong_person, *_generic_flood(count=12)]))
    assert wrong_person.id not in _preservation_ids(merged)


def test_duplicate_parent_does_not_consume_all_preservation_slots():
    sibling_a = _item(
        "dup-parent:0:80",
        "Commander Alpha sailed from Gamma Harbor toward Delta Bay after the battle.",
        family="dup-parent",
        score=0.7,
    )
    sibling_b = _item(
        "dup-parent:120:200",
        "Commander Alpha landed near Delta Bay after the battle.",
        family="dup-parent",
        score=0.65,
    )
    other = _item(
        "other:0:80",
        "Commander Alpha fled from Gamma Harbor into Delta Bay after the battle.",
        family="other-parent",
        score=0.6,
    )
    merged = _merge_channels(
        ("CANONICAL", QUERY, [sibling_a, sibling_b, other, *_generic_flood(count=12)]),
    )
    preserved = [
        item
        for item in merged
        if (item.metadata.get("retrieval_provenance") or {}).get("final_merge_reason")
        == "subject_episode_movement"
    ]
    assert len(preserved) <= 2
    families = {
        item.metadata.get("parent_id") or item.metadata.get("source_chunk_id")
        for item in preserved
    }
    assert len(families) == len(preserved)


def test_budget_remains_twenty():
    qualified = _item(QUALIFIED_ID, QUALIFIED_TEXT, family="qualified-family", score=0.2)
    merged = _stress_merge(qualified)
    assert len(merged) == DEFAULT_COVERAGE_BUDGET


CONTROL_QUERY = "Trace Commander Alpha's route after the Battle of Zeta toward Delta Bay."


def _qualifies(text: str) -> bool:
    item = _item("control:0:200", text, family="control")
    return _qualifies_subject_episode_movement(CONTROL_QUERY, rerank_evidence(CONTROL_QUERY, [item], pool_relative=False)[0])


def test_explicit_local_subject_episode_movement_qualifies():
    assert _qualifies("Commander Alpha sailed from Gamma Harbor to Delta Bay after the Battle of Zeta.")


@pytest.mark.parametrize("text", [
    "Commander Alpha did not sail from Gamma Harbor to Delta Bay after the Battle of Zeta.",
    "Commander Alpha would have sailed from Gamma Harbor to Delta Bay after the Battle of Zeta.",
    "It was said that Commander Alpha sailed from Gamma Harbor to Delta Bay after the Battle of Zeta.",
    "Commander Beta sailed from Gamma Harbor to Delta Bay while Commander Alpha remained behind after the Battle of Zeta.",
    "Commander Alpha defeated Commander Beta. He sailed from Gamma Harbor to Delta Bay after the Battle of Zeta.",
    "Commander Alpha sailed from Gamma Harbor to Delta Bay in a different campaign.",
    "Commander Alpha remained at Gamma Harbor after the Battle of Zeta.",
    "The army sailed from Gamma Harbor to Delta Bay after the Battle of Zeta.",
    "Gamma Harbor and Delta Bay were important ports after the Battle of Zeta.",
    "Commander Alpha policy steered the debate toward reform after the Battle of Zeta.",
])
def test_unsafe_or_incomplete_candidate_is_not_preserved(text: str):
    assert not _qualifies(text)


def test_cross_parent_duplicate_prefix_uses_one_preservation_slot():
    text = "Commander Alpha sailed from Gamma Harbor toward Delta Bay after the battle."
    merged = _merge_channels(("CANONICAL", QUERY, [
        _item("copy-a:0:100", text, family="copy-a"),
        _item("copy-b:0:100", text, family="copy-b"),
        *_generic_flood(),
    ]))
    assert len(_preservation_ids(merged) & {"copy-a:0:100", "copy-b:0:100"}) == 1


LUCULLUS_QUERY = "Trace Lucullus's route from Ephesus through Pontus toward Tigranocerta."
LUCULLUS_WAR_QUERY = "Trace Lucullus during the war against Mithridates."
POMPEY_QUERY = "Trace Pompey's route after the Battle of Pharsalus to Egypt."


def _qualifies_query(query: str, text: str) -> bool:
    item = _item("control:0:200", text, family="control")
    return _qualifies_subject_episode_movement(query, rerank_evidence(query, [item], pool_relative=False)[0])


def test_wrong_query_person_mithridates_not_preserved_for_lucullus():
    text = "Mithridates sailed from Pontus to the Euxine while Lucullus remained in Asia."
    assert not _qualifies_query(LUCULLUS_QUERY, text)


def test_query_subject_is_explicit_mover_may_preserve():
    text = "Lucullus sailed from Ephesus to Chios during the campaign in Pontus."
    assert _qualifies_query(LUCULLUS_QUERY, text)


def test_opponent_mention_not_preserved():
    text = "Mithridates sailed from Pontus to Colchis."
    assert not _qualifies_query(LUCULLUS_WAR_QUERY, text)


@pytest.mark.parametrize("text", [
    "They say Pompey sailed from Cyprus to Egypt after Pharsalus.",
    "It was reported that Pompey sailed from Cyprus to Egypt.",
    "The course was directed toward Egypt.",
    "Pompey considered the course toward Egypt.",
])
def test_hearsay_reported_or_actorless_course_not_preserved(text: str):
    assert not _qualifies_query(POMPEY_QUERY, text)
