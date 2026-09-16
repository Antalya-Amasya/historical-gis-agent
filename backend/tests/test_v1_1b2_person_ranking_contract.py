"""V1.1B2.2: freeze primary/secondary person ranking contract (score-free ordering).

These tests encode the approved SEPARATE_PRIMARY_SECONDARY_SIGNALS architecture
without implementing it. Relative ordering and classification assertions only;
no numeric weight snapshots unless documenting current RED diagnostics.
"""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retrieval_intents import primary_route_subject
from backend.app.rag.query_roles import analyze_query
from backend.tests.test_v1_1b1_2_subject_authority_retrieval_context import (
    MITHRIDATES_QUERY,
    _ref,
    _trace_mithridates_ref,
)

ALPHA_QUERY = (
    "Trace Commander Alpha's campaign against Commander Beta from Gamma Harbor to Delta Bay."
)

LUCULLUS_QUERY = (
    "Trace Lucullus's campaign movements against Mithridates from Pontus through "
    "Armenia and into Asia Minor."
)
ALEXANDER_QUERY = "Trace Alexander's march against Porus toward the Hydaspes."
SULLA_QUERY = "Trace Sulla's campaign against Mithridates in Greece."
POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)
CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)

PRIMARY_MOVER = "Commander Alpha marched from Gamma Harbor to Delta Bay."
PRIMARY_CONTEXT = "Commander Alpha remained at Gamma Harbor during the campaign."
OPPONENT_ONLY = "Commander Beta marched from Gamma Harbor to Delta Bay."
PRIMARY_OPPONENT_MOVER = (
    "Commander Alpha marched from Gamma Harbor to Delta Bay while Commander Beta followed."
)
OPPONENT_MOVER_PRIMARY_CONTEXT = (
    "Commander Beta marched from Gamma Harbor to Delta Bay while Commander Alpha remained behind."
)
BOTH_NO_MOVEMENT = "Commander Alpha met Commander Beta near Gamma Harbor."
NEITHER_PERSON = "The army marched from Gamma Harbor to Delta Bay."
PRONOUN_MOVEMENT = "Commander Alpha was at Gamma Harbor. He marched to Delta Bay."
AMBIGUOUS_MULTI = "Commander Alpha and Commander Beta moved through the region."

_BOUNDED_OPPONENT_UPLIFT = 0.02


def _evidence(identifier: str, text: str, *, vector_rank: int = 5, **metadata) -> Evidence:
    base = {
        "document_id": "doc",
        "semantic_candidate": True,
        "vector_rank": vector_rank,
    }
    base.update(metadata)
    return Evidence(
        id=identifier,
        author=str(metadata.get("author", "Author")),
        work=str(metadata.get("work", "Work")),
        locator="section",
        excerpt=text,
        text=text,
        score=0.5,
        metadata=base,
    )


def _ranking(query: str, identifier: str, text: str, **metadata) -> dict:
    ranked = rerank_evidence(
        query,
        [_evidence(identifier, text, **metadata)],
        pool_relative=False,
    )
    return ranked[0].metadata["retrieval_ranking"]


def _final(query: str, identifier: str, text: str, **metadata) -> float:
    return _ranking(query, identifier, text, **metadata)["final_score"]


def _compare_pair(
    query: str,
    left_id: str,
    left_text: str,
    right_id: str,
    right_text: str,
    *,
    left_metadata: dict | None = None,
    right_metadata: dict | None = None,
) -> tuple[float, float]:
    left_metadata = left_metadata or {}
    right_metadata = right_metadata or {}
    return (
        _final(query, left_id, left_text, **left_metadata),
        _final(query, right_id, right_text, **right_metadata),
    )



# ---------------------------------------------------------------------------
# Phase 1 — harness uses production rerank_evidence without mocks
# ---------------------------------------------------------------------------


def test_harness_uses_production_rerank_path():
    ranking = _ranking(ALPHA_QUERY, "harness", PRIMARY_MOVER)
    assert "final_score" in ranking
    assert "person_support" in ranking
    assert ranking["final_score"] > 0.0


# ---------------------------------------------------------------------------
# Phase 2 — generic synthetic matrix + invariants A–F
# ---------------------------------------------------------------------------


def test_invariant_a_explicit_primary_mover_beats_primary_context():
    mover, context = _compare_pair(
        ALPHA_QUERY,
        "primary-mover",
        PRIMARY_MOVER,
        "primary-context",
        PRIMARY_CONTEXT,
    )
    assert mover > context


def test_invariant_b_primary_mover_beats_comparable_opponent_mover():
    primary_only, opponent_only = _compare_pair(
        ALPHA_QUERY,
        "primary-mover",
        PRIMARY_MOVER,
        "opponent-only",
        OPPONENT_ONLY,
    )
    assert primary_only > opponent_only


def test_invariant_c_primary_plus_opponent_must_not_auto_beat_primary_only():
    primary_only, primary_plus_opponent = _compare_pair(
        ALPHA_QUERY,
        "primary-only",
        PRIMARY_MOVER,
        "primary-plus-opponent",
        PRIMARY_OPPONENT_MOVER,
    )
    assert primary_plus_opponent <= primary_only + _BOUNDED_OPPONENT_UPLIFT


@pytest.mark.xfail(
    strict=True,
    reason="B2 RED_EXPECTED: opponent mover + primary context must not tie or beat explicit primary mover",
)
def test_invariant_d_primary_mover_beats_opponent_mover_with_primary_context():
    primary_mover, opponent_mover_context = _compare_pair(
        ALPHA_QUERY,
        "primary-mover",
        PRIMARY_MOVER,
        "opponent-mover-context",
        OPPONENT_MOVER_PRIMARY_CONTEXT,
    )
    assert primary_mover > opponent_mover_context


def test_invariant_e_opponent_only_remains_retrieval_eligible():
    ranking = _ranking(ALPHA_QUERY, "opponent-only", OPPONENT_ONLY)
    assert ranking["final_score"] > 0.0


@pytest.mark.xfail(
    strict=True,
    reason="B2 RED_EXPECTED: opponent-only must not receive primary-subject equivalence",
)
def test_invariant_e_opponent_only_must_not_receive_primary_equivalence():
    primary_mover, opponent_only = _compare_pair(
        ALPHA_QUERY,
        "primary-mover",
        PRIMARY_MOVER,
        "opponent-only",
        OPPONENT_ONLY,
    )
    assert opponent_only < primary_mover


@pytest.mark.xfail(
    strict=True,
    reason="B2 RED_EXPECTED: metadata-only primary presence should rank below passage-local primary support",
)
def test_invariant_f_passage_local_primary_beats_metadata_only():
    passage_local, metadata_only = _compare_pair(
        ALPHA_QUERY,
        "passage-local",
        PRIMARY_MOVER,
        "metadata-only",
        NEITHER_PERSON,
        right_metadata={"author": "Commander Alpha", "work": "Campaign Journal"},
    )
    assert passage_local > metadata_only


@pytest.mark.parametrize(
    ("case_id", "text", "metadata", "relation", "other_id", "other_text", "other_metadata"),
    [
        pytest.param(
            "primary-mover",
            PRIMARY_MOVER,
            {},
            "gt",
            "primary-context",
            PRIMARY_CONTEXT,
            {},
            id="primary-mover",
        ),
        pytest.param(
            "primary-vs-opponent-comparable",
            PRIMARY_MOVER,
            {},
            "gt",
            "opponent-only",
            OPPONENT_ONLY,
            {},
            id="primary-vs-opponent-comparable",
        ),
        pytest.param(
            "primary-plus-opponent",
            PRIMARY_OPPONENT_MOVER,
            {},
            "lte_primary_only",
            "primary-only",
            PRIMARY_MOVER,
            {},
            id="primary-plus-opponent",
        ),
        pytest.param(
            "opponent-mover-context",
            OPPONENT_MOVER_PRIMARY_CONTEXT,
            {},
            "lt",
            "primary-mover",
            PRIMARY_MOVER,
            {},
            marks=pytest.mark.xfail(
                strict=True,
                reason="B2 RED_EXPECTED: opponent mover + primary context must stay below explicit primary mover",
            ),
            id="opponent-mover-primary-context",
        ),
        pytest.param(
            "both-no-movement",
            BOTH_NO_MOVEMENT,
            {},
            "lt",
            "primary-mover",
            PRIMARY_MOVER,
            {},
            id="both-no-movement",
        ),
        pytest.param(
            "neither-person",
            NEITHER_PERSON,
            {},
            "lt",
            "primary-mover",
            PRIMARY_MOVER,
            {},
            marks=pytest.mark.xfail(
                strict=True,
                reason="B2 RED_EXPECTED: person-absent movement must stay below explicit primary mover",
            ),
            id="neither-person",
        ),
        pytest.param(
            "metadata-only",
            NEITHER_PERSON,
            {"author": "Commander Alpha", "work": "Campaign Journal"},
            "lt",
            "primary-mover",
            PRIMARY_MOVER,
            {},
            marks=pytest.mark.xfail(
                strict=True,
                reason="B2 RED_EXPECTED: metadata-only primary presence must stay below passage-local primary support",
            ),
            id="metadata-only",
        ),
        pytest.param(
            "pronoun-movement",
            PRONOUN_MOVEMENT,
            {},
            "lt",
            "primary-mover",
            PRIMARY_MOVER,
            {},
            marks=pytest.mark.xfail(
                strict=True,
                reason="B2 RED_EXPECTED: pronoun movement must stay below explicit primary mover",
            ),
            id="pronoun-movement",
        ),
        pytest.param(
            "ambiguous-multi-person",
            AMBIGUOUS_MULTI,
            {},
            "lt",
            "primary-mover",
            PRIMARY_MOVER,
            {},
            id="ambiguous-multi-person",
        ),
    ],
)
def test_generic_matrix_relative_order(
    case_id: str,
    text: str,
    metadata: dict,
    relation: str,
    other_id: str,
    other_text: str,
    other_metadata: dict,
):
    left, right = _compare_pair(
        ALPHA_QUERY,
        case_id,
        text,
        other_id,
        other_text,
        left_metadata=metadata,
        right_metadata=other_metadata,
    )
    if relation == "gt":
        assert left > right
    elif relation == "lt":
        assert left < right
    elif relation == "lte_primary_only":
        assert left <= right + _BOUNDED_OPPONENT_UPLIFT
    else:
        raise AssertionError(f"unknown relation {relation!r}")


# ---------------------------------------------------------------------------
# Phase 3 — unknown primary contract (Mithridates benchmark)
# ---------------------------------------------------------------------------


def test_unknown_primary_no_authoritative_subject():
    roles = analyze_query(MITHRIDATES_QUERY)
    assert primary_route_subject(MITHRIDATES_QUERY, roles) is None


def test_unknown_primary_weak_person_context_retained():
    ranking = _ranking(
        MITHRIDATES_QUERY,
        "mithridates-movement",
        "Mithridates withdrew from Pontus toward the interior of Anatolia.",
    )
    assert ranking["final_score"] > 0.0
    assert ranking["person_support"] > 0.0


def test_unknown_primary_does_not_publish_authoritative_subject():
    roles = analyze_query(MITHRIDATES_QUERY)
    ranking = _ranking(
        MITHRIDATES_QUERY,
        "mithridates-movement",
        "Mithridates marched from Pontus through Asia Minor toward Greece.",
    )
    assert primary_route_subject(MITHRIDATES_QUERY, roles) is None
    assert ranking["final_score"] > 0.0


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from backend.app.core.config import settings
        from backend.app.rag.http_store import build_production_retriever

        retriever = build_production_retriever(settings)
        retriever.retrieve("Rome", 1)
        return retriever
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"production chroma unavailable: {exc}")


@pytest.mark.integration
def test_mithridates_trusted_evidence_proposal_retrieval_safety(production_retriever):
    ref = _ref("g5r-mithridates-001")
    trace = _trace_mithridates_ref(production_retriever, ref)
    assert trace["proposal_present"] is True
    assert trace["union_present"] is True
    assert trace["final_present"] is True


# ---------------------------------------------------------------------------
# Phase 4 — historical regression matrix
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="B2 RED_EXPECTED: Lucullus mover should outscore Mithridates mover + Lucullus context",
)
def test_historical_lucullus_mover_beats_mithridates_mover_with_lucullus_context():
    lucullus_mover, mithridates_mover_context = _compare_pair(
        LUCULLUS_QUERY,
        "lucullus-mover",
        "Lucullus marched through Armenia toward Asia Minor after Mithridates withdrew.",
        "mithridates-mover-context",
        "Mithridates marched through Armenia into Pontus while Lucullus remained in camp.",
    )
    assert lucullus_mover > mithridates_mover_context


@pytest.mark.xfail(
    strict=True,
    reason="B2 RED_EXPECTED: Alexander mover should outscore Porus mover + Alexander context",
)
def test_historical_alexander_mover_beats_porus_mover_with_alexander_context():
    alexander_mover, porus_mover_context = _compare_pair(
        ALEXANDER_QUERY,
        "alexander-mover",
        "Alexander crossed the Hydaspes toward Porus after securing the ford.",
        "porus-mover-context",
        "Porus marched along the Hydaspes while Alexander watched from the opposite bank.",
    )
    assert alexander_mover > porus_mover_context


@pytest.mark.xfail(
    strict=True,
    reason="B2 RED_EXPECTED: Sulla mover should outscore Mithridates mover + Sulla context",
)
def test_historical_sulla_mover_beats_mithridates_mover_with_sulla_context():
    sulla_mover, mithridates_mover_context = _compare_pair(
        SULLA_QUERY,
        "sulla-mover",
        "Sulla marched through Greece pursuing Mithridates toward the coast.",
        "mithridates-mover-context",
        "Mithridates marched through Greece while Sulla held the lines near Athens.",
    )
    assert sulla_mover > mithridates_mover_context


def test_pompey_single_person_regression_control():
    ranking = _ranking(
        POMPEY_QUERY,
        "pompey-route",
        "Pompey sailed from Greece toward the eastern Mediterranean after Pharsalus.",
    )
    assert ranking["person_support"] >= 0.08
    assert ranking["final_score"] >= 1.0


def test_caesar_single_person_regression_control():
    ranking = _ranking(
        CAESAR_QUERY,
        "caesar-route",
        "Caesar crossed the Adriatic into Epirus before advancing toward Pharsalus.",
    )
    assert ranking["person_support"] >= 0.04
    assert ranking["final_score"] >= 0.9


# ---------------------------------------------------------------------------
# Phase 5 — duplication contract
# ---------------------------------------------------------------------------


def test_duplication_opponent_mention_does_not_auto_uplift():
    primary_only, primary_with_opponent = _compare_pair(
        ALPHA_QUERY,
        "primary-only",
        PRIMARY_MOVER,
        "primary-with-opponent",
        PRIMARY_OPPONENT_MOVER,
    )
    assert primary_with_opponent <= primary_only + _BOUNDED_OPPONENT_UPLIFT


@pytest.mark.xfail(
    strict=True,
    reason="B2 RED_EXPECTED: parallel person/entity/joint paths must stay bounded for one primary match",
)
def test_duplication_parallel_person_path_mass_is_capped():
    ranking = _ranking(ALPHA_QUERY, "primary-mover", PRIMARY_MOVER)
    duplicate_mass = ranking["entity_support"] + ranking["joint_support"]
    assert duplicate_mass <= 0.04


def test_duplication_secondary_name_count_does_not_accumulate():
    one_secondary, three_secondary = _compare_pair(
        ALPHA_QUERY,
        "one-secondary",
        "Commander Alpha marched from Gamma Harbor while Commander Beta watched.",
        "three-secondary",
        (
            "Commander Alpha marched from Gamma Harbor while Commander Beta, Commander Gamma, "
            "and Commander Delta watched."
        ),
    )
    assert three_secondary <= one_secondary + _BOUNDED_OPPONENT_UPLIFT
