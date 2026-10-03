"""Provider-free person relevance contracts on synthetic frozen passage pools."""
from copy import deepcopy

import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retriever import ChromaHistoricalRetriever


def ev(identifier, text, **metadata):
    return Evidence(id=identifier, author="Author", work="Work", locator="Book 1",
                    excerpt=text, text=text, score=0, metadata={
                        "source_chunk_id": identifier, **metadata})


def detail(item):
    return item.metadata["retrieval_ranking"]


def select(query, items):
    return ChromaHistoricalRetriever(None)._finalize_selection(query, items, 10)


@pytest.mark.parametrize("source", ["legacy-span", "sentence-span"])
def test_local_unseen_person_beats_parent_only_support_in_both_packings(source):
    good = ev("good", "Neralis Vexon marched into Ardan with the legion.",
              lexical_candidate=True, lexical_score=10, source_chunk_id=source)
    bad = ev("bad", "Torven Valdis marched into Ardan with the legion.",
             semantic_candidate=True, vector_rank=1, distance=.01,
             heading="NERALIS VEXON", parent_person="Neralis Vexon")
    out = select("Neralis Vexon march route into Ardan", [bad, good])
    assert out[0].id == "good"
    # Both remain inspectable: this is relevance, not admission/authority.
    assert {e.id for e in out} == {"bad", "good"}


def test_explicit_competitor_penalty_is_bounded_and_keeps_semantic_signal():
    bad = ev("bad", "Torven Valdis marched into Ardan with the legion.",
             semantic_candidate=True, vector_rank=1, distance=.01)
    out = rerank_evidence("Neralis Vexon marched into Ardan", [bad])[0]
    assert 0 < detail(out)["person_mismatch_penalty"] <= .32
    assert detail(out)["semantic_vector_bonus"] > 0
    assert out.score > 0


def test_same_surname_competitor_does_not_dominate_full_name():
    good = ev("z-full", "Quintus Neralis Vexon marched into Ardan.",
              lexical_candidate=True, lexical_score=10)
    bad = ev("a-surname", "Lucius Neralis Vexon marched into Ardan.",
             semantic_candidate=True, vector_rank=1, distance=.01,
             lexical_candidate=True, lexical_score=10)
    out = select("Quintus Neralis Vexon marched into Ardan", [bad, good])
    assert out[0].id == "z-full"
    assert detail(next(e for e in out if e.id == "a-surname"))["person_support"] == 0


def test_unrelated_praenomen_does_not_veto_requested_surname():
    item = ev("local", "Vexon marched into Ardan. Lucius gave orders to the guards.",
              lexical_candidate=True, lexical_score=10)
    assert detail(rerank_evidence("Quintus Neralis Vexon marched into Ardan", [item])[0])["person_support"] == .04


def test_praenomen_alone_cannot_support_unseen_multi_token_person():
    item = ev("given", "Gaius Norbanus marched into Ardan.",
              semantic_candidate=True, vector_rank=1, distance=.01)
    assert detail(rerank_evidence("Gaius Neralis Vexon marched into Ardan", [item])[0])["person_support"] == 0


def test_orthographic_praenomen_variant_with_local_surname_is_supported():
    item = ev("variant", "Caius Vexon sailed into Ardan. Publius commanded another army.",
              lexical_candidate=True, lexical_score=10)
    assert detail(rerank_evidence("Gaius Neralis Vexon marched into Ardan", [item])[0])["person_support"] == .04


def test_semantic_vector_still_contributes_for_equal_local_evidence():
    low = ev("low", "Neralis Vexon marched into Ardan.",
             semantic_candidate=True, vector_rank=2, distance=.5)
    high = ev("high", low.text, semantic_candidate=True, vector_rank=1, distance=.25)
    out = rerank_evidence("Neralis Vexon marched into Ardan", [low, high])
    assert out[0].id == "high" and out[0].score > out[1].score


def test_equal_lexical_scores_do_not_inherit_source_id_relevance():
    a = ev("a", "Quintus Neralis Vexon served as a priest.", lexical_candidate=True, lexical_score=10)
    z = ev("z", "Quintus Neralis Vexon served as an ambassador.", lexical_candidate=True, lexical_score=10)
    out = rerank_evidence("Quintus Neralis Vexon", [a, z])
    assert out[0].score == out[1].score
    assert detail(out[0])["lexical_rank"] == detail(out[1])["lexical_rank"]
    # Same full name with no distinguishing query context remains unresolved.
    assert all(detail(e)["person_mismatch_penalty"] == 0 for e in out)


def test_full_name_support_beats_ambiguous_surname_with_same_lexical_score():
    full = ev("full", "Quintus Neralis Vexon visited the town.", lexical_candidate=True, lexical_score=10)
    surname = ev("surname", "Vexon visited the town.", lexical_candidate=True, lexical_score=10)
    assert rerank_evidence("Quintus Neralis Vexon", [surname, full])[0].id == "full"


def test_unseen_same_surname_cannot_use_parent_vector_to_dominate_full_name():
    full = ev("full", "Neralis Vexon marched into Ardan.", lexical_candidate=True, lexical_score=10)
    ambiguous = ev("ambiguous", "Torven Vexon marched into Ardan.",
                   lexical_candidate=True, lexical_score=10,
                   semantic_candidate=True, vector_rank=1, distance=.01)
    out = rerank_evidence("Neralis Vexon marched into Ardan", [ambiguous, full])
    assert out[0].id == "full"
    prior = detail(next(e for e in out if e.id == "ambiguous"))
    assert 0 < prior["semantic_vector_bonus"] < prior["source_vector_bonus"]


def test_directional_diversity_does_not_promote_wrong_actor_over_local_person():
    wrong = ev("wrong", "Torven Valdis marched from the camp into Ardan.",
               semantic_candidate=True, vector_rank=1, distance=.01)
    right = ev("right", "Neralis Vexon crossed into Ardan.",
               lexical_candidate=True, lexical_score=10)
    out = select("Neralis Vexon march route into Ardan", [wrong, right])
    assert out[0].id == "right"


def test_parent_identity_fields_do_not_change_passage_person_support():
    item = ev("local", "Neralis Vexon marched into Ardan.", lexical_candidate=True, lexical_score=10)
    parent = item.model_copy(update={"metadata": {**item.metadata, "heading": "TORVEN VALDIS"}})
    a = rerank_evidence("Neralis Vexon marched into Ardan", [item])[0]
    b = rerank_evidence("Neralis Vexon marched into Ardan", [parent])[0]
    assert detail(a)["person_support"] == detail(b)["person_support"] == .08


def test_rare_lexical_entity_retains_strength():
    rare = ev("rare", "Ardanium was besieged by the army.", lexical_candidate=True, lexical_score=30)
    vague = ev("vague", "The armies fought in another distant city.", semantic_candidate=True, vector_rank=1, distance=.01)
    assert select("Ardanium besieged", [vague, rare])[0].id == "rare"


def test_authority_metadata_and_input_evidence_are_immutable():
    item = ev("wrong", "Torven Valdis marched into Ardan.", semantic_candidate=True,
              distance=.01, vector_rank=1, actor_authority="UNRESOLVED",
              movement_authority="UNRESOLVED", historical_route="UNKNOWN",
              travel_mode="UNKNOWN", completion=False, chronology="UNKNOWN")
    before = deepcopy(item.model_dump())
    result = rerank_evidence("Neralis Vexon marched into Ardan", [item])[0]
    assert item.model_dump() == before
    for key, value in item.metadata.items():
        assert result.metadata[key] == value
    assert result.text == item.text
