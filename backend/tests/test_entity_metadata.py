"""Open-world, retrieval-only entity metadata contracts; no provider calls."""
from copy import deepcopy

import pytest

from backend.app.models import Evidence
from backend.app.rag.entity_metadata import resolve_query_person, entity_compatibility, person_records


def profiles():
    return [dict(entity_id=key, canonical_name="Neralis Vexon", entity_type="PERSON",
                 aliases=["Neralis Vexon", "Vexon"], roles_titles=[role], context_places=[place],
                 time_range=bounds, provenance=[{"resource": "synthetic_control"}])
            for key, role, place, bounds in [
                ("a", "ambassador", "Ardan", {"start_year": -220, "end_year": -200}),
                ("b", "priest", "Belmar", {"start_year": -170, "end_year": -150})]]


def ev(text, **metadata):
    return Evidence(id="local", author="Livy", work="History", locator="Book 1",
                    excerpt=text, text=text, score=0, metadata=metadata)


def test_known_exact_person_has_provenance():
    q = resolve_query_person("Hannibal")
    assert q["match_type"] == "EXACT" and q["entity_type"] == "PERSON"
    r = next(r for r in person_records() if r["entity_id"] == q["entity_id"])
    assert r["provenance"] and r["normalized_tokens"]


@pytest.mark.parametrize("alias", ["Julius Caesar", "凯撒"])
def test_existing_known_alias_preserves_canonical_candidate(alias):
    q = resolve_query_person(alias)
    assert q["entity_id"] == resolve_query_person("Caesar")["entity_id"]


def test_unseen_literal_person_has_no_registry_miss_penalty():
    q = resolve_query_person("Neralis Vexon marched into Ardan")
    assert q["match_type"] == "LITERAL_UNKNOWN" and q["entity_type"] == "PERSON"
    assert q["entity_id"] is None
    s = entity_compatibility("Neralis Vexon marched into Ardan", "Neralis Vexon marched into Ardan.")
    assert s["entity_bonus"] == 0 and s["entity_compatibility"] == "NO_LOCAL_ENTITY"


def test_bare_same_name_stays_ambiguous():
    q = resolve_query_person("Neralis Vexon", records=profiles())
    assert q["match_type"] == "AMBIGUOUS" and q["entity_id"] is None
    s = entity_compatibility("Neralis Vexon", "Neralis Vexon was the priest.", records=profiles())
    assert s["entity_compatibility"] == "AMBIGUOUS" and s["entity_bonus"] == 0


def test_context_independently_distinguishes_homonym_candidates():
    q = "Neralis Vexon ambassador in Ardan"
    assert resolve_query_person(q, records=profiles())["entity_id"] == "a"
    a = entity_compatibility(q, "Neralis Vexon, the ambassador, spoke in Ardan.", records=profiles())
    b = entity_compatibility(q, "Neralis Vexon, the priest, spoke in Belmar.", records=profiles())
    assert a["entity_compatibility"] == "MATCH" and a["entity_bonus"] == .08
    assert b["entity_compatibility"] == "CONFLICT" and b["entity_bonus"] == -.08


def test_unresolved_passage_is_not_forced_to_a_known_identity():
    s = entity_compatibility("Neralis Vexon ambassador", "Neralis Vexon spoke.", records=profiles())
    assert s["entity_compatibility"] == "AMBIGUOUS" and s["entity_bonus"] == 0


def test_coarse_temporal_overlap_requires_explicit_era_and_is_bounded():
    q = "Neralis Vexon in 210 BCE"
    assert resolve_query_person(q, records=profiles())["entity_id"] == "a"
    s = entity_compatibility(q, "Neralis Vexon spoke in 210 BCE.", records=profiles())
    assert s["entity_bonus"] == .08
    assert resolve_query_person("Neralis Vexon in 210", records=profiles())["entity_id"] is None


def test_conflicting_query_context_remains_ambiguous():
    q = resolve_query_person("Neralis Vexon priest ambassador", records=profiles())
    assert q["entity_id"] is None


@pytest.mark.parametrize("name", ["Rome", "Delphi", "Cilicia"])
def test_place_surface_is_never_a_curated_person(name):
    q = resolve_query_person(name)
    assert q["entity_type"] != "PERSON" and q["entity_id"] is None


def test_author_metadata_is_not_person_or_movement_support():
    s = entity_compatibility("Livy route", "The army sailed into Africa.")
    assert s["entity_bonus"] == 0 and s["passage_entity_ids"] == []
    q = resolve_query_person("According to Livy, Caesar crossed the river.")
    assert q["entity_id"] == resolve_query_person("Caesar")["entity_id"]


def test_orthographic_variants_work_without_registry_lookup():
    records = [dict(entity_id="x", canonical_name="Gaius Neralis Vexon", entity_type="PERSON",
                    aliases=["Caius Neralis Vexon"], provenance=[{}])]
    a = resolve_query_person("Caius Neralis Vexon", records=records)
    b = resolve_query_person("Gaius Neralis Vexon", records=records)
    assert a["entity_id"] == b["entity_id"] == "x"


def test_unknown_identity_does_not_become_place_or_invalid():
    for q in ("Curio", "Neralis Vexon"):
        result = resolve_query_person(q)
        assert result["entity_type"] != "PLACE" and result["match_type"] == "LITERAL_UNKNOWN"


def test_registry_covers_existing_resources_beyond_diagnostics():
    records = person_records()
    assert len(records) == 16
    assert all(r["entity_type"] == "PERSON" and r["provenance"] for r in records)
    assert all(r["time_range"] is None and not r["external_ids"] for r in records)
    assert len([r for r in records if r["identity_basis"] != "SOURCE_CONTEXT_CANDIDATE_NOT_AUTHORITY"]) == 14


def test_registry_and_observations_are_not_mutated():
    records = profiles();before = deepcopy(records)
    entity_compatibility("Neralis Vexon ambassador", "Neralis Vexon was the priest.", records=records)
    assert records == before


@pytest.mark.parametrize("source", ["legacy", "sentence"])
def test_shadow_same_name_context_ranks_matching_profile(source, monkeypatch):
    import backend.app.rag.entity_metadata as entity
    from backend.app.rag.evidence_ranking import rerank_evidence
    monkeypatch.setattr(entity, "person_records", lambda: profiles())
    a = ev("Neralis Vexon, the ambassador, spoke in Ardan.", lexical_candidate=True, lexical_score=10, source_chunk_id=source)
    b = a.model_copy(update={"id": "other", "text": "Neralis Vexon, the priest, spoke in Belmar."})
    ranked = rerank_evidence("Neralis Vexon ambassador in Ardan", [b, a], entity_metadata=True)
    assert ranked[0].id == "local"
    scores = {r.id: r.metadata["retrieval_ranking"] for r in ranked}
    assert scores["local"]["entity_bonus"] > 0 > scores["other"]["entity_bonus"]
    bare = rerank_evidence("Neralis Vexon", [a, b], entity_metadata=True)
    assert all(r.metadata["retrieval_ranking"]["query_entity_match_type"] == "AMBIGUOUS" for r in bare)
    assert all(r.metadata["retrieval_ranking"]["entity_bonus"] == 0 for r in bare)


def test_unknown_shadow_score_equals_registry_disabled_score():
    from backend.app.rag.evidence_ranking import rerank_evidence
    item = ev("Neralis Vexon marched into Ardan.", lexical_candidate=True, lexical_score=10)
    a = rerank_evidence("Neralis Vexon marched into Ardan", [item])[0]
    b = rerank_evidence("Neralis Vexon marched into Ardan", [item], entity_metadata=True)[0]
    assert a.score == b.score and b.metadata["retrieval_ranking"]["entity_bonus"] == 0


def test_shadow_authority_and_input_metadata_remain_immutable(monkeypatch):
    import backend.app.rag.entity_metadata as entity
    from backend.app.rag.evidence_ranking import rerank_evidence
    monkeypatch.setattr(entity, "person_records", lambda: profiles())
    item = ev("Neralis Vexon, the priest, spoke in Belmar.", lexical_candidate=True, lexical_score=10,
              actor_authority="UNKNOWN", movement_authority="UNKNOWN", historical_route="UNKNOWN",
              completion=False, travel_mode="UNKNOWN", chronology="UNKNOWN", event_membership="UNKNOWN",
              route_admission="UNRESOLVED")
    before = deepcopy(item.model_dump())
    result = rerank_evidence("Neralis Vexon ambassador in Ardan", [item], entity_metadata=True)[0]
    assert item.model_dump() == before and result.text == item.text
    for k, v in item.metadata.items():
        assert result.metadata[k] == v
    assert result.author == "Livy" and result.metadata["retrieval_ranking"]["entity_bonus"] < 0


def test_default_reranker_does_not_load_entity_registry(monkeypatch):
    import backend.app.rag.entity_metadata as entity
    from backend.app.rag.evidence_ranking import rerank_evidence
    def forbidden():
        raise AssertionError("default production path read shadow registry")
    monkeypatch.setattr(entity, "person_records", forbidden)
    assert "entity_bonus" not in rerank_evidence("Neralis Vexon", [ev("Neralis Vexon spoke.")])[0].metadata["retrieval_ranking"]


def test_conflicting_role_does_not_promote_another_actors_movement(monkeypatch):
    import backend.app.rag.entity_metadata as entity
    import backend.app.rag.retriever as retrieval
    from backend.app.rag.evidence_ranking import rerank_evidence
    monkeypatch.setattr(entity, "person_records", lambda: profiles())
    monkeypatch.setattr(retrieval, "rerank_evidence", lambda q, items: rerank_evidence(q, items, entity_metadata=True))
    bad = ev("Neralis Vexon, the priest, died in Belmar. King Torven marched from Belmar to Ardan.",
             semantic_candidate=True, vector_rank=1, distance=.01, lexical_candidate=True, lexical_score=10)
    good = ev("Neralis Vexon, the ambassador, returned from Ardan.", lexical_candidate=True, lexical_score=10)
    good = good.model_copy(update={"id": "good", "metadata": {**good.metadata, "source_chunk_id": "good"}})
    out = retrieval.ChromaHistoricalRetriever(None)._finalize_selection("Neralis Vexon route as ambassador from Ardan", [bad, good], 10)
    assert out[0].id == "good" and {r.id for r in out} == {"local", "good"}
