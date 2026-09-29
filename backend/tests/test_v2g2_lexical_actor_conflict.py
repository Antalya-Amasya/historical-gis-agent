"""Actor-name substitutions must not change lexical heading-conflict decisions."""
from __future__ import annotations

from backend.app.rag.lexical_index import LexicalEvidenceIndex
from backend.app.rag.query_roles import analyze_query


class _Corpus:
    def __init__(self, documents: list[str], heading: str = "ARISTON") -> None:
        self.documents = documents
        self.heading = heading

    def get(self, include=None) -> dict:
        return {
            "ids": [f"source-{index}" for index in range(len(self.documents))],
            "documents": self.documents,
            "metadatas": [{"heading": self.heading} for _ in self.documents],
        }


def _results(documents: list[str], *, query: str = "Trace Ariston's route through Gaul.", heading: str = "ARISTON") -> dict:
    candidates = LexicalEvidenceIndex(_Corpus(documents, heading)).query(query, len(documents))
    return {candidate.text: candidate for candidate in candidates}


def test_known_and_unseen_explicit_actors_have_the_same_conflict_decision() -> None:
    documents = [f"{name} marched through Gaul." for name in ("Brutus", "Damon", "Nereus")]
    results = _results(documents)
    assert set(results) == set(documents)
    assert {item.metadata["lexical_provenance_subject_score"] for item in results.values()} == {0}
    assert len({item.lexical_score for item in results.values()}) == 1


def test_explicit_query_actor_uses_body_support_without_heading_boost() -> None:
    results = _results(["Ariston marched through Gaul.", "Damon marched through Gaul."])
    query_actor = results["Ariston marched through Gaul."]
    assert query_actor.metadata["lexical_body_subject_score"] > 0
    assert query_actor.metadata["lexical_provenance_subject_score"] == 0
    assert results["Damon marched through Gaul."].metadata["lexical_provenance_subject_score"] == 0


def test_implicit_and_pronoun_led_movement_keep_heading_support() -> None:
    documents = ["He marched through Gaul.", "From Gaul he marched toward Rome."]
    results = _results(documents)
    assert all(results[text].metadata["lexical_provenance_subject_score"] > 0 for text in documents)


def test_named_object_is_not_a_conflicting_movement_actor() -> None:
    text = "He marched from Gaul after speaking with Brutus."
    assert _results([text])[text].metadata["lexical_provenance_subject_score"] > 0


def test_unrelated_neighbor_cannot_change_actor_conflict() -> None:
    target = "Damon marched through Gaul."
    implicit = "He marched through Gaul."
    unrelated = "Nereus debated policy in Greece."
    for documents in ([target, implicit], [target, implicit, unrelated], [unrelated, implicit, target]):
        results = _results(documents)
        assert results[target].metadata["lexical_provenance_subject_score"] == 0
        assert results[implicit].metadata["lexical_provenance_subject_score"] > 0


def test_place_mention_before_pronoun_is_not_an_explicit_actor() -> None:
    text = "From Athens he marched through Gaul."
    assert _results([text])[text].metadata["lexical_provenance_subject_score"] > 0


def test_candidate_order_reversal_does_not_change_scores_or_conflicts() -> None:
    documents = ["Brutus marched through Gaul.", "Damon marched through Gaul.", "He marched through Gaul."]
    forward = _results(documents)
    reverse = _results(list(reversed(documents)))
    for text in documents:
        assert forward[text].lexical_score == reverse[text].lexical_score
        assert (
            forward[text].metadata["lexical_provenance_subject_score"]
            == reverse[text].metadata["lexical_provenance_subject_score"]
        )


def test_full_name_conflict_and_surname_only_uncertainty_remain_distinct() -> None:
    documents = [
        "Marcus Valerius marched through Gaul.",
        "Valerius marched through Gaul.",
        "Lucius Valerius marched through Gaul.",
    ]
    index = LexicalEvidenceIndex(_Corpus(documents, "LUCIUS VALERIUS"))
    index.ensure_built()
    roles = analyze_query("Trace Lucius Valerius from Rome to Capua.")
    supported = frozenset({"lucius", "valerius"})
    assert index._body_actor_conflicts(next(i for i in index._passages if i.startswith("source-0:")), roles, supported)
    assert not index._body_actor_conflicts(next(i for i in index._passages if i.startswith("source-1:")), roles, supported)
    assert not index._body_actor_conflicts(next(i for i in index._passages if i.startswith("source-2:")), roles, supported)


def test_opponent_named_in_query_is_not_the_heading_subject() -> None:
    text = "Damon marched through Gaul."
    results = _results(
        [text], query="Trace Ariston during the war against Damon through Gaul.", heading="ARISTON",
    )
    assert results[text].metadata["lexical_provenance_subject_score"] == 0
