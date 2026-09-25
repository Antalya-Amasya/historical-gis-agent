"""R4-C2: combined bounded lexical policy (generic first, named corpus as controls)."""
from __future__ import annotations

import json

from backend.app.rag.lexical_index import LexicalEvidenceIndex
from backend.app.rag.query_roles import analyze_query


class _Corpus:
    def __init__(self, docs, metas=None):
        self._docs = docs
        self._metas = metas or [{} for _ in docs]

    def get(self, include=None):
        return {
            "ids": [f"c{i}" for i in range(len(self._docs))],
            "documents": self._docs,
            "metadatas": self._metas,
        }


def _index(docs, metas=None):
    return LexicalEvidenceIndex(_Corpus(docs, metas))


def test_body_constraint_outranks_expanded_movement_mass():
    query = "What movement of Marius is attested during the negotiations involving Bocchus?"
    docs = [
        "Marius arrived at Cirta. Messengers came from Bocchus asking for a conference.",
        (
            "Marius marched, sailed, crossed, advanced, and proceeded through the province with the army."
        ),
    ]
    results = _index(docs).query(query, 2)
    by_text = {item.text: item.lexical_score for item in results}
    constrained = next(text for text in by_text if "Bocchus" in text)
    expanded_only = next(text for text in by_text if "Bocchus" not in text)
    assert by_text[constrained] > by_text[expanded_only]


def test_body_location_outranks_expanded_movement_without_location():
    query = "Trace Marius from Cirta toward Bocchus."
    docs = [
        "Marius arrived at Cirta and met Bocchus.",
        (
            "Marius marched, sailed, crossed, advanced, proceeded, departed, "
            "travelled, entered, passed, and moved with the army."
        ),
    ]
    results = _index(docs).query(query, 2)
    located = next(item for item in results if "Cirta" in item.text)
    movement_only = next(item for item in results if "Cirta" not in item.text)
    assert located.lexical_score > movement_only.lexical_score


def test_literal_action_outranks_expansion_only_match():
    query = "Trace Marius marched from Cirta."
    roles = analyze_query(query)
    assert "marched" in roles.action_terms
    docs = [
        "Marius marched from Cirta with the army.",
        "Marius advanced from Cirta with the army.",
    ]
    results = _index(docs).query(query, 2)
    literal = next(item for item in results if "marched" in item.text)
    expanded = next(item for item in results if "advanced" in item.text)
    assert literal.lexical_score > expanded.lexical_score


def test_pronoun_movement_under_matching_heading_outranks_unrelated_heading():
    docs = [
        "He marched so fast and put to sea in winter, passing the Ionian Sea.",
        "He marched so fast and put to sea in winter, passing the Ionian Sea.",
    ]
    metas = [
        {"heading": "CAESAR", "navigation_path_json": json.dumps(["CAESAR"])},
        {"heading": "THESEUS", "navigation_path_json": json.dumps(["THESEUS"])},
    ]
    results = _index(docs, metas).query("Trace Caesar route march across the Adriatic", 2)
    caesar = next(item for item in results if item.metadata.get("heading") == "CAESAR")
    theseus = next(item for item in results if item.metadata.get("heading") == "THESEUS")
    assert caesar.metadata["lexical_provenance_subject_score"] > 0
    assert theseus.metadata["lexical_provenance_subject_score"] == 0
    assert caesar.lexical_score > theseus.lexical_score


def test_expanded_movement_still_gates_provenance():
    docs = ["He put to sea in winter and landed near Oricum."]
    metas = [{"heading": "CAESAR", "navigation_path_json": json.dumps(["CAESAR"])}]
    results = _index(docs, metas).query("Trace Caesar route across the Adriatic", 1)
    assert results[0].metadata["lexical_movement_score"] > 0
    assert results[0].metadata["lexical_provenance_subject_score"] > 0


def test_sort_remains_score_then_id():
    docs = [
        "Alpha marched from Gamma Harbor to Delta Bay.",
        "Alpha marched from Gamma Harbor to Delta Bay.",
    ]
    results = _index(docs).query("Trace Alpha marched from Gamma Harbor to Delta Bay.", 2)
    scores = [item.lexical_score for item in results]
    assert scores == sorted(scores, reverse=True)
    tied = [item for item in results if item.lexical_score == results[0].lexical_score]
    assert [item.id for item in tied] == sorted(item.id for item in tied)
