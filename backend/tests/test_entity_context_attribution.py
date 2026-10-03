"""Mention ownership controls; injected profiles are not identity authority."""
from copy import deepcopy

import pytest

from backend.app.rag.entity_metadata import entity_compatibility, resolve_query_person


def profiles(name, roles=("ambassador", "priest"), places=("Ardan", "Belmar")):
    return [dict(entity_id=key, canonical_name=name, entity_type="PERSON", aliases=[],
                 roles_titles=[role], context_places=[place], time_range=None)
            for key, role, place in zip(("envoy", "priest"), roles, places)]


@pytest.mark.parametrize("name", ["Aelius Cassianus", "Neratius Philon"])
@pytest.mark.parametrize("template", [
    "{name}, the ambassador, waited.",
    "{name} departed from Ardan.",
    "{name}, the ambassador from Ardan, waited.",
    "{name}, ambassador to Ardan, departed.",
    "{name} served as ambassador in Ardan.",
    "The ambassador {name} arrived in Ardan.",
    "{name} was appointed ambassador in Ardan.",
    "{name} waited. {name}, the ambassador, arrived in Ardan.",
])
def test_same_person_owns_context(name, template):
    result = entity_compatibility(name + " ambassador Ardan", template.format(name=name), records=profiles(name))
    assert result["entity_compatibility"] == "MATCH" and result["entity_bonus"] == .08


@pytest.mark.parametrize("name", ["Aelius Cassianus", "Neratius Philon"])
@pytest.mark.parametrize("template", [
    "{name} waited. King Torven, the ambassador, arrived in Ardan.",
    "According to the ambassador Torven in Ardan, {name} waited.",
    "{name} waited while King Torven, the ambassador from Ardan, arrived.",
    "{name} met King Torven, ambassador from Ardan.",
    "{name} met torven, ambassador from Ardan.",
    "{name} waited and torven, the ambassador, arrived in Ardan.",
    "{name} waited, the ambassador Torven arrived in Ardan.",
    "{name} waited. Torven was ambassador.",
    "{name} waited while Torven departed from Ardan.",
    "{name} waited. Torven arrived in Ardan.",
    "Livy, writing from Ardan, says {name} waited.",
    "{name} was appointed ambassador. He departed for Ardan.",
])
def test_context_owned_by_other_person_or_unresolved_pronoun_is_neutral(name, template):
    text = template.format(name=name)
    result = entity_compatibility(name + " ambassador Ardan", text, records=profiles(name))
    assert result["entity_bonus"] == 0 and result["entity_compatibility"] == "AMBIGUOUS"
    assert result["entity_context_match"] == []


@pytest.mark.parametrize("name", ["Aelius Cassianus", "Neratius Philon"])
def test_bare_conflict_unknown_and_query_ownership(name):
    records = profiles(name); before = deepcopy(records)
    assert resolve_query_person(name, records=records)["entity_id"] is None
    assert resolve_query_person(name + " waited while Torven was ambassador in Ardan", records=records)["entity_id"] is None
    conflicting = name + ", ambassador and priest in Ardan, waited."
    assert resolve_query_person(conflicting, records=records)["entity_id"] is None
    result = entity_compatibility(name + " ambassador Ardan", conflicting, records=records)
    assert result["entity_bonus"] == 0
    unknown = entity_compatibility("Nikanor Dorion", "Nikanor Dorion arrived in Ardan.", records=records)
    assert unknown["entity_bonus"] == 0 and unknown["entity_compatibility"] == "NO_LOCAL_ENTITY"
    conflict = entity_compatibility(name + " ambassador Ardan", name + ", the priest, died in Belmar. Torven arrived in Ardan.", records=records)
    assert conflict["entity_compatibility"] == "CONFLICT" and conflict["entity_bonus"] == -.08
    assert records == before


def test_shared_role_alone_is_ambiguous():
    records = profiles("Aelius Cassianus", roles=("ambassador", "ambassador"))
    assert resolve_query_person("Aelius Cassianus ambassador", records=records)["entity_id"] is None


def test_other_person_may_own_its_own_apposition():
    records = profiles("Torven")
    result = entity_compatibility("Torven ambassador Ardan", "Aelius Cassianus met King Torven, ambassador from Ardan.", records=records)
    assert result["entity_bonus"] == .08


@pytest.mark.parametrize("name,roles,places", [
    ("Titus Manlius", ("consul", "praetor"), ("Etruria", "Sardinia")),
    ("Lucius Cornelius Scipio", ("consul", "consul"), ("Asia", "Etruria")),
])
def test_held_out_historical_labels_with_in_memory_profiles(name, roles, places):
    records = profiles(name, roles, places)
    assert resolve_query_person(name, records=records)["entity_id"] is None
    query = f"{name} {roles[0]} {places[0]}"
    for text in (f"{name} departed for {places[0]}.",
                 f"The {roles[0]} {name} departed for {places[0]}."):
        assert entity_compatibility(query, text, records=records)["entity_bonus"] == .08
    role_only = entity_compatibility(query, f"{name}, the {roles[0]}, departed.", records=records)
    assert role_only["entity_bonus"] == (.08 if roles[0] != roles[1] else 0)
    other = f"{name} waited while Torven, the {roles[0]}, departed for {places[0]}."
    assert entity_compatibility(query, other, records=records)["entity_bonus"] == 0


def test_pictor_local_ambassador_and_priest_other_actor():
    query = "Quintus Fabius Pictor route from Delphi to Rome as ambassador"
    assert resolve_query_person("Quintus Fabius Pictor")["entity_id"] is None
    good = "Quintus Fabius Pictor, the ambassador, returned from Delphi to Rome."
    assert entity_compatibility(query, good)["entity_bonus"] == .08
    bad = "The flamen quirinalis, Quintus Fabius Pictor, died also. This year king Prusias arrived at Rome with his son Nicomedes."
    assert entity_compatibility(query, bad)["entity_bonus"] <= 0
    borrowed = "Quintus Fabius Pictor waited while Torven, the ambassador, returned from Delphi to Rome."
    assert entity_compatibility(query, borrowed)["entity_bonus"] == 0
