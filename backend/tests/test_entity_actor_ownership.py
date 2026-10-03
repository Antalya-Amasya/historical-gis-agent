"""Actor ownership is independent of case and registry coverage."""
from copy import deepcopy

import pytest

from backend.app.rag.entity_metadata import entity_compatibility, resolve_query_person
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.models import Evidence


def profiles(name, places=("Veloria", "Neralon"), roles=("ambassador", "priest")):
    return [dict(entity_id=id, canonical_name=name, entity_type="PERSON", aliases=[],
                 roles_titles=[role], context_places=[place], time_range=None)
            for id,role,place in zip(("a","b"),roles,places)]


NAMES = [("Dorieus Varro", "Nikanor Melanthes", "Veloria"),
         ("Nikanor Melanthes", "Cassius Philon", "Neralon"),
         ("Cassius Philon", "Dorieus Varro", "Galveth")]


@pytest.mark.parametrize("target,other,place", NAMES)
@pytest.mark.parametrize("case", [str, str.lower, str.upper])
@pytest.mark.parametrize("template", [
    "{target} waited while {other} departed from {place}.",
    "{target} negotiated that {other} should sail from {place}.",
    "{target} met {other}, ambassador from {place}.",
    "{target} asked {other} to return to {place}.",
    "{target} waited, {other}, the ambassador, returned to {place}.",
    "According to {other} in {place}, {target} waited.",
])
def test_other_actor_does_not_lend_context(target, other, place, case, template):
    records=profiles(target, places=(place,"Ardan"))
    result=entity_compatibility(target+" ambassador "+place,
                                case(template.format(target=target,other=other,place=place)),records=records)
    assert result["entity_bonus"] == 0 and result["entity_compatibility"] == "AMBIGUOUS"


@pytest.mark.parametrize("target,other,place", NAMES)
@pytest.mark.parametrize("template", [
    "{target} returned to {place}.",
    "{target} asked to be excused from returning to {place}.",
    "{target} told to return to {place}.",
    "{target}, the ambassador, returned to {place}.",
    "The ambassador {target} returned to {place}.",
    "{target} served as ambassador in {place}.",
    "{target} was appointed ambassador in {place}.",
    "{target} returned from Sicily to {place}.",
])
def test_same_subject_owns_proposition(target, other, place, template):
    records=profiles(target, places=(place,"Ardan"))
    result=entity_compatibility(target+" ambassador "+place,template.format(target=target,place=place),records=records)
    assert result["entity_compatibility"] == "MATCH" and result["entity_bonus"] == .08


@pytest.mark.parametrize("case", [str,str.lower,str.upper])
def test_subordinate_actor_can_own_its_own_movement(case):
    result=entity_compatibility("Nikanor Melanthes Veloria",case("Dorieus Varro negotiated that Nikanor Melanthes should sail from Veloria."),records=profiles("Nikanor Melanthes"))
    assert result["entity_bonus"] == .08


def test_exact_real_syphax_source():
    text="and in the case of syphax, who was still endeavoring to negotiate a reconciliation on the terms that scipio should sail from libya and hannibal from italy, he received his proposition not in a trustful mood, but to the end that he might ruin him."
    assert entity_compatibility("Syphax Libya",text,records=profiles("Syphax",places=("Libya","Numidia")))["entity_bonus"] == 0


@pytest.mark.parametrize("case", [str,str.lower,str.upper])
def test_caesar_does_not_own_cassius_title(case):
    text="Caesar, inasmuch as he kept in remembrance that Lucius Cassius, the consul, had been slain, and his army routed and made to pass under the yoke by the Helvetii, did not think that their request ought to be granted."
    assert entity_compatibility("Caesar consul",case(text),records=profiles("Caesar",roles=("consul","praetor")))["entity_bonus"] == 0


@pytest.mark.parametrize("case", [str,str.lower,str.upper])
def test_cotta_does_not_own_oppius_title(case):
    text="Marcus Cotta dismissed the quaestor Publius Oppius because of bribery and suspicion of conspiracy, though he himself had made great profit out of Bithynia."
    assert entity_compatibility("Marcus Cotta quaestor",case(text),records=profiles("Marcus Cotta",roles=("quaestor","praetor")))["entity_bonus"] == 0


def test_h22_repeated_explicit_marcellus():
    text="When a decision was rendered, it was to the effect that Marcellus was not guilty; that the Syracusans, however, were deserving of a certain degree of kind treatment not for their acts but for their words and supplications. As Marcellus asked to be excused from returning to Sicily, they sent Lævinus."
    assert entity_compatibility("Marcellus Sicily",text,records=profiles("Marcellus",places=("Sicily","Africa")))["entity_bonus"] == .08


def test_no_coreference_coordination_bare_or_unknown_enrichment():
    records=profiles("Dorieus Varro")
    assert resolve_query_person("Dorieus Varro",records=records)["entity_id"] is None
    for text in ("Dorieus Varro and Nikanor Melanthes went to Veloria.",
                 "Dorieus Varro was appointed ambassador. He returned to Veloria."):
        assert entity_compatibility("Dorieus Varro ambassador Veloria",text,records=records)["entity_bonus"] == 0
    unknown=entity_compatibility("Nikanor Melanthes", "Nikanor Melanthes returned to Veloria.",records=records)
    assert unknown["entity_bonus"] == 0 and unknown["entity_compatibility"] == "NO_LOCAL_ENTITY"


def test_parenthetical_place_is_not_a_person():
    text="Dorieus Varro (returning from Sicily) sailed to Africa."
    assert entity_compatibility("Dorieus Varro Africa",text,records=profiles("Dorieus Varro",places=("Africa","Numidia")))["entity_bonus"] == .08


def test_role_preposition_is_not_a_person_name():
    records=profiles("Nikanor Melanthes",roles=("mediator","praetor"))
    text="Nikanor Melanthes, who acted as mediator between them, showed consideration for Dorieus Varro."
    assert entity_compatibility("Nikanor Melanthes mediator",text,records=records)["entity_bonus"] == .08


def test_pictor_and_source_author_regression():
    q="Quintus Fabius Pictor route from Delphi to Rome as ambassador"
    assert resolve_query_person("Quintus Fabius Pictor")["entity_id"] is None
    assert entity_compatibility(q,"Quintus Fabius Pictor, the ambassador, returned from Delphi to Rome.")["entity_bonus"] == .08
    assert entity_compatibility(q,"The flamen quirinalis, Quintus Fabius Pictor, died also. This year king Prusias arrived at Rome with his son Nicomedes.")["entity_bonus"] == -.08
    assert entity_compatibility(q,"Fabius Pictor, a near kinsman to Maximus, was sent to consult the oracle of Delphi.")["entity_bonus"] == 0


def test_defaults_registry_miss_and_authority_immutable(monkeypatch):
    import backend.app.rag.entity_metadata as entity
    records=profiles("Dorieus Varro");before=deepcopy(records)
    monkeypatch.setattr(entity,"person_records",lambda:records)
    item=Evidence(id="local",author="Livy",work="History",locator="local",text="Dorieus Varro asked to be excused from returning to Veloria.",excerpt="local",score=0,metadata=dict(actor="UNKNOWN",movement="UNKNOWN",endpoints="UNKNOWN",completion=False,chronology="UNKNOWN",episode="UNKNOWN",travel_mode="UNKNOWN",route_admission="UNKNOWN",lexical_candidate=True,lexical_score=10))
    original=deepcopy(item.model_dump())
    default=rerank_evidence("Dorieus Varro Veloria",[item])[0]
    assert default.model_dump()==rerank_evidence("Dorieus Varro Veloria",[item],entity_metadata=False)[0].model_dump()
    shadow=rerank_evidence("Dorieus Varro Veloria",[item],entity_metadata=True)[0]
    assert item.model_dump()==original and records==before
    for key,value in item.metadata.items():assert shadow.metadata[key]==value
    unknown="Nikanor Melanthes returned to Veloria"
    assert rerank_evidence(unknown,[item])[0].score==rerank_evidence(unknown,[item],entity_metadata=True)[0].score
