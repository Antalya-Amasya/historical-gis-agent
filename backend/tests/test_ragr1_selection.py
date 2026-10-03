"""Provider-free contracts for hybrid fusion, overlap quality, and provenance."""
from copy import deepcopy
import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.app.rag.query_bridge import HistoricalQueryBridge


def ev(identifier, text, **metadata):
    return Evidence(id=identifier, author="Primary author", work="Primary work",
                    locator="section unavailable", excerpt=text[:500], text=text,
                    score=0.0, metadata={"source_chunk_id": identifier, **metadata})


def select(query, items, k=10, diagnostics=None):
    return ChromaHistoricalRetriever(None)._finalize_selection(query, items, k, diagnostics=diagnostics)


def test_semantic_rank_one_survives_unsupported_literal_name_distractors():
    # A source-level alias signal can survive without adding an alias registry.
    target = ev("target", "Pompey himself hastened to Cilicia with forces.",
                semantic_candidate=True, vector_rank=1, distance=.295)
    distractors = [ev(f"consul-{i}", "Pompeius was consul during another campaign.",
                     lexical_candidate=True, lexical_score=20) for i in range(20)]
    east = ev("east", "The east was quiet.", lexical_candidate=True, lexical_score=1)
    result = select("Pompeius campaign in the east", [*distractors, east, target])
    assert result[0].id == "target"
    assert result[0].metadata["retrieval_ranking"]["semantic_vector_bonus"] > 0


def test_setovia_lexical_hit_beats_many_generic_semantic_candidates():
    target = ev("setovia", "While he was besieging the city Setovia, he prevented the force from entering.",
                lexical_candidate=True, lexical_score=30)
    generic = [ev(f"generic-{i}", "The army discussed distant campaigns and city walls.",
                  semantic_candidate=True, vector_rank=i+1, distance=.01) for i in range(20)]
    assert select("Setovia besieged", [*generic, target], 5)[0].id == "setovia"


def test_semantic_and_lexical_agreement_has_bounded_benefit():
    lexical = ev("lex", "Neralis Vexon marched into Ardan.", lexical_candidate=True, lexical_score=20)
    dual = lexical.model_copy(update={"id": "dual", "metadata": {
        **lexical.metadata, "semantic_candidate": True, "vector_rank": 1, "distance": .25,
    }})
    ranked = rerank_evidence("Neralis Vexon marched into Ardan", [lexical, dual])
    assert ranked[0].id == "dual"
    without_distance=dual.model_copy(update={"metadata": {k:v for k,v in dual.metadata.items() if k!="distance"}})
    uncalibrated=rerank_evidence("Neralis Vexon marched into Ardan",[lexical,without_distance])
    prior_score=next(e.score for e in uncalibrated if e.id=="dual")
    assert ranked[0].score > ranked[1].score
    assert 0 < ranked[0].score-prior_score <= .321


def test_lexical_only_exact_actor_remains_competitive():
    exact = ev("exact", "Marcus marched from Rome to Capua with his army.",
               lexical_candidate=True, lexical_score=20)
    wrong = ev("wrong", "Brutus marched from Rome to Capua with his army.",
               semantic_candidate=True, vector_rank=1, distance=.01)
    assert select("Marcus marched from Rome to Capua", [wrong, exact], 1)[0].id == "exact"


@pytest.mark.parametrize("reverse", [False, True])
def test_overlap_winner_is_quality_based_before_directional_diversity(reverse):
    strong = ev("strong", "Caesar crossed Gaul with his army. Caesar marched through Gaul and crossed the river with his soldiers after a long campaign.",
                semantic_candidate=True, vector_rank=1, distance=.1,
                source_chunk_id="shared", passage_start=0, passage_end=120)
    weak = ev("weak", "Caesar marched from the camp to the town.",
              semantic_candidate=True, vector_rank=2, distance=.5,
              source_chunk_id="shared", passage_start=0, passage_end=80)
    trace = []
    items = [weak, strong] if reverse else [strong, weak]
    assert select("Caesar route in Gaul", items, 1, trace)[0].id == "strong"
    assert next(x for x in trace if x["candidate_id"] == "weak")["selection_reason"] == "OVERLAP_EVICTION"


def test_same_source_distinct_statements_survive_and_exact_spans_are_deduplicated():
    a = ev("a", "Marcus marched from Rome to Capua.", lexical_candidate=True, lexical_score=20,
           source_chunk_id="same", passage_start=0, passage_end=40)
    b = ev("b", "Marcus marched from Capua to Naples.", lexical_candidate=True, lexical_score=20,
           source_chunk_id="same", passage_start=40, passage_end=80)
    duplicate = a.model_copy(update={"id": "duplicate"})
    trace=[]
    result=select("Marcus marched", [duplicate, b, a], 5, trace)
    assert {x.id for x in result} == {"a", "b"}
    assert next(x for x in trace if x["candidate_id"] == "duplicate")["selection_reason"] == "EXACT_DUPLICATE"


def test_unseen_multi_token_person_keeps_exact_evidence():
    target=ev("unseen", "Neralis Vexon marched into Ardan with his army.",
              semantic_candidate=True, vector_rank=2, distance=.3)
    unrelated=ev("other", "Brutus marched into Ardan with his army.",
                 semantic_candidate=True, vector_rank=1, distance=.01)
    assert select("Neralis Vexon marched into Ardan", [unrelated,target], 1)[0].id == "unseen"


@pytest.mark.parametrize("nominal,verb", [("entry","entering"),("arrival","arriving"),("departure","departing")])
def test_nominal_movement_ranking_does_not_treat_prepositions_as_people(nominal,verb):
    target=ev("negative", f"The force was prevented from {verb} the city.",
              lexical_candidate=True, lexical_score=20)
    ranked=rerank_evidence(f"{nominal} into the city was prevented", [target])
    ranking=ranked[0].metadata["retrieval_ranking"]
    assert ranking["person_support"] == .08
    assert ranking["action_support"] == .12


def test_negative_entry_survives_positive_entry_distractors():
    target=ev("prevented", "The force was prevented from entering the city.",
              lexical_candidate=True, lexical_score=20)
    positive=[ev(f"positive-{i}", "He made his entry into the city with his army.",
                 lexical_candidate=True, lexical_score=30) for i in range(20)]
    result=select("entry into the city was prevented", [*positive,target], 5)
    assert result[0].id == "prevented"


def test_wrong_actor_ranking_remains_downstream_rejectable():
    from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
    wrong=ev("wrong", "Brutus marched from Rome to Capua.",
             semantic_candidate=True, vector_rank=1, distance=.1)
    result=select("Marcus marched from Rome to Capua", [wrong])
    actor=EvidenceGroundedHistoricalEventExtractor._ground_movement_actor(result[0].text)
    assert actor.actor_text == "Brutus"
    assert actor.actor_text != "Marcus"


def test_vector_similarity_is_monotonic_bounded_and_not_recovered_from_rerank_score():
    items=[ev(str(i), "Unrelated historical prose.", semantic_candidate=True,
              vector_rank=i+1, distance=d) for i,d in enumerate([.5,.3,.25,.01])]
    ranked=rerank_evidence("unregistered subject",items)
    bonuses={e.id:e.metadata["retrieval_ranking"]["semantic_vector_bonus"] for e in ranked}
    assert 0 == bonuses["0"] < bonuses["1"] < bonuses["2"] <= bonuses["3"] <= .32
    again=rerank_evidence("unregistered subject",ranked)
    assert {e.id:e.metadata["retrieval_ranking"]["semantic_vector_bonus"] for e in again} == bonuses
    missing=ev("missing", "Unrelated prose.",semantic_candidate=True,vector_rank=1)
    assert rerank_evidence("unregistered subject",[missing])[0].metadata["retrieval_ranking"]["vector_similarity"] is None


def test_candidate_observability_includes_losses_without_normal_response_payload():
    target=ev("target", "Marcus marched from Rome to Capua.",lexical_candidate=True,lexical_score=20,merged_rank=1)
    noise=ev("noise", "Unrelated distant weather.",semantic_candidate=True,vector_rank=1,distance=.5,merged_rank=2)
    other=ev("other", "Marcus marched from Rome to Naples.",lexical_candidate=True,lexical_score=1,merged_rank=3)
    trace=[];result=select("Marcus marched",[noise,other,target],1,trace)
    assert len(trace)==3
    assert next(x for x in trace if x["candidate_id"]=="noise")["selection_reason"]=="LOW_RERANK_SCORE"
    assert next(x for x in trace if x["candidate_id"]=="other")["selection_reason"]=="TOP_K_TRUNCATION"
    assert trace[0]["ranking"]["pre_rerank_score"] == 0
    assert "diagnostics" not in result[0].metadata
    assert result[0].metadata["selection_status"] == "SELECTED"


def test_ranking_never_mutates_authority_or_source_metadata():
    item=ev("source", "Marcus was prevented from entering Capua.",semantic_candidate=True,
            vector_rank=1,distance=.25,actor_authority="UNRESOLVED",completed_movement=False,
            historical_mode="UNKNOWN",chronology="UNRESOLVED",route_membership="UNKNOWN")
    before=deepcopy(item.model_dump())
    result=select("Marcus entry into Capua",[item],1)[0]
    assert item.model_dump()==before
    for key in ("actor_authority","completed_movement","historical_mode","chronology","route_membership"):
        assert result.metadata[key]==item.metadata[key]
    assert result.text==item.text


class QueryOnlyStore:
    def __init__(self):self.queries=[]
    def query(self,q,k,filters):
        self.queries.append(q)
        return {"ids":[["source"]],"documents":[["Caesar marched into Gaul."]],
                "metadatas":[[{"author":"Caesar","work":"War"}]],"distances":[[.1]]}
    def __getattr__(self,name):
        if name in {"add","update","upsert","delete","reset","rebuild"}:
            raise AssertionError("retrieval attempted corpus mutation")
        raise AttributeError(name)


def test_bilingual_bridge_preserves_latin_aliases_and_query_only_store():
    store=QueryOnlyStore();retriever=ChromaHistoricalRetriever(store,HistoricalQueryBridge())
    zh=retriever.retrieve("凯撒的行军路线",1)
    en=retriever.retrieve("Caesar march route",1)
    assert zh[0].text==en[0].text
    assert "caesar" in store.queries[0].lower()
    assert zh[0].metadata["pre_rerank_score"]>0
    assert zh[0].metadata["merged_rank"]==1


def test_navigation_cannot_gain_a_vector_bonus():
    nav=ev("nav", "The following is contained: how the army moved.",semantic_candidate=True,vector_rank=1,distance=.01)
    ranking=rerank_evidence("army route",[nav])[0].metadata["retrieval_ranking"]
    assert ranking["semantic_vector_bonus"]==0
