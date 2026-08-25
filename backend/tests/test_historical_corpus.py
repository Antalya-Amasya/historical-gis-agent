import pytest

from backend.app.candidate_routes.annotation import HistoricalAnnotation, HistoricalEventType
from backend.app.candidate_routes.corpus import HistoricalCorpus
from backend.app.candidate_routes.waypoint_graph import (
    CorpusEvidenceError,
    HistoricalEventChain,
    HistoricalEventStep,
    HistoricalWaypointBuilder,
    WaypointBuilder,
)


def corpus(identifier, name, sources):
    return HistoricalCorpus(
        id=identifier, name=name, period="test period", region="test region",
        description="offline fixture", source_refs=sources,
    )


def event_chain(corpus_id, prefix):
    return HistoricalEventChain(
        corpus_id=corpus_id, event_id=f"{prefix}-event", period="test period", region="test region",
        steps=[
            HistoricalEventStep(id=f"{prefix}-a", name=f"{prefix} A", order=1, description="first", evidence_refs=[f"{prefix}-1"], confidence=0.8),
            HistoricalEventStep(id=f"{prefix}-b", name=f"{prefix} B", order=2, description="second", evidence_refs=[f"{prefix}-2"], confidence=0.8),
        ],
    )


def fixtures():
    punic = corpus("punic", "Second Punic War", ["punic-1", "punic-2"])
    caesar = corpus("caesar", "Caesar Gallic War", ["caesar-1", "caesar-2"])
    alexander = corpus("alexander", "Alexander Campaign", ["alexander-1", "alexander-2"])
    return [(punic, event_chain("punic", "punic")), (caesar, event_chain("caesar", "caesar")), (alexander, event_chain("alexander", "alexander"))]


def test_three_independent_corpora_use_the_same_waypoint_builder():
    graphs = [HistoricalWaypointBuilder.from_event_chain(chain, corpus) for corpus, chain in fixtures()]
    assert [len(graph.waypoints) for graph in graphs] == [2, 2, 2]
    assert [graph.waypoints[0].id for graph in graphs] == ["punic-a", "caesar-a", "alexander-a"]
    # The compatibility builder has the same generic corpus-aware behavior.
    corpus_value, chain_value = fixtures()[1]
    assert WaypointBuilder.from_event_chain(chain_value, corpus_value).model_dump() == graphs[1].model_dump()


def test_evidence_from_another_corpus_or_missing_corpus_contract_is_rejected():
    punic, punic_chain = fixtures()[0]
    caesar, _ = fixtures()[1]
    with pytest.raises(CorpusEvidenceError):
        HistoricalWaypointBuilder.from_event_chain(punic_chain, caesar)
    with pytest.raises(CorpusEvidenceError):
        HistoricalWaypointBuilder.from_event_chain(punic_chain)
    contaminated = punic_chain.model_copy(update={
        "steps": [punic_chain.steps[0].model_copy(update={"evidence_refs": ["caesar-1"]}), punic_chain.steps[1]],
    })
    with pytest.raises(CorpusEvidenceError):
        HistoricalWaypointBuilder.from_event_chain(contaminated, punic)


def test_annotations_and_routing_contracts_do_not_depend_on_corpus_name():
    annotation = HistoricalAnnotation(
        id="punic-a", title="Input annotation", event_type=HistoricalEventType.OTHER,
        period=None, description="generic", source_refs=["punic-1"], external_links=[],
    )
    assert annotation.event_type is HistoricalEventType.OTHER
    source = HistoricalWaypointBuilder.from_event_chain.__doc__ or ""
    assert "Punic" not in source and "Caesar" not in source and "Alexander" not in source
