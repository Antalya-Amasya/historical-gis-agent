from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import (
    diversify_route_evidence,
    is_navigation_or_heading,
    is_route_or_movement_query,
    normalized_tokens,
    rerank_evidence,
)
from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retriever import ChromaHistoricalRetriever
from backend.app.rag.lexical_index import LexicalEvidenceIndex


def evidence(identifier: str, text: str, score: float, **metadata) -> Evidence:
    return Evidence(id=identifier, author="Author", work="Work", locator="section unavailable", excerpt=text, text=text, score=score, metadata={"document_id": "doc", **metadata})


def test_body_statement_outranks_explicit_contents_navigation():
    navigation = evidence("toc", "The following is contained: How the ruler was murdered (chapters 1-4).", 0.82)
    prose = evidence("body", "The ruler was slain by conspirators after a violent struggle in the senate.", 0.77)
    ranked = rerank_evidence("assassination of the ruler", [navigation, prose])
    assert [item.id for item in ranked] == ["body", "toc"]
    assert ranked[1].metadata["retrieval_ranking"]["navigation_penalty"] > 0


def test_normal_chapter_prose_is_not_navigation_and_classical_spelling_normalizes():
    prose = evidence("body", "Chapter seven: Caesar was slain by conspirators in the senate after a struggle.", 0.7)
    ranked = rerank_evidence("assassination of Cæsar", [prose])
    assert normalized_tokens("Cæsar") == {"caesar"}
    assert ranked[0].metadata["retrieval_ranking"]["navigation_penalty"] == 0
    assert ranked[0].metadata["retrieval_ranking"]["action_support"] > 0


def test_vector_score_remains_part_of_order_and_provenance_text_is_unchanged():
    stronger = evidence("stronger", "A general discussed a treaty with the senate in detail.", 0.91, source_file="a.epub", semantic_candidate=True, vector_rank=1)
    weaker = evidence("weaker", "A general discussed a treaty with the senate in detail.", 0.72, source_file="b.epub", semantic_candidate=True, vector_rank=2)
    ranked = rerank_evidence("general treaty", [weaker, stronger])
    assert [item.id for item in ranked] == ["stronger", "weaker"]
    assert ranked[0].text == stronger.text
    assert ranked[0].metadata["source_file"] == "a.epub"


class Store:
    def __init__(self): self.requested = None
    def query(self, _query, top_k, _filters):
        self.requested = top_k
        ids = [f"id-{index}" for index in range(25)]
        return {
            "ids": [ids],
            "documents": [[f"Body prose {index}" for index in range(25)]],
            "metadatas": [[{"author": "Author", "work": "Work", "document_id": "doc"} for _ in ids]],
            "distances": [[0.1 + index / 1000 for index in range(25)]],
        }


def test_bounded_candidate_pool_returns_requested_final_k():
    store = Store()
    result = ChromaHistoricalRetriever(store).retrieve("ordinary historical question", top_k=7)
    assert store.requested == 21
    assert len(result) == 7


class Corpus:
    def get(self, **_kwargs):
        return {
            "ids": ["body", "contents", "other"],
            "documents": [
                "The leader was slain by conspirators in the senate.",
                "The following is contained: how the leader was murdered.",
                "A treaty was concluded far away.",
            ],
            "metadatas": [{"author": "Author", "work": "Work", "document_id": "doc"}] * 3,
        }


def test_lexical_index_is_deterministic_normalized_and_expands_generic_actions():
    index = LexicalEvidenceIndex(Corpus())
    first = index.query("assassination of the leader", 5)
    second = index.query("assassination of the leader", 5)
    assert [item.id for item in first] == [item.id for item in second]
    assert {item.metadata["source_chunk_id"] for item in first} == {"body", "contents"}


def test_relevant_semantic_child_outranks_background_sibling():
    target = evidence("target", "The governor was murdered by conspirators after the council meeting.", .1, semantic_candidate=True, vector_rank=1)
    background = evidence("background", "The harvest was plentiful and the weather remained mild throughout the province.", .1, semantic_candidate=True, vector_rank=2)
    assert rerank_evidence("assassination of the governor", [background, target])[0].id == "target"


def test_weak_semantic_child_does_not_inherit_parent_near_one_relevance():
    weak = evidence("weak", "The weather was mild in the distant province.", .1, semantic_candidate=True, vector_rank=1)
    score = rerank_evidence("murder of the governor", [weak])[0].metadata["retrieval_ranking"]
    assert score["parent_semantic_prior"] == 1.0
    assert score["passage_local_support"] == 0.0
    assert score["semantic_relevance"] == 0.2


def test_strong_lexical_only_factual_passage_beats_weak_semantic_child():
    lexical = evidence("lexical", "The governor was murdered by conspirators in the council.", .1, lexical_candidate=True, lexical_score=20)
    semantic = evidence("semantic", "A distant harvest was noted in spring.", .1, semantic_candidate=True, vector_rank=1)
    assert rerank_evidence("assassination of the governor", [semantic, lexical])[0].id == "lexical"


def test_strong_semantic_factual_passage_beats_lexical_navigation_noise():
    semantic = evidence("semantic", "The governor was killed by conspirators in the council after a struggle.", .1, semantic_candidate=True, vector_rank=1)
    noise = evidence("noise", "The following is contained: murder of the governor.", .1, lexical_candidate=True, lexical_score=20)
    assert rerank_evidence("assassination of the governor", [noise, semantic])[0].id == "semantic"


def test_dual_channel_factual_passage_wins_with_bounded_confidence():
    dual = evidence("dual", "The governor was slain by conspirators in the council after a struggle.", .1, semantic_candidate=True, vector_rank=1, lexical_candidate=True, lexical_score=15)
    single = evidence("single", "The governor was slain by conspirators in the council after a struggle.", .1, semantic_candidate=True, vector_rank=2)
    ranked = rerank_evidence("assassination of the governor", [single, dual])
    assert ranked[0].id == "dual"
    assert ranked[0].metadata["retrieval_ranking"]["channel_confidence"] == .02


def test_missing_semantic_channel_is_neutral_not_negative():
    item = evidence("lexical", "The governor was killed by conspirators.", .1, lexical_candidate=True, lexical_score=9)
    details = rerank_evidence("murder of the governor", [item])[0].metadata["retrieval_ranking"]
    assert details["semantic_relevance"] == 0.0 and details["passage_relevance"] == details["lexical_support"]


def test_missing_lexical_channel_is_neutral_not_negative():
    item = evidence("semantic", "The governor was killed by conspirators.", .1, semantic_candidate=True, vector_rank=1)
    details = rerank_evidence("murder of the governor", [item])[0].metadata["retrieval_ranking"]
    assert details["lexical_support"] == 0.0 and details["passage_relevance"] == details["semantic_relevance"]


def test_dual_channel_confidence_is_not_additive_channel_double_counting():
    item = evidence("dual", "The governor was killed by conspirators.", .1, semantic_candidate=True, vector_rank=1, lexical_candidate=True, lexical_score=10)
    details = rerank_evidence("murder of the governor", [item])[0].metadata["retrieval_ranking"]
    assert details["passage_relevance"] == max(details["semantic_relevance"], details["lexical_support"])
    assert details["channel_confidence"] == .02


def test_lexical_raw_score_gap_changes_relevance():
    high = evidence("high", "The governor was killed by conspirators.", .1, lexical_candidate=True, lexical_score=20)
    low = evidence("low", "The governor was killed by conspirators.", .1, lexical_candidate=True, lexical_score=2)
    details = {item.id: item.metadata["retrieval_ranking"] for item in rerank_evidence("murder governor", [low, high])}
    assert details["high"]["lexical_score_relevance"] > details["low"]["lexical_score_relevance"]


def test_semantic_rank_gap_changes_prior():
    first = evidence("first", "The governor was killed by conspirators.", .1, semantic_candidate=True, vector_rank=1)
    last = evidence("last", "The governor was killed by conspirators.", .1, semantic_candidate=True, vector_rank=10)
    details = {item.id: item.metadata["retrieval_ranking"] for item in rerank_evidence("murder governor", [last, first])}
    assert details["first"]["parent_semantic_prior"] > details["last"]["parent_semantic_prior"]


def test_rank_matters_without_erasing_lexical_score_gap():
    high = evidence("high", "The governor was killed by conspirators.", .1, lexical_candidate=True, lexical_score=50)
    low = evidence("low", "The governor was killed by conspirators.", .1, lexical_candidate=True, lexical_score=1)
    details = {item.id: item.metadata["retrieval_ranking"] for item in rerank_evidence("murder governor", [high, low])}
    assert details["high"]["lexical_rank_relevance"] > details["low"]["lexical_rank_relevance"]
    assert details["high"]["lexical_support"] > details["low"]["lexical_support"]


def test_weak_lexical_tail_remains_weak():
    head = evidence("head", "The governor was killed by conspirators.", .1, lexical_candidate=True, lexical_score=100)
    tail = evidence("tail", "The governor was killed by conspirators.", .1, lexical_candidate=True, lexical_score=.01)
    details = {item.id: item.metadata["retrieval_ranking"] for item in rerank_evidence("murder governor", [head, tail])}
    assert details["tail"]["lexical_support"] < .25


def test_ties_are_deterministic_and_final_top_k_is_exact():
    items = [evidence(identifier, "A governor discussed a treaty.", .1, semantic_candidate=True, vector_rank=1) for identifier in ("c", "a", "b")]
    first = [item.id for item in rerank_evidence("governor treaty", items)]
    second = [item.id for item in rerank_evidence("governor treaty", items)]
    assert first == second and len(first) == 3


def test_channel_pool_size_change_does_not_make_weak_child_near_top():
    weak = evidence("weak", "The weather was mild.", .1, semantic_candidate=True, vector_rank=1)
    fillers = [evidence(f"f{index}", "Unrelated weather account.", .1, semantic_candidate=True, vector_rank=index + 2) for index in range(19)]
    details = rerank_evidence("murder governor", [weak, *fillers])[0].metadata["retrieval_ranking"]
    assert details["semantic_relevance"] <= .2


def test_non_overlapping_passage_identities_remain_distinct_before_suppression():
    passages = LexicalEvidenceIndex(Corpus()).query("leader slain", 5)
    assert len({item.id for item in passages}) == len(passages)


def test_classical_orthography_and_generic_violent_action_equivalence_preserved():
    item = evidence("body", "The Cæsar was slain by conspirators in the council.", .1, lexical_candidate=True, lexical_score=4)
    details = rerank_evidence("assassination of Caesar", [item])[0].metadata["retrieval_ranking"]
    assert details["entity_support"] > 0 and details["action_support"] > 0


def test_chinese_text_does_not_raise_in_ranking_path():
    item = evidence("zh", "总督在议会中被阴谋者杀害。", .1, semantic_candidate=True, vector_rank=1)
    assert len(rerank_evidence("总督遭到刺杀", [item])) == 1


def test_query_roles_separate_person_location_action_and_generic():
    spain = analyze_query("Scipio military actions in Spain")
    assert spain.person_terms == frozenset({"scipio"})
    assert spain.location_terms == frozenset({"spain"})
    assert {"military", "actions"} <= spain.generic_terms
    assert not (spain.person_terms & {"military", "actions", "spain"})
    caesar = analyze_query("assassination of Julius Caesar")
    assert caesar.person_terms == frozenset({"julius", "caesar"})
    assert caesar.action_terms == frozenset({"assassination"})
    assert "murdered" in caesar.expanded_action_terms


def test_action_only_wrong_person_does_not_match_correct_person_action_support():
    correct = evidence("correct", "Julius Caesar was slain by conspirators in the senate.", .1, lexical_candidate=True, lexical_score=8)
    wrong = evidence("wrong", "The sailors stabbed the captives after the wreck.", .1, lexical_candidate=True, lexical_score=20)
    ranked = rerank_evidence("assassination of Julius Caesar", [wrong, correct])
    details = {item.id: item.metadata["retrieval_ranking"] for item in ranked}
    assert ranked[0].id == "correct"
    assert details["correct"]["action_support"] > details["wrong"]["action_support"]
    assert details["wrong"]["person_support"] == 0


def test_generic_only_military_actions_do_not_beat_person_and_location():
    generic = evidence("generic", "These military actions were discussed in the assembly.", .1, lexical_candidate=True, lexical_score=20)
    grounded = evidence("grounded", "Scipio renewed military operations in Spain after taking New Carthage.", .1, lexical_candidate=True, lexical_score=8)
    assert rerank_evidence("Scipio military actions in Spain", [generic, grounded])[0].id == "grounded"


def test_person_plus_location_outranks_person_at_unrelated_place():
    spain = evidence("spain", "Scipio trained the army in Spain throughout the winter.", .1, lexical_candidate=True, lexical_score=10)
    egypt = evidence("egypt", "Scipio reviewed the army in Egypt throughout the winter.", .1, lexical_candidate=True, lexical_score=10)
    ranked = rerank_evidence("Scipio military actions in Spain", [egypt, spain])
    assert ranked[0].id == "spain"
    assert ranked[0].metadata["retrieval_ranking"]["location_support"] > ranked[1].metadata["retrieval_ranking"]["location_support"]


def test_epub3_nav_alone_does_not_mark_historical_prose_as_navigation():
    prose = evidence(
        "prose",
        "The consul marched at dawn and the army joined battle near the river.",
        0.5,
        navigation_source="epub3_nav",
    )
    assert not is_navigation_or_heading(prose)
    assert rerank_evidence("assassination of the consul", [prose])[0].metadata["retrieval_ranking"]["navigation_penalty"] == 0


def test_epub3_nav_with_chapter_listing_is_navigation():
    listing = evidence(
        "listing",
        "How the consul was murdered (chapters 19-22). About the burial (chapters 23-34).",
        0.5,
        navigation_source="epub3_nav",
    )
    prose = evidence("prose", "The consul was murdered in the senate after a long debate among the conspirators.", 0.5, navigation_source="epub3_nav")
    assert is_navigation_or_heading(listing)
    assert not is_navigation_or_heading(prose)
    ranked = rerank_evidence("assassination of the consul", [listing, prose])
    assert ranked[0].id == "prose"
    assert ranked[1].metadata["retrieval_ranking"]["navigation_penalty"] > 0


def test_contents_and_index_metadata_remain_navigation():
    contents = evidence("contents", "Book one names the consuls of the year.", 0.5, navigation_source="contents")
    index = evidence("index", "Book one names the consuls of the year.", 0.5, navigation_source="index")
    assert is_navigation_or_heading(contents) and is_navigation_or_heading(index)


def test_conflicting_praenomen_is_not_full_person_identity():
    julius = evidence("julius", "Julius Caesar was slain by conspirators in the senate.", .1, lexical_candidate=True, lexical_score=10)
    lucius = evidence("lucius", "Lucius Caesar was slain during the street fighting in the city.", .1, lexical_candidate=True, lexical_score=10)
    details = {item.id: item.metadata["retrieval_ranking"] for item in rerank_evidence("assassination of Julius Caesar", [lucius, julius])}
    assert details["julius"]["person_support"] > details["lucius"]["person_support"]
    assert rerank_evidence("assassination of Julius Caesar", [lucius, julius])[0].id == "julius"


def test_natural_language_what_did_scipio_do_in_spain_roles():
    roles = analyze_query("What did Scipio do in Spain?")
    assert roles.person_terms == frozenset({"scipio"})
    assert roles.location_terms == frozenset({"spain"})
    assert not (roles.person_terms & {"what", "did", "do"})


def test_caesar_crossed_the_rubicon_roles():
    roles = analyze_query("Caesar crossed the Rubicon")
    assert roles.person_terms == frozenset({"caesar"})
    assert "crossed" not in roles.person_terms
    assert "rubicon" not in roles.person_terms


def test_what_happened_at_the_battle_of_actium_roles():
    roles = analyze_query("What happened at the Battle of Actium?")
    assert roles.location_terms == frozenset({"actium"})
    assert "battle" in roles.action_terms
    assert not (roles.person_terms & {"what", "happened", "the"})


def test_tell_me_about_tiberius_gracchus_roles():
    roles = analyze_query("Tell me about Tiberius Gracchus")
    assert roles.person_terms == frozenset({"tiberius", "gracchus"})
    assert not (roles.person_terms & {"tell", "me", "about"})


def test_chinese_query_roles_do_not_raise():
    roles = analyze_query("总督遭到刺杀")
    assert roles.person_terms == frozenset()


def test_index_heading_with_epub3_nav_is_navigation():
    item = evidence("idx", "Scipio Africanus, 12.", 0.5, heading="INDEX.", navigation_source="epub3_nav")
    assert is_navigation_or_heading(item)
    assert rerank_evidence("Scipio military actions in Spain", [item])[0].metadata["retrieval_ranking"]["navigation_penalty"] > 0


def test_contents_heading_is_navigation():
    item = evidence("contents", "Book I. The consuls of the year.", 0.5, heading="CONTENTS")
    assert is_navigation_or_heading(item)
    assert rerank_evidence("Battle of Cannae", [item])[0].metadata["retrieval_ranking"]["navigation_penalty"] > 0


def test_route_diversification_prevents_one_source_family_from_consuming_budget():
    concentrated = [
        evidence(f"a-{index}", f"The army marched onward in episode {index}.", .9 - index / 100,
                 source_chunk_id="source-a", semantic_candidate=True, vector_rank=index + 1)
        for index in range(8)
    ]
    independent = [
        evidence("b", "The army crossed the river and arrived at the city.", .4, source_chunk_id="source-b", lexical_candidate=True, lexical_score=4),
        evidence("c", "The commander moved through the pass with the army.", .3, source_chunk_id="source-c", lexical_candidate=True, lexical_score=3),
    ]
    ranked = rerank_evidence("historical army route", [*concentrated, *independent])
    result = diversify_route_evidence("historical army route", ranked)
    assert {item.metadata["source_chunk_id"] for item in result[:3]} == {"source-a", "source-b", "source-c"}
    assert len({item.metadata["source_chunk_id"] for item in result[:3]}) == 3


def test_route_diversification_preserves_evidence_and_provenance_without_chronology():
    route = Evidence(
        id="route", author="Livy", work="History", locator="Book XXI",
        excerpt="The army crossed the river and entered the province.",
        text="The army crossed the river and entered the province.", score=.8,
        metadata={"document_id": "doc", "source_chunk_id": "route-source", "semantic_candidate": True, "vector_rank": 1},
    )
    unrelated = evidence("other", "The senate debated a decree.", .7, source_chunk_id="other-source",
                         semantic_candidate=True, vector_rank=2)
    result = diversify_route_evidence("军队路线", rerank_evidence("军队路线", [route, unrelated]))
    assert result[0].id == "route"
    assert result[0].author == "Livy" and result[0].work == "History"
    assert result[0].metadata["source_chunk_id"] == "route-source"
    assert "sequence" not in result[0].metadata and "chronology" not in result[0].metadata


def test_route_diversification_isolated_from_ordinary_qa_and_navigation_is_fallback():
    prose = evidence("prose", "The senate discussed a treaty with the allies.", .8, source_chunk_id="prose", semantic_candidate=True, vector_rank=1)
    navigation = evidence("navigation", "The following is contained: treaty with the allies.", .7,
                          source_chunk_id="navigation", semantic_candidate=True, vector_rank=2)
    ranked = rerank_evidence("senate treaty", [navigation, prose])
    assert not is_route_or_movement_query("senate treaty")
    assert diversify_route_evidence("senate treaty", ranked) == ranked
    route_ranked = rerank_evidence("army route", [navigation, prose])
    assert diversify_route_evidence("army route", route_ranked)[0].id == "prose"
