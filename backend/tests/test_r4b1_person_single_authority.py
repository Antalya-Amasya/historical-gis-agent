from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence


QUERY = "Trace Commander Alpha's campaign against Commander Beta from Gamma Harbor to Delta Bay."


def _item(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="1",
        excerpt=text,
        text=text,
        score=0.5,
        metadata={"document_id": "doc", "semantic_candidate": True, "vector_rank": 5},
    )


def _rank(text: str, query: str = QUERY) -> dict:
    ranked = rerank_evidence(query, [_item("x", text)], pool_relative=False)
    return ranked[0].metadata["retrieval_ranking"]


def test_explicit_actor_outranks_context_only_name():
    actor = _rank("Commander Alpha marched from Gamma Harbor to Delta Bay.")
    context = _rank("The garrison at Gamma Harbor awaited Commander Alpha after the council.")
    assert actor["primary_subject_support"] >= 0.08
    assert context["primary_subject_support"] <= 0.04
    assert actor["final_score"] > context["final_score"]


def test_person_signal_contributes_once_to_final_score():
    with_person = _rank("Commander Alpha marched from Gamma Harbor to Delta Bay.")
    without_person = _rank("The army marched from Gamma Harbor to Delta Bay.")
    delta = with_person["final_score"] - without_person["final_score"]
    duplicate = with_person["entity_support"] + with_person["joint_support"]
    assert with_person["primary_subject_support"] >= 0.08
    assert without_person["primary_subject_support"] == 0.0
    assert delta > 0
    assert delta + 1e-9 >= duplicate or duplicate == 0.0
    # Person must not be added again as independent entity/joint authority.
    assert with_person["final_score"] + 1e-9 >= (
        with_person["passage_relevance"]
        + with_person["channel_confidence"]
        + with_person["location_support"]
        + with_person["action_support"]
        + with_person["generic_support"]
        + with_person["statement_bonus"]
        - with_person["navigation_penalty"]
    )
    reconstructed = (
        with_person["passage_relevance"]
        + with_person["channel_confidence"]
        + with_person["location_support"]
        + with_person["action_support"]
        + with_person["generic_support"]
        + with_person["statement_bonus"]
        - with_person["navigation_penalty"]
    )
    assert with_person["final_score"] == reconstructed


def test_movement_and_location_remain_without_person():
    score = _rank("The army marched from Gamma Harbor to Delta Bay.")
    assert score["primary_subject_support"] == 0.0
    assert score["action_support"] > 0
    assert score["location_support"] > 0
    assert score["final_score"] > 0


def test_opponent_name_with_other_explicit_actor_is_not_mover_support():
    score = _rank("Commander Beta marched from Gamma Harbor to Delta Bay while Commander Alpha remained in camp.")
    assert score["primary_subject_support"] < 0.08
    actor = _rank("Commander Alpha marched from Gamma Harbor to Delta Bay.")
    assert actor["final_score"] > score["final_score"]


def test_pronoun_subject_does_not_invent_person_authority():
    score = _rank("He marched from Gamma Harbor to Delta Bay.")
    assert score["primary_subject_support"] == 0.0
    assert score["person_support"] < 0.08


def test_input_order_does_not_change_scores():
    a = _item("a", "Commander Alpha marched from Gamma Harbor to Delta Bay.")
    b = _item("b", "The army marched from Gamma Harbor to Delta Bay.")
    first = rerank_evidence(QUERY, [a, b], pool_relative=False)
    second = rerank_evidence(QUERY, [b, a], pool_relative=False)
    by_first = {item.id: item.metadata["retrieval_ranking"]["final_score"] for item in first}
    by_second = {item.id: item.metadata["retrieval_ranking"]["final_score"] for item in second}
    assert by_first == by_second
    assert [item.id for item in first] == [item.id for item in second]
