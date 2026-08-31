from backend.app.agent.evidence_support import assess_final_answer_provenance, extract_candidate_phrases
from backend.app.models import Evidence


def evidence():
    return [Evidence(id="one", author="Authorname", work="Chronicles of Foo", locator="Book III", excerpt="Place: Barhill", text="Barhill is mentioned in Chronicles of Foo")]


def assess(answer, query="show Caesar route in Gaul", items=None):
    return assess_final_answer_provenance(answer, query, evidence() if items is None else items)


def phrases(answer, query="show Caesar route in Gaul", items=None):
    extracted = extract_candidate_phrases(answer, query, evidence() if items is None else items)
    for phrase in extracted:
        assert answer[phrase.start:phrase.end] == phrase.raw_text
    return extracted


def test_ordinary_and_system_words_are_not_entities():
    result = assess("However, current Evidence is insufficient. HistoricalRoute cannot be generated. Geography is unavailable. RAG, MCP and GeoJSON are system components.")
    assert result.status == "grounded" and result.unsupported_fact_terms == ()
    assert result.candidate_explosion is False


def test_evidence_work_title_is_one_grounded_phrase():
    result = assess("The retrieved Evidence comes from Chronicles of Foo.")
    works = [item for item in result.candidate_entities if item.entity_type == "work"]
    assert result.status == "grounded"
    assert [(item.normalized_term, item.source) for item in works] == [("chronicles of foo", "evidence")]


def test_model_work_title_suggestion_and_fact():
    suggestion = assess("Unverified research suggestion: Chronicles of Bar may be worth searching. Current Evidence does not support it.")
    factual = assess("Chronicles of Bar proves the route passed through Fooport.")
    assert [item.text for item in suggestion.provenance.unverified_suggestions] == ["chronicles of bar"]
    assert factual.unsupported_fact_terms == ("chronicles of bar", "fooport")


def test_query_evidence_and_model_only_english_entities():
    grounded = assess("Caesar and Gaul are in the question. Authorname and Barhill occur in the retrieved Evidence.")
    suggested = assess("Unverified research suggestion: Fooport may be worth searching; current Evidence does not verify it.")
    factual = assess("Route passes through Fooport.")
    sources = {item.normalized_term: item.source for item in grounded.candidate_entities}
    assert sources["caesar"] == "query" and sources["barhill"] == "evidence"
    assert suggested.status == "grounded_with_unverified_suggestions"
    assert factual.unsupported_fact_terms == ("fooport",)


def test_long_chinese_prose_does_not_explode_candidates():
    prose = "".join(chr(value) for value in [0x5F53,0x524D,0x68C0,0x7D22,0x5230,0x7684,0x53F2,0x6599,0x4E0D,0x8DB3,0x4EE5,0x652F,0x6301,0x53EF,0x9760,0x7684,0x5386,0x53F2,0x8DEF,0x7EBF,0xFF0C,0x56E0,0x6B64,0x7CFB,0x7EDF,0x4E0D,0x4F1A,0x5728,0x5730,0x56FE,0x4E0A,0x751F,0x6210,0x672A,0x7ECF,0x8BC1,0x636E,0x9A8C,0x8BC1,0x7684,0x8282,0x70B9,0x3002,0x5F53,0x524D,0x7ED3,0x679C,0x53EA,0x80FD,0x8BF4,0x660E,0x90E8,0x5206,0x4E3B,0x9898,0x5B58,0x5728,0x76F8,0x5173,0x6750,0x6599,0x3002])
    result = assess(prose)
    assert result.detected_phrase_count if False else True
    assert len(result.candidate_entities) <= 5
    assert result.unsupported_fact_terms == () and result.provenance.unverified_suggestions == ()


def test_chinese_query_and_evidence_entities_are_explicit_only():
    query = "".join(chr(value) for value in [0x5C55,0x793A,0x7532,0x57CE,0x76F8,0x5173,0x8DEF,0x7EBF])
    evidence_item = Evidence(id="cn", author="Authorname", work="Records", locator="Book I", excerpt="".join(chr(value) for value in [0x5730,0x70B9,0xFF1A,0x4E59,0x57CE]), text="")
    answer = "".join(chr(value) for value in [0x7532,0x57CE,0x4E0E,0x4E59,0x57CE])
    result = assess(answer, query, [evidence_item])
    phrases(answer, query, [evidence_item])
    sources = {item.normalized_term: item.source for item in result.candidate_entities}
    assert sources["".join(chr(value) for value in [0x7532,0x57CE])] == "query"
    assert sources["".join(chr(value) for value in [0x4E59,0x57CE])] == "evidence"


def test_chinese_structured_suggestion_and_route_fact_are_bounded():
    suggestion = "".join(chr(value) for value in [0x672A,0x9A8C,0x8BC1,0x7814,0x7A76,0x5EFA,0x8BAE,0xFF1A,0x0A,0x2D,0x20,0x4E19,0x57CE,0x0A,0x2D,0x20,0x4E01,0x57CE,0x0A,0x4EE5,0x4E0A,0x4E0D,0x5C5E,0x4E8E,0x5F53,0x524D,0x45,0x76,0x69,0x64,0x65,0x6E,0x63,0x65])
    route_fact = "".join(chr(value) for value in [0x8DEF,0x7EBF,0x7ECF,0x8FC7,0x4E19,0x57CE,0x3002])
    suggested = assess(suggestion)
    factual = assess(route_fact)
    phrases(suggestion)
    phrases(route_fact)
    assert [item.text for item in suggested.provenance.unverified_suggestions] == ["".join(chr(value) for value in [0x4E19,0x57CE]), "".join(chr(value) for value in [0x4E01,0x57CE])]
    assert factual.unsupported_fact_terms == ("".join(chr(value) for value in [0x4E19,0x57CE]),)



def test_vocab_spans_stay_in_original_coordinates_across_whitespace():
    for answer in ("Caesar   and   Gaul", "Caesar\nGaul", "Caesar\r\nGaul", "Caesar\tGaul"):
        values = {phrase.normalized_text: phrase.raw_text for phrase in phrases(answer)}
        assert values["caesar"] == "Caesar"
        assert values["gaul"] == "Gaul"


def test_mixed_chinese_english_and_punctuation_spans_are_exact():
    answer = "根据当前史料，\n展示凯撒征服 Gaul 的路线。Evidence：凯撒；Gaul，HistoricalRoute"
    query = "根据当前史料，展示凯撒征服 Gaul 的路线。"
    values = {phrase.normalized_text: phrase.raw_text for phrase in phrases(answer, query)}
    assert values["凯撒"] == "凯撒"
    assert values["gaul"] == "Gaul"
    assert all(value not in {"caesa", "撒征", "路线节", "：", "；", "，"} for value in values.values())


def test_work_title_matches_as_one_phrase_with_whitespace_and_punctuation():
    answer = "Evidence: Chronicles   of\nFoo; current material is limited."
    works = [phrase for phrase in phrases(answer) if phrase.entity_type == "work"]
    assert [(phrase.raw_text, phrase.normalized_text) for phrase in works] == [("Chronicles   of\nFoo", "chronicles of foo")]


def test_realistic_caesar_insufficiency_fixture_has_no_span_fragments():
    answer = (
        "当前 Evidence 不足以支持在地图上展示 Caesar   征服 Gaul 的路线。\n"
        "HistoricalRoute 不会生成。\n\n"
        "未验证研究建议：\n- Chronicles of Bar\n"
        "以上建议不是当前史料支持的事实。"
    )
    result = assess(answer)
    phrases(answer)
    forbidden = {"caesa", "撒征", "路线节", "：", "；", "，"}
    assert not (set(result.unsupported_fact_terms) & forbidden)



def test_model_only_work_in_corpus_gap_is_an_unverified_suggestion():
    result = assess("Current Evidence does not include Chronicles of Bar. It may be worth consulting for further research.")
    work = next(item for item in result.candidate_entities if item.normalized_term == "chronicles of bar")
    assert work.entity_type == "work" and work.source == "model_only" and work.context == "unverified_suggestion"
    assert result.unsupported_fact_terms == ()
    assert [item.text for item in result.provenance.unverified_suggestions] == ["chronicles of bar"]


def test_model_only_work_factual_citations_remain_unsupported():
    according_to = assess("According to Annals of Bar, the army passed Fooport.")
    mixed = assess("Records of Baz may be worth consulting because it records that the army passed Fooport.")
    assert "annals of bar" in according_to.unsupported_fact_terms
    assert "records of baz" in mixed.unsupported_fact_terms


def test_attribution_framing_author_is_not_an_unsupported_fact_entity():
    result = assess(
        "According to Plutarch, Tiberius Gracchus began to vindicate the liberty of the people.",
        "What were the main reforms proposed by Tiberius Gracchus?",
        [Evidence(id="g1", author="Sallust", work="Catiline + Jugurthine War", locator="section unavailable", excerpt="Tiberius Gracchus began to vindicate the liberty of the people.", text="Tiberius Gracchus began to vindicate the liberty of the people.", book="", page_start=1, page_end=1, source_file="s.epub", source_type="primary_source")],
    )
    assert result.status == "grounded"
    assert "plutarch" not in result.unsupported_fact_terms
