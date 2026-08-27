from backend.app.rag.evaluation.chinese_query_bridge_experiment import transform, validate_lexicon


LEXICON = [
    {"chinese_form": "凯撒", "english_forms": ["Caesar"], "type": "person", "evidence_source": "fixture", "reason": "actor"},
    {"chinese_form": "渡过", "english_forms": ["crossing"], "type": "concept", "evidence_source": "fixture", "reason": "action"},
]


def test_bridge_is_deterministic_and_keeps_original_query():
    one = transform("凯撒渡过卢比孔河", "Caesar crossing the Rubicon", LEXICON, "STRUCTURED_BRIDGE")
    two = transform("凯撒渡过卢比孔河", "Caesar crossing the Rubicon", LEXICON, "STRUCTURED_BRIDGE")
    assert one == two
    assert one[0].startswith("凯撒渡过卢比孔河 | entities: Caesar | concepts: crossing")


def test_lexicon_rejects_document_ids_chunk_ids_and_author_injection():
    bad = [
        {"chinese_form": "坏", "english_forms": ["appian_roman_history_civil_wars"], "type": "person", "evidence_source": "fixture", "reason": "bad"},
        {"chinese_form": "更坏", "english_forms": ["a7abe1d61880c569f79e4470e05a3bb8"], "type": "person", "evidence_source": "fixture", "reason": "bad"},
        {"chinese_form": "作者", "english_forms": ["Appian"], "type": "person", "evidence_source": "fixture", "reason": "bad"},
    ]
    assert len(validate_lexicon(bad)) == 3
