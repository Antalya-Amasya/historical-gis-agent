"""Deterministic, bounded retrieval-query bridge for Roman Republic Chinese queries."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BRIDGE_VERSION = "roman_republic_concept_v1"
# Frozen in the accepted V1 artifact order. V2 expansion mappings are intentionally absent.
V1_ENTRIES = (
    ("凯撒", ("Caesar", "Julius Caesar"), "person"), ("庞培", ("Pompey",), "person"), ("汉尼拔", ("Hannibal",), "person"), ("马略", ("Marius",), "person"), ("苏拉", ("Sulla",), "person"), ("喀提林", ("Catiline",), "person"), ("朱古达", ("Jugurtha",), "person"), ("阿尔卑斯山", ("Alps",), "place"), ("卢比孔河", ("Rubicon",), "place"), ("高卢", ("Gaul",), "place"), ("坦尼", ("Cannae",), "place"), ("第二次布匏战争", ("Second Punic War",), "event_entity"), ("内战", ("civil war",), "concept"), ("阴谋", ("conspiracy",), "concept"), ("战争", ("war",), "concept"), ("会战", ("battle",), "concept"), ("渡过", ("crossing",), "concept"), ("翻越", ("crossing",), "concept"), ("军事行动", ("campaign",), "concept"), ("冲突", ("conflict",), "concept"),
)


@dataclass(frozen=True)
class QueryBridgeResult:
    original_query: str
    retrieval_query: str
    applied: bool
    matched_entries: tuple[str, ...]
    bridge_version: str | None
    failure: str | None = None


class HistoricalQueryBridge:
    def __init__(self, enabled: bool = True, entries=V1_ENTRIES):
        self.enabled, self.entries = enabled, tuple(entries)

    def transform(self, original_query: str) -> QueryBridgeResult:
        if not self.enabled or not original_query or not any("\u4e00" <= char <= "\u9fff" for char in original_query):
            return QueryBridgeResult(original_query, original_query, False, (), None)
        try:
            entity_forms, concept_forms, matched = [], [], []
            for chinese_form, english_forms, entry_type in self.entries:
                if chinese_form in original_query:
                    target = concept_forms if entry_type in {"concept", "event_entity"} else entity_forms
                    for form in english_forms:
                        if form not in target:
                            target.append(form)
                    matched.append(chinese_form)
            if not matched:
                return QueryBridgeResult(original_query, original_query, False, (), None)
            retrieval_query = " ".join([original_query, *entity_forms, *concept_forms])
            return QueryBridgeResult(original_query, retrieval_query, True, tuple(matched), BRIDGE_VERSION)
        except Exception as exc:  # bounded, observable fallback; retrieval remains available
            logger.warning("historical_query_bridge_fallback error_type=%s", type(exc).__name__)
            return QueryBridgeResult(original_query, original_query, False, (), BRIDGE_VERSION, type(exc).__name__)
