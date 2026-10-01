"""Bounded retrieval-only expansion from production entity aliases."""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BRIDGE_VERSION = "production_aliases_v2"
# Legacy evaluation compatibility only. Neither retrieval nor grounding consumes
# this inventory; production identity aliases live in registries/subject_aliases.json.
V1_ENTRIES = (
    ("凯撒", ("Caesar", "Julius Caesar"), "person"), ("庞培", ("Pompey",), "person"), ("汉尼拔", ("Hannibal",), "person"), ("马略", ("Marius",), "person"), ("苏拉", ("Sulla",), "person"), ("喀提林", ("Catiline",), "person"), ("朱古达", ("Jugurtha",), "person"), ("阿尔卑斯山", ("Alps",), "place"), ("卢比孔河", ("Rubicon",), "place"), ("高卢", ("Gaul",), "place"), ("坦尼", ("Cannae",), "place"), ("第二次布匏战争", ("Second Punic War",), "event_entity"), ("内战", ("civil war",), "concept"), ("阴谋", ("conspiracy",), "concept"), ("战争", ("war",), "concept"), ("会战", ("battle",), "concept"), ("渡过", ("crossing",), "concept"), ("翻越", ("crossing",), "concept"), ("军事行动", ("campaign",), "concept"), ("冲突", ("conflict",), "concept"), ("路线", ("route", "march", "movement"), "concept"), ("行军", ("march", "movement"), "concept"),
)


@dataclass(frozen=True)
class QueryBridgeResult:
    original_query: str
    retrieval_query: str
    applied: bool
    matched_entries: tuple[str, ...]
    bridge_version: str | None
    failure: str | None = None


@lru_cache(maxsize=1)
def _registry_entries():
    from backend.app.agent.evidence_support import SUBJECT_ALIASES
    from backend.app.geography.place_registry import records, physical_records
    from backend.app.rag.campaign_ontology import HistoricalCampaignOntology

    groups = [(name, forms, "entity") for name, forms in SUBJECT_ALIASES.items()]
    groups.extend((r["canonical_name"], r["aliases"], "place") for r in [*records(), *physical_records()])
    groups.extend((e.title, e.aliases, "event_entity") for e in HistoricalCampaignOntology.default().entities)
    entries = []
    for canonical, aliases, kind in groups:
        # Only identity labels are used: no match_terms, evidence flags, dates,
        # coordinates, route contexts or source references become query hints.
        english = tuple(dict.fromkeys(form for form in [canonical, *aliases]
                                     if not any("\u4e00" <= c <= "\u9fff" for c in form)))[:4]
        for alias in dict.fromkeys([canonical, *aliases]):
            entries.append((alias, english, kind))
    return tuple(entries)


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class HistoricalQueryBridge:
    def __init__(self, enabled: bool = True, entries=None):
        self.enabled = enabled
        self.entries = None if entries is None else tuple(entries)

    def transform(self, original_query: str) -> QueryBridgeResult:
        if not self.enabled or not original_query:
            return QueryBridgeResult(original_query, original_query, False, (), None)
        try:
            query = _normalized(original_query)
            entries = _registry_entries() if self.entries is None else self.entries
            forms, matched, seen = [], [], set()
            for alias, english_forms, _kind in entries:
                needle = _normalized(alias)
                if not needle:
                    continue
                chinese = any("\u4e00" <= c <= "\u9fff" for c in needle)
                pattern = re.escape(needle) if chinese else rf"(?<!\w){re.escape(needle)}(?!\w)"
                if not re.search(pattern, query):
                    continue
                for form in english_forms:
                    if len(forms) == 16:
                        break
                    # Do not repeat labels already present in the original query.
                    if _normalized(form) not in seen and not re.search(rf"(?<!\w){re.escape(_normalized(form))}(?!\w)", query):
                        forms.append(form)
                        seen.add(_normalized(form))
                matched.append(alias)
            if not forms:
                return QueryBridgeResult(original_query, original_query, False, tuple(dict.fromkeys(matched)), None)
            retrieval_query = " ".join([original_query, *forms])
            return QueryBridgeResult(original_query, retrieval_query, True, tuple(dict.fromkeys(matched)), BRIDGE_VERSION)
        except Exception as exc:  # bounded, observable fallback; retrieval remains available
            logger.warning("historical_query_bridge_fallback error_type=%s", type(exc).__name__)
            return QueryBridgeResult(original_query, original_query, False, (), BRIDGE_VERSION, type(exc).__name__)
