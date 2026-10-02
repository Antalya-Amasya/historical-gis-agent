"""Offline, opt-in PERSON observations for retrieval; never event authority."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from backend.app.rag.query_roles import analyze_query, normalized_tokens


def _surface(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).replace("æ", "ae").replace("Æ", "AE")
    return " ".join("gaius" if t == "caius" else t for t in
                    re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold()))


@lru_cache(maxsize=1)
def person_records() -> tuple[dict, ...]:
    return tuple(json.loads((Path(__file__).parent / "registries" / "persons.json")
                            .read_text(encoding="utf-8"))["entities"])


@lru_cache(maxsize=1)
def _place_surfaces() -> frozenset[str]:
    # Read the existing geography labels; do not build a second place registry.
    from backend.app.routes.place_aliases import HISTORICAL_PLACE_ALIASES
    from backend.app.geography.place_registry import physical_records
    labels = [_surface(form) for r in HISTORICAL_PLACE_ALIASES
              for form in (r.canonical_name, *r.aliases)]
    labels.extend(_surface(form) for r in physical_records()
                  for form in (r["canonical_name"], *r["aliases"]))
    return frozenset(labels)


def _matches(text: str, records) -> list[tuple[int, int, str, dict]]:
    value = _surface(text)
    result = []
    for record in records:
        if record["entity_type"] != "PERSON":
            continue
        for form in (record["canonical_name"], *record["aliases"]):
            alias = _surface(form)
            match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", value)
            if match:
                # A citation/author mention is not the requested route subject.
                prefix = value[max(0, match.start() - 35):match.start()].rstrip()
                if re.search(r"(?:according to|written by|writings of)\s*$", prefix):
                    continue
                result.append((match.start(), -len(alias), alias, record))
    return sorted(result, key=lambda r: (r[0], r[1], r[3]["entity_id"]))


def _context(text: str, record: dict) -> list[str]:
    value = _surface(text)
    hits = [form for form in (*record.get("roles_titles", ()), *record.get("context_places", ()))
            if re.search(r"(?<!\w)" + re.escape(_surface(form)) + r"(?!\w)", value)]
    bounds = record.get("time_range")
    if bounds:
        # Only explicit era-marked years; never infer passage chronology.
        for year, era in re.findall(r"\b(\d{1,4})\s*(BCE|BC|CE|AD)\b", text, re.I):
            year = -int(year) if era.casefold() in {"bce", "bc"} else int(year)
            if bounds["start_year"] <= year <= bounds["end_year"]:
                hits.append("explicit_year_overlap")
    return sorted(set(hits))


def resolve_query_person(query: str, *, records=None) -> dict:
    records = person_records() if records is None else records
    matches = _matches(query, records)
    if not matches:
        roles = analyze_query(query)
        # LiteralUnknown is open-world. Single untyped nouns stay unresolved;
        # existing retrieval support remains intact regardless of this label.
        words = roles.person_sequence
        surface = " ".join(words)
        is_place = _surface(query) in _place_surfaces()
        plausible = len(words) >= 2 and not roles.multiple_person_phrases_detected
        return {"entity_type": "PLACE" if is_place else ("PERSON" if plausible else "UNRESOLVED"),
                "surface": surface or query, "canonical_candidate": None, "entity_id": None,
                "candidate_ids": [], "match_type": "LITERAL_UNKNOWN", "context_match": [],
                "basis": "literal_query_roles_no_registry_requirement"}
    first = matches[0]
    candidates = {r[3]["entity_id"]: r[3] for r in matches if r[:2] == first[:2]}
    contexts = {key: _context(query, r) for key, r in candidates.items()}
    # Common context cannot distinguish homonyms. Require independent, unique
    # support; keep ties ambiguous instead of silently picking a registry row.
    scores = {key: len(values) for key, values in contexts.items()}
    best = max(scores.values())
    winners = [key for key in candidates if scores[key] == best]
    entity_id = winners[0] if len(winners) == 1 else None
    selected = candidates[entity_id] if entity_id else first[3]
    canonical = _surface(selected["canonical_name"])
    match_type = ("AMBIGUOUS" if not entity_id else
                  "EXACT" if first[2] == canonical else
                  "PARTIAL" if len(first[2].split()) == 1 and len(canonical.split()) > 1 else "ALIAS")
    return {"entity_type": "PERSON", "surface": first[2],
            "canonical_candidate": selected["canonical_name"] if entity_id else None,
            "entity_id": entity_id, "candidate_ids": sorted(candidates),
            "match_type": match_type, "context_match": contexts.get(entity_id, []),
            "basis": "curated_labels_and_explicit_local_context"}


def entity_compatibility(query: str, text: str, *, records=None, query_entity=None) -> dict:
    records = person_records() if records is None else records
    q = query_entity or resolve_query_person(query, records=records)
    candidates = [r for r in records if r["entity_id"] in q["candidate_ids"]]
    matches = _matches(text, candidates)
    local_ids = sorted({m[3]["entity_id"] for m in matches})
    contexts = {key: _context(text, next(r for r in candidates if r["entity_id"] == key)) for key in local_ids}
    local_best = max((len(values) for values in contexts.values()), default=0)
    local_winners = [key for key in local_ids if len(contexts[key]) == local_best]
    classification, bonus, reason = "NO_LOCAL_ENTITY", 0.0, None
    if local_ids and q["entity_id"] is None:
        classification = "AMBIGUOUS"
    elif local_ids:
        desired = q["entity_id"]
        if len(candidates) > 1 and len(local_winners) == 1 and local_best:
            if desired == local_winners[0]:
                classification, bonus = "MATCH", 0.08
            else:
                classification, bonus, reason = "CONFLICT", -0.08, "different_explicit_role_context"
        elif len(candidates) > 1:
            classification = "AMBIGUOUS"
        else:
            classification = "COMPATIBLE"
            # Alias enrichment is modest; repeated literal names get no bonus.
            if matches[0][2] != q["surface"]:
                bonus = 0.04
    return {"query_entity_surface": q["surface"], "query_entity_id": q["entity_id"],
            "query_entity_match_type": q["match_type"], "query_entity_type": q["entity_type"],
            "passage_entity_ids": local_ids, "entity_compatibility": classification,
            "entity_context_match": contexts.get(q["entity_id"], []),
            "entity_conflict_reason": reason, "entity_bonus": bonus}
