"""Offline, opt-in PERSON observations for retrieval; never event authority."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.lexical_index import _SENTENCE


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
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", value):
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



def _owned_contexts(text: str, candidates: list[dict]) -> dict[str, list[str]]:
    """Explicit mention-local observations; no pronoun or citation inheritance.

    Context starts at the mention, stops at a clause/other capitalized name,
    and may include an immediately adjacent pre-nominal role. Place labels
    are candidate context strings, not a new geographic identity resource.
    """
    contexts = {r["entity_id"]: set() for r in candidates}
    allowed = {_surface(form) for r in candidates
               for form in (*r.get("roles_titles", ()), *r.get("context_places", ()))}
    role_words = set(" ".join(_surface(role) for r in candidates
                             for role in r.get("roles_titles", ())).split())
    sentences = [m.group() for m in _SENTENCE.finditer(text)]
    for index, sentence in enumerate(sentences):
        # A two-sentence pronoun continuation is deliberately unresolved.
        # Do not combine the named role with a later pronoun's place context.
        if index + 1 < len(sentences) and re.match(r"\s*(?:he|she|they)\b", sentences[index + 1], re.I):
            if any(_context(sentences[index + 1], r) for r in candidates):
                continue
        for clause in re.split(r"[;:]|\b(?:while|whereas|but|although|because)\b", sentence, flags=re.I):
            words = [(m, _surface(m.group())) for m in re.finditer(r"[^\W_]+", clause) if _surface(m.group())]
            offsets = {}; offset = 0
            for i, (_, word) in enumerate(words):
                offsets[offset] = i; offset += len(word) + 1
            for start, _, alias, record in _matches(clause, candidates):
                if start not in offsets:
                    continue
                first = offsets[start]; last = first + len(alias.split())
                if last > len(words) or " ".join(w for _, w in words[first:last]) != alias:
                    continue
                citation = re.match(r"\s*(?:according to|written by|writings of)\b", clause, re.I)
                if citation and ("," not in clause or words[first][0].start() < clause.index(",")):
                    continue
                end = len(clause)
                suffix = clause[words[last - 1][0].end():]
                # A coordinated clause cannot lend its subject's context.
                # Keep coordinated role/place labels so contradictions survive.
                for conjunction in re.finditer(r"\band\s+([^\W_]+)", suffix, re.I):
                    if _surface(conjunction.group(1)) not in role_words | allowed:
                        end = words[last - 1][0].end() + conjunction.start()
                        break
                for name in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", suffix):
                    if _surface(name.group()) not in allowed:
                        lead = suffix[:name.start()]
                        lead_words = list(re.finditer(r"[^\W_]+", lead))
                        cut = name.start()
                        for token in reversed(lead_words):
                            if _surface(token.group()) not in role_words | allowed | {"the", "a", "an", "from", "in", "to", "of"}:
                                break
                            cut = token.start()
                        end = min(end, words[last - 1][0].end() + cut); break
                # Relational verbs introduce somebody else's role/object even
                # when that name is lowercase; they cannot supply context here.
                relation = re.search(r"\b(?:met|meet|saw|called|appointed|asked|told|heard|reported|says|said)\b", suffix, re.I)
                if relation and not re.match(r"\s+(?:was|is|had been|has been)\s+", suffix, re.I):
                    end = min(end, words[last - 1][0].end() + relation.start())
                before = first
                while before and words[before - 1][1] in role_words | {"the", "a", "an"}:
                    before -= 1
                span = clause[words[before][0].start():end]
                contexts[record["entity_id"]].update(_context(span, record))
    return {key: sorted(values) for key, values in contexts.items()}


def _context_winners(contexts: dict[str, list[str]], candidates: list[dict]) -> list[str]:
    # Contradictory explicit roles cannot be outvoted by extra place hits.
    common_roles = set.intersection(*(set(r.get("roles_titles", ())) for r in candidates)) if candidates else set()
    role_owners = [r["entity_id"] for r in candidates if
                   (set(r.get("roles_titles", ())) - common_roles) & set(contexts.get(r["entity_id"], ()))]
    if len(role_owners) > 1:
        return role_owners
    best = max((len(values) for values in contexts.values()), default=0)
    return [key for key, values in contexts.items() if len(values) == best]


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
    contexts = _owned_contexts(query, list(candidates.values()))
    # Common context cannot distinguish homonyms. Require independent, unique
    # support; keep ties ambiguous instead of silently picking a registry row.
    winners = _context_winners(contexts, list(candidates.values()))
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
    contexts = {key: values for key, values in _owned_contexts(text, candidates).items() if key in local_ids}
    local_best = max((len(values) for values in contexts.values()), default=0)
    local_winners = _context_winners(contexts, [r for r in candidates if r["entity_id"] in local_ids])
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
