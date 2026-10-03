"""Offline, opt-in PERSON observations for retrieval; never event authority."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from backend.app.rag.query_roles import analyze_query, _PRAENOMINA, _ACTION_UNION
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



def _ownership_end(suffix: str, role_words: set[str], places: frozenset[str]) -> int:
    """Stop at explicit argument/subject changes, independent of name case."""
    end = len(suffix)
    # Unknown subordinate subjects do not require registry membership or NER.
    boundary = re.search(r"\b(?:that|whether)\b", suffix, re.I)
    if boundary:
        end = boundary.start()
    for conjunction in re.finditer(r"\band\s+([^\W_]+)", suffix, re.I):
        if _surface(conjunction.group(1)) not in role_words | places:
            end = min(end, conjunction.start())
            break
    # Existing Roman name normalization is a rejection resource, not identity.
    words = list(re.finditer(r"[^\W_]+", suffix))
    actor_end = None
    for token in words:
        if _surface(token.group()) in _PRAENOMINA:
            end = min(end, token.start())
            actor_end = token.start()
            break
    offsets = {}
    offset = 0
    for token in words:
        offsets[offset] = token.start()
        offset += len(_surface(token.group())) + 1
    for start, _, _, _ in _matches(suffix, person_records()):
        if start in offsets:
            actor_end = offsets[start]
            end = min(end, actor_end)
            break
    # Name + apposition / role + name are structural person cues, in any case.
    if role_words:
        roles = "|".join(re.escape(w) for w in sorted(role_words))
        name = r"[^\W\d_]+"
        for match in re.finditer(rf"\b(?P<name>{name}),\s*(?:the\s+)?(?:{roles})\b", suffix, re.I):
            if _surface(match.group('name')) not in places | role_words | {'the', 'a', 'an'}:
                end = min(end, match.start())
        for match in re.finditer(rf"\b(?:{roles})\s+(?P<name>{name})\b", suffix, re.I):
            token = _surface(match.group('name'))
            if token not in places | role_words | _ACTION_UNION | {'from', 'to', 'in', 'at', 'of', 'for', 'with', 'among', 'between', 'as', 'by', 'and', 'was', 'is', 'served', 'waited', 'spoke', 'died'}:
                end = min(end, match.start())
    # A location object is not a person boundary. Unknown typed objects are
    # conservative; infinitives remain part of the named subject's proposition.
    for match in re.finditer(r"\b(?:to|from|in|at|near|into|for)\s+([^\W_]+)", suffix, re.I):
        token = _surface(match.group(1))
        if token not in places | role_words | _ACTION_UNION | {'the', 'a', 'an', 'be', 'consult', 'inquire', 'return', 'returning'} and not token.isdigit():
            end = min(end, match.start())
    relation = re.search(r"\b(?:met|meet|saw|called|appointed|asked|told|heard|reported|says|said)\b", suffix, re.I)
    if relation:
        prefix = suffix[:relation.start()]
        after = suffix[relation.end():]
        passive = re.search(r"\b(?:was|is|had been|has been)\s*$", prefix, re.I)
        infinitive = re.match(r"\s+to\b", after, re.I)
        if not passive and not infinitive:
            end = min(end, relation.start())
    # Remove a pre-nominal role that attaches to the next actor at the boundary.
    if actor_end == end:
        lead_words = list(re.finditer(r"[^\W_]+", suffix[:end]))
        for token in reversed(lead_words):
            if _surface(token.group()) not in role_words | {'the', 'a', 'an'}:
                break
            end = token.start()
    return end


def _owned_contexts(text: str, candidates: list[dict]) -> dict[str, list[str]]:
    """Mention-local observations, without citation or pronoun inheritance."""
    contexts = {r['entity_id']: set() for r in candidates}
    role_words = set(' '.join(_surface(role) for r in candidates
                             for role in r.get('roles_titles', ())).split())
    places = _place_surfaces() | frozenset(_surface(place) for r in candidates
                                         for place in r.get('context_places', ()))
    sentences = [m.group() for m in _SENTENCE.finditer(text)]
    for index, sentence in enumerate(sentences):
        if index + 1 < len(sentences) and re.match(r"\s*(?:he|she|they)\b", sentences[index + 1], re.I):
            if any(_context(sentences[index + 1], r) for r in candidates):
                continue
        for clause in re.split(r"[;:]|\b(?:while|whereas|but|although|because)\b", sentence, flags=re.I):
            words = [(m, _surface(m.group())) for m in re.finditer(r"[^\W_]+", clause) if _surface(m.group())]
            offsets = {}
            offset = 0
            for i, (_, word) in enumerate(words):
                offsets[offset] = i
                offset += len(word) + 1
            for start, _, alias, record in _matches(clause, candidates):
                if start not in offsets:
                    continue
                first = offsets[start]
                last = first + len(alias.split())
                if last > len(words) or ' '.join(w for _, w in words[first:last]) != alias:
                    continue
                citation = re.match(r"\s*(?:according to|written by|writings of)\b", clause, re.I)
                if citation and (',' not in clause or words[first][0].start() < clause.index(',')):
                    continue
                mention_end = words[last - 1][0].end()
                suffix = clause[mention_end:]
                end = mention_end + _ownership_end(suffix, role_words, places)
                before = first
                while before and words[before - 1][1] in role_words | {'the', 'a', 'an'}:
                    before -= 1
                contexts[record['entity_id']].update(_context(clause[words[before][0].start():end], record))
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
