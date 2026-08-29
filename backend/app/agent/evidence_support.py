"""Deterministic relevance and provenance assessment for grounded Agent outputs."""
from __future__ import annotations
from dataclasses import dataclass
import re
import re
from backend.app.routes.place_aliases import HISTORICAL_PLACE_ALIASES

# Maintainable bilingual normalization data, intentionally separate from runtime contracts.
SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "caesar": ("caesar", "凯撒"),
    "gaul": ("gaul", "gallic", "gallia", "高卢"),
    "hannibal": ("hannibal", "汉尼拔"),
    "alps": ("alps", "alpine", "阿尔卑斯"),
    "polybius": ("polybius", "波利比乌斯"),
    "livy": ("livy", "李维"),
}


@dataclass(frozen=True)
class EvidenceRelevance:
    evidence_id: str
    matched_terms: tuple[str, ...]
    score: int
    relevant: bool


@dataclass(frozen=True)
class EvidenceSupportAssessment:
    status: str
    relevant_evidence_ids: tuple[str, ...]
    relevant_count: int
    total_count: int
    matched_subject_terms: tuple[str, ...]
    missing_subject_terms: tuple[str, ...]
    reason: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def _aliases_present(text: str, canonical: str) -> bool:
    normalized = _normalize(text)
    return any(alias in normalized for alias in SUBJECT_ALIASES[canonical])


def extract_subject_terms(user_query: str) -> tuple[str, ...]:
    return tuple(term for term in SUBJECT_ALIASES if _aliases_present(user_query, term))


def assess_evidence_support(user_query: str, requested_output: str, evidence: list) -> EvidenceSupportAssessment:
    subject_terms = extract_subject_terms(user_query)
    if requested_output != "historical_route":
        return EvidenceSupportAssessment("sufficient", (), 0, len(evidence), (), (), "route support not requested")
    if not subject_terms:
        return EvidenceSupportAssessment("insufficient", (), 0, len(evidence), (), (), "no specific route subject could be extracted")
    relevances: list[EvidenceRelevance] = []
    covered: set[str] = set()
    for item in evidence:
        haystack = " ".join(str(getattr(item, field, "") or "") for field in ("author", "work", "book", "locator", "excerpt", "text"))
        matched = tuple(term for term in subject_terms if _aliases_present(haystack, term))
        relevance = EvidenceRelevance(str(getattr(item, "id", "")), matched, len(matched) * 2, bool(matched))
        relevances.append(relevance)
        covered.update(matched)
    relevant = tuple(item.evidence_id for item in relevances if item.relevant)
    missing = tuple(term for term in subject_terms if term not in covered)
    if not relevant:
        return EvidenceSupportAssessment("irrelevant", (), 0, len(evidence), (), subject_terms, "no accumulated Evidence matches the request subject")
    if missing:
        return EvidenceSupportAssessment("insufficient", relevant, len(relevant), len(evidence), tuple(sorted(covered)), missing, "Evidence only covers part of the request subject")
    return EvidenceSupportAssessment("sufficient", relevant, len(relevant), len(evidence), tuple(sorted(covered)), (), "Accumulated Evidence covers all extracted request subject terms")


def has_unsupported_route_pattern(answer: str | None) -> bool:
    text = _normalize(answer or "")
    chinese_route = "\u8def\u7ebf.{0,24}(?:\u7ecf\u8fc7|\u5305\u62ec|\u7531.+\u5230)"
    patterns = (r"(?:→|->|—>)", r"\broute\s+(?:consists of|passes through|goes from|goes through)\b", chinese_route)
    return any(re.search(pattern, text) for pattern in patterns)


# Small general-purpose filters. Entity extraction is phrase-first; this set only
# excludes language and application concepts that are never historical entities.
LINGUISTIC_STOPWORDS = frozenset({"however", "therefore", "moreover", "current", "available", "these", "this", "that", "with", "without", "from", "into", "about", "because", "although", "key", "main", "major"})
SYSTEM_TERMS = frozenset({"evidence", "route", "historicalroute", "historical", "reconstruction", "suggestion", "suggestions", "unverified", "supported", "unsupported", "model", "research", "system", "map", "node", "geojson", "rag", "mcp", "geography"})
TITLE_CONNECTORS = ("de", "of", "the", "in", "on", "et")


@dataclass(frozen=True)
class CandidatePhrase:
    raw_text: str
    normalized_text: str
    start: int
    end: int
    entity_type: str


@dataclass(frozen=True)
class CandidateEntityAssessment:
    term: str
    normalized_term: str
    entity_like: bool
    entity_type: str
    source: str
    context: str
    reason: str


@dataclass(frozen=True)
class UnverifiedSuggestion:
    text: str
    provenance: str = "model_suggestion"
    verified: bool = False
    eligible_for_route: bool = False


@dataclass(frozen=True)
class AnswerProvenance:
    evidence_grounded_claims: tuple[str, ...]
    unverified_suggestions: tuple[UnverifiedSuggestion, ...]
    evidence_gaps: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class FinalGroundingAssessment:
    status: str
    provenance: AnswerProvenance
    candidate_entities: tuple[CandidateEntityAssessment, ...]
    unsupported_fact_terms: tuple[str, ...]
    ignored_non_entity_terms: tuple[str, ...]
    detected_work_titles: tuple[str, ...]
    unsupported_route_content: bool
    candidate_explosion: bool
    reason: str


def _text_of(item) -> str:
    return " ".join(str(getattr(item, field, "") or "") for field in ("author", "work", "book", "locator", "excerpt", "text"))


def _metadata_values(evidence: list) -> dict[str, set[str]]:
    values = {"author": set(), "work": set(), "other": set()}
    for item in evidence:
        for field in ("author", "work", "book", "locator"):
            value = _normalize(str(getattr(item, field, "") or ""))
            if value:
                values["work" if field in {"work", "book"} else "author" if field == "author" else "other"].add(value)
    return values


def _alias_terms() -> set[str]:
    terms: set[str] = set()
    for place in HISTORICAL_PLACE_ALIASES:
        terms.add(place.canonical_name.lower())
        terms.update(alias.lower() for alias in place.aliases)
    return terms


def _explicit_chinese_query_entities(user_query: str) -> set[str]:
    entities: set[str] = set()
    # Bilingual subject normalization and place aliases are explicit entity data.
    normalized = _normalize(user_query)
    for canonical in extract_subject_terms(user_query):
        entities.update(SUBJECT_ALIASES[canonical])
    for alias in _alias_terms():
        if alias in normalized:
            entities.add(alias)
    # A deliberately narrow request form for a named subject before "related route".
    for match in re.finditer("(?:\u5c55\u793a|\u663e\u793a|\u7ed8\u5236)\\s*([\u4e00-\u9fff]{2,4}?)(?:\u76f8\u5173)?\u8def\u7ebf", user_query or ""):
        entities.add(match.group(1))
    return {value.lower() for value in entities}


def _explicit_chinese_evidence_entities(item) -> set[str]:
    entities: set[str] = set()
    # Metadata is structured provenance; preserve complete Chinese field values.
    for field in ("author", "work", "book", "locator"):
        value = str(getattr(item, field, "") or "").strip()
        if value and re.fullmatch("[\u4e00-\u9fff]{2,24}", value):
            entities.add(value.lower())
    source = _text_of(item)
    for canonical in extract_subject_terms(source):
        entities.update(alias.lower() for alias in SUBJECT_ALIASES[canonical])
    normalized = _normalize(source)
    for alias in _alias_terms():
        if alias in normalized:
            entities.add(alias)
    # Text is accepted only through an explicit label, never arbitrary prose spans.
    for match in re.finditer("(?:\u5730\u70b9|\u5730\u540d|place)\\s*[:\uff1a]\\s*([\u4e00-\u9fff]{2,8})", source, re.IGNORECASE):
        entities.add(match.group(1).lower())
    return entities


def _find_vocab_spans(original_text: str, vocab: str) -> list[tuple[int, int]]:
    """Return spans in the original answer's coordinate space.

    ``vocab`` is canonicalized for comparisons, never for source offsets.
    English whitespace in a canonical phrase may match any original whitespace;
    Chinese terms remain exact original-text substring matches.
    """
    if re.search("[\u4e00-\u9fff]", vocab):
        positions: list[tuple[int, int]] = []
        offset = original_text.find(vocab)
        while offset >= 0:
            positions.append((offset, offset + len(vocab)))
            offset = original_text.find(vocab, offset + 1)
        return positions
    pattern = re.escape(vocab).replace(r"\ ", r"\s+")
    return [(match.start(), match.end()) for match in re.finditer(rf"(?<!\w){pattern}(?!\w)", original_text, re.IGNORECASE)]


def _query_vocab(user_query: str) -> set[str]:
    tokens = {match.lower() for match in re.findall(r"\b[A-Z][A-Za-z-]{2,}\b", user_query or "")}
    tokens.update(extract_subject_terms(user_query))
    tokens.update(_explicit_chinese_query_entities(user_query))
    normalized = _normalize(user_query)
    tokens.update(alias for alias in _alias_terms() if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized))
    return tokens


def _evidence_vocab(evidence: list) -> set[str]:
    vocab: set[str] = set()
    for item in evidence:
        source = _text_of(item)
        vocab.update(match.lower() for match in re.findall(r"\b[A-Z][A-Za-z-]{2,}\b", source))
        vocab.update(extract_subject_terms(source))
        vocab.update(_explicit_chinese_evidence_entities(item))
    values = _metadata_values(evidence)
    for group in values.values():
        vocab.update(group)
    return vocab

def _title_spans(text: str) -> list[tuple[str, int, int]]:
    token = r"[A-Z][A-Za-z-]{2,}"
    connector = "|".join(TITLE_CONNECTORS)
    pattern = rf"\b{token}\s+(?:{connector})\s+{token}(?:\s+{token}){{0,3}}\b"
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(pattern, text or "")]


def _strong_entity_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for pattern in (
        r"\b(?:through|via|at|from|to|near)\s+([A-Z][A-Za-z-]{2,})\b",
        r"\b([A-Z][A-Za-z-]{2,})\s+(?:was|is|were|served|marked|became|occurred|happened)\b",
        r"\b(?i:key|main|major)\s+(?:locations|places|nodes)\s+(?:include|are)\s+([A-Z][A-Za-z-]{2,}(?:\s*,\s*[A-Z][A-Za-z-]{2,})*)",
    ):
        for match in re.finditer(pattern, text or ""):
            group = match.group(1)
            offset = match.start(1)
            for item in re.finditer(r"[A-Z][A-Za-z-]{2,}", group):
                spans.append((item.group(0), offset + item.start(), offset + item.end()))
    return spans


def _phrase_entity_type(raw: str, normalized: str, evidence: list) -> str:
    metadata = _metadata_values(evidence)
    if normalized in SYSTEM_TERMS or normalized in LINGUISTIC_STOPWORDS:
        return "system_term"
    if normalized in _alias_terms():
        return "place"
    if normalized in metadata["work"] or len(_title_spans(raw)) > 0:
        return "work"
    return "unknown_entity_like"


def extract_candidate_phrases(answer: str, user_query: str, evidence: list) -> tuple[CandidatePhrase, ...]:
    query_vocab = _query_vocab(user_query)
    evidence_vocab = _evidence_vocab(evidence)
    spans: dict[tuple[int, int], CandidatePhrase] = {}

    def add(raw: str, start: int, end: int, forced_type: str | None = None):
        if answer[start:end] != raw:
            raise ValueError("CandidatePhrase spans must reference the original answer text")
        normalized = _normalize(raw)
        if not normalized:
            return
        entity_type = forced_type or _phrase_entity_type(raw, normalized, evidence)
        spans[(start, end)] = CandidatePhrase(raw, normalized, start, end, entity_type)

    for raw, start, end in _title_spans(answer):
        add(raw, start, end, "work")
    for vocab in query_vocab | evidence_vocab | _alias_terms():
        if len(vocab) < 2:
            continue
        for start, end in _find_vocab_spans(answer, vocab):
            add(answer[start:end], start, end)
    for raw, start, end in _strong_entity_spans(answer):
        add(raw, start, end)
    for match in re.finditer(r"\b([A-Z][A-Za-z-]{2,})\s+may\s+be\s+worth\s+searching\b", answer):
        add(match.group(1), match.start(1), match.end(1))
    for match in re.finditer(r"\b(?:unverified\s+research\s+suggestions?|research\s+suggestions?)\s*:\s*([A-Z][A-Za-z-]{2,})", answer, re.IGNORECASE):
        add(match.group(1), match.start(1), match.end(1))

    # Chinese candidates require explicit boundaries, never n-gram substring scans.
    suggestion_active = any(marker in answer for marker in ("\u672a\u9a8c\u8bc1\u7814\u7a76\u5efa\u8bae", "\u53ef\u8fdb\u4e00\u6b65\u68c0\u7d22"))
    if suggestion_active:
        for match in re.finditer("(?:^|\n)\\s*(?:[-*]|\\d+[.)])\\s*([\u4e00-\u9fff]{2,8})(?=\\s|[\u3002\uff1b\uff0c,;\n]|$)", answer):
            add(match.group(1), match.start(1), match.end(1), "unknown_entity_like")
        for match in re.finditer("(?:\u672a\u9a8c\u8bc1\u7814\u7a76\u5efa\u8bae|\u53ef\u8fdb\u4e00\u6b65\u68c0\u7d22)\\s*[:\uff1a]\\s*([\u4e00-\u9fff]{2,8})(?=[\u3002\uff1b\uff0c,;\n]|$)", answer):
            add(match.group(1), match.start(1), match.end(1), "unknown_entity_like")
    for match in re.finditer("\u8def\u7ebf\u7ecf\u8fc7([\u4e00-\u9fff]{2,8})(?=[\u3002\uff1b\uff0c,;\n]|$)", answer):
        add(match.group(1), match.start(1), match.end(1), "unknown_entity_like")
    for match in re.finditer("\u5230\u8fbe([\u4e00-\u9fff]{2,8})(?=[\u3002\uff1b\uff0c,;\n]|$)", answer):
        add(match.group(1), match.start(1), match.end(1), "unknown_entity_like")

    phrase_values = list(spans.values())
    # Longest complete title wins; telemetry counts unique normalized phrases.
    phrase_values = [phrase for phrase in phrase_values if not any(other != phrase and other.entity_type == "work" and other.start <= phrase.start and phrase.end <= other.end for other in phrase_values)]
    unique: dict[str, CandidatePhrase] = {}
    for phrase in sorted(phrase_values, key=lambda item: (-(item.end - item.start), item.start)):
        unique.setdefault(phrase.normalized_text, phrase)
    return tuple(sorted(unique.values(), key=lambda item: (item.start, item.end)))
def _blocks(answer: str) -> list[str]:
    return [block for block in re.split(r"\n\s*\n", answer) if block.strip()] or [answer]


def _is_suggestion_context(block: str) -> bool:
    normalized = _normalize(block)
    patterns = (r"\bunverified\b", r"\bnot supported by (?:the )?current evidence\b", r"\bnot verified\b", r"\b(?:further|future) research\b", r"\b(?:worth|for) searching\b", r"\bto be verified\b")
    chinese_markers = ("\u672a\u9a8c\u8bc1", "\u540e\u7eed\u68c0\u7d22", "\u8fdb\u4e00\u6b65\u68c0\u7d22", "\u5f85\u9a8c\u8bc1", "\u5f53\u524d\u8bc1\u636e\u4e0d\u652f\u6301", "\u5f53\u524d\u53f2\u6599\u4e0d\u652f\u6301")
    return any(re.search(pattern, normalized) for pattern in patterns) or any(marker in block for marker in chinese_markers)


def _has_unattributed_itinerary(text: str) -> bool:
    return bool(re.search(r"\b(?:key|main|major)\s+(?:locations|places|nodes)\s+(?:include|are)\s+[A-Z]", text or "", re.IGNORECASE))


def _work_title_context(block: str, phrase: CandidatePhrase) -> str | None:
    """Classify only a model-only bibliographic title's local discourse role."""
    escaped = re.escape(phrase.raw_text)
    factual_patterns = (
        rf"\baccording to\s+{escaped}\b",
        rf"\b{escaped}\b.{{0,80}}\b(?:proves|confirms|records that|states that|demonstrates|shows that|clearly describes)\b",
    )
    if any(re.search(pattern, block, re.IGNORECASE) for pattern in factual_patterns):
        return "factual_citation"
    corpus_gap_patterns = (
        rf"\b(?:current (?:evidence|corpus)|the current corpus)\b.{{0,96}}\b(?:does not|doesn't|do not|don't)\s+(?:include|contain|have)\b.{{0,96}}{escaped}",
        rf"\b{escaped}\b.{{0,96}}\b(?:is not|isn't|was not|wasn't)\s+(?:in|part of)\s+(?:the )?current (?:evidence|corpus)\b",
        rf"\b{escaped}\b.{{0,96}}\b(?:may|might|could)\s+be\s+worth\s+(?:adding|consulting|searching)\b",
        rf"\b{escaped}\b.{{0,96}}\b(?:could|may)\s+be\s+added\b.{{0,96}}\b(?:further|future) research\b",
        rf"\b(?:unverified research suggestion|supplementary source)\s*:\s*{escaped}\b",
    )
    if any(re.search(pattern, block, re.IGNORECASE) for pattern in corpus_gap_patterns):
        return "corpus_gap"
    return None


def _has_fact_assertion(block: str, phrase: CandidatePhrase) -> bool:
    escaped = re.escape(phrase.raw_text)
    relation = rf"(?:through|via|at|from|to|near)\s+{escaped}\b|\b{escaped}\b.{{0,56}}(?:route|campaign|battle|location|node|proves|records)"
    simple = rf"\b{escaped}\b\s+(?:was|is|were|served|marked|became|occurred|happened)\b"
    general_fact = r"\b(?:major|decisive|key)\s+battle\s+(?:occurred|happened)\b"
    chinese = rf"{escaped}.{{0,18}}(?:\u662f|\u4f4d\u4e8e|\u7ecf\u8fc7|\u53d1\u751f|\u51b3\u5b9a\u6027|\u6218\u5f79|\u8282\u70b9)"
    return bool(re.search(relation, block, re.IGNORECASE) or re.search(simple, block, re.IGNORECASE) or re.search(general_fact, block, re.IGNORECASE) or re.search(chinese, block))


def classify_candidate_phrase(phrase: CandidatePhrase, context: str, user_query: str, evidence: list) -> CandidateEntityAssessment:
    query_vocab = _query_vocab(user_query)
    evidence_vocab = _evidence_vocab(evidence)
    if phrase.entity_type in {"system_term", "ordinary_prose"} or phrase.normalized_text in SYSTEM_TERMS or phrase.normalized_text in LINGUISTIC_STOPWORDS:
        return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, False, phrase.entity_type, "system", "non_claim", "system or ordinary prose phrase")
    if phrase.normalized_text in query_vocab:
        return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, True, "query_subject", "query", "grounded_fact", "present in user query")
    if phrase.normalized_text in evidence_vocab:
        return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, True, phrase.entity_type, "evidence", "grounded_fact", "present in accumulated Evidence")
    if phrase.entity_type == "work":
        work_context = _work_title_context(context, phrase)
        if work_context == "factual_citation":
            return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, True, phrase.entity_type, "model_only", "unsupported_fact", "model-only work title was used as a factual citation")
        if work_context == "corpus_gap":
            return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, True, phrase.entity_type, "model_only", "unverified_suggestion", "model-only work title was explicitly limited to a corpus gap or research suggestion")
    if _has_fact_assertion(context, phrase) or has_unsupported_route_pattern(context):
        return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, True, phrase.entity_type, "model_only", "unsupported_fact", "model-only phrase used in factual or route context")
    if _is_suggestion_context(context):
        return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, True, phrase.entity_type, "model_only", "unverified_suggestion", "explicitly limited to a research suggestion")
    return CandidateEntityAssessment(phrase.raw_text, phrase.normalized_text, True, phrase.entity_type, "model_only", "unsupported_fact", "model-only entity-like phrase lacks explicit suggestion context")


def event_relation_supports_answer(answer: str | None, events: list, evidence: list) -> bool:
    """A bounded Event may aid QA only through wholly visible Evidence refs."""
    visible = {str(item.id) for item in evidence[:8]}
    text = answer or ""
    entities = {item.lower() for item in re.findall(r"\b[A-Z][A-Za-z]+\b", text) if item.lower() not in {"the", "a", "an", "evidence"}}
    if not entities:
        return False
    for event in events:
        refs = set(getattr(event, "evidence_refs", []) or [])
        if not refs or not refs.issubset(visible):
            continue
        context = " ".join([str(getattr(event, "name", "")), str(getattr(event, "summary", "")), *getattr(event, "source_statements", [])]).lower()
        # Every material named entity must be traceable through the derived event.
        if all(entity in context for entity in entities):
            return True
    return False


def assess_final_answer_provenance(answer: str | None, user_query: str, evidence: list) -> FinalGroundingAssessment:
    answer = answer or ""
    phrases = extract_candidate_phrases(answer, user_query, evidence)
    assessments = tuple(classify_candidate_phrase(phrase, next((block for block in _blocks(answer) if phrase.normalized_text in _normalize(block)), answer), user_query, evidence) for phrase in phrases)
    unsupported = tuple(sorted(item.normalized_term for item in assessments if item.entity_like and item.source == "model_only" and item.context == "unsupported_fact"))
    suggestions = tuple(UnverifiedSuggestion(text=item.normalized_term) for item in assessments if item.entity_like and item.source == "model_only" and item.context == "unverified_suggestion")
    grounded = tuple(sorted(item.normalized_term for item in assessments if item.entity_like and item.context == "grounded_fact"))
    ignored = tuple(sorted(item.normalized_term for item in assessments if not item.entity_like))
    route_signal = has_unsupported_route_pattern(answer) or _has_unattributed_itinerary(answer)
    evidence_subjects: set[str] = set()
    for item in evidence:
        evidence_subjects.update(extract_subject_terms(_text_of(item)))
    provenance = AnswerProvenance(grounded, suggestions, tuple(sorted(set(extract_subject_terms(user_query)) - evidence_subjects)), ("current Evidence is insufficient for route construction",))
    work_titles = tuple(sorted(item.normalized_term for item in assessments if item.entity_type == "work"))
    candidate_explosion = len(assessments) > 30
    if unsupported or route_signal:
        return FinalGroundingAssessment("unsupported_fact", provenance, assessments, unsupported, ignored, work_titles, route_signal, candidate_explosion, "model-only entity-like phrases or unattributed route content were presented as facts")
    if suggestions:
        return FinalGroundingAssessment("grounded_with_unverified_suggestions", provenance, assessments, (), ignored, work_titles, False, candidate_explosion, "model-only phrases were explicitly limited to research suggestions")
    return FinalGroundingAssessment("grounded", provenance, assessments, (), ignored, work_titles, False, candidate_explosion, "final answer stayed within query and Evidence-grounded phrases")


_EVIDENCE_CITATION = re.compile(
    r"\[Evidence:\s*(?P<id>.+?)\s+\N{EM DASH}\s*(?P<author>[^,\]]+),\s*(?P<work>[^,\]]+),\s*(?P<locator>[^\]]+)\]",
    re.IGNORECASE,
)

def validate_evidence_selection(ids: tuple[str, ...], evidence: list, *, require_selection: bool) -> tuple[str, ...]:
    if require_selection and not ids: return ("missing_evidence_selection",)
    visible = {str(item.id) for item in evidence[:8]}
    return tuple(f"unknown_evidence_id:{identifier}" for identifier in dict.fromkeys(ids) if not identifier or identifier not in visible)

def render_evidence_citations(ids: tuple[str, ...], evidence: list) -> str:
    by_id = {str(item.id): item for item in evidence[:8]}
    return " ".join(f"[Evidence: {identifier} \N{EM DASH} {by_id[identifier].author}, {by_id[identifier].work}, {by_id[identifier].locator}]" for identifier in dict.fromkeys(ids))


def validate_evidence_citations(answer: str | None, evidence: list, *, require_citation: bool) -> tuple[str, ...]:
    """Validate only the compact, tool-visible Evidence citation contract."""
    citations = list(_EVIDENCE_CITATION.finditer(answer or ""))
    if require_citation and not citations:
        return ("missing_evidence_citation",)
    by_id = {str(item.id): item for item in evidence}
    issues: list[str] = []
    for citation in citations:
        identifier = citation.group("id").strip()
        item = by_id.get(identifier)
        if item is None:
            issues.append(f"unknown_evidence_id:{identifier}")
            continue
        for field in ("author", "work", "locator"):
            expected = _normalize(str(getattr(item, field, "") or "")).strip(" .;:")
            actual = _normalize(citation.group(field)).strip(" .;:")
            if actual != expected:
                issues.append(f"mismatched_evidence_{field}:{identifier}")
    return tuple(issues)
