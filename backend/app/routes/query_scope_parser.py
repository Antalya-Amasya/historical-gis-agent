"""Deterministic query scope parsing with explicit character-span ownership."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.app.models import TemporalPrecision

_BROAD_POSSESSIVE_ROUTE_SUBJECT = re.compile(
    r"\b(?:show|trace|reconstruct|follow)\s+(.+?)['\u2019]s\s+(?:historical\s+)?"
    r"(?:routes?|movements?)\b",
    re.IGNORECASE,
)
_POSSESSIVE_ROUTE_SUBJECT = re.compile(
    r"\b([A-Z][A-Za-z'\u2019-]+(?:\s+[A-Z][A-Za-z'\u2019-]+){0,3})['\u2019]s\s+"
    r"(?:historical\s+)?(?:routes?|movements?)\b",
    re.IGNORECASE,
)
_TRACE_PERSON_SUBJECT = re.compile(
    r"\b(?:trace|reconstruct|follow)\s+"
    r"((?:[A-Z][A-Za-z'\u2019-]+(?:\s+[A-Z][A-Za-z'\u2019-]+){0,3}))"
    r"(?=\s+(?:from|route|during|in|throughout|for)\b|['\u2019]s\b|[.;]|$)",
    re.IGNORECASE,
)
_FROM_MARKER = re.compile(r"\bfrom\b", re.IGNORECASE)
_OPTIONAL_THE = re.compile(r"^\s*(?:the\s+)?", re.IGNORECASE)
_VIA_MARKER = re.compile(r"\s+(?:across|via|through)\s+(?:the\s+)?", re.IGNORECASE)
_DEST_MARKER = re.compile(r"\s+(?:to|toward|towards|into)\s+(?:the\s+)?", re.IGNORECASE)
_STRICT_DEST_MARKER = re.compile(r"\s+(?:to|into)\s+(?:the\s+)?", re.IGNORECASE)
_ROUTE_CONTINUATION = re.compile(
    r"(?:\b(?:and|or)\s+(?:then\s+)?(?:from|to|toward|towards|into)\b|\s*/\s*|\bonward\b|"
    r"\bbefore\s+proceeding\b|\blater\s+moving\b|(?:,\s*)?\bthen\s+(?:to|toward|towards|into|from)\b)",
    re.IGNORECASE,
)
_PARTIAL_ROUTE_CONTINUATION = re.compile(
    r"(?:,\s*)?\bthen\s+(?:to|toward|towards|into)\b|\bonward\b|"
    r"\bbefore\s+proceeding\b|\blater\s+moving\b",
    re.IGNORECASE,
)
_COMPETING_CONTEXT_ROUTE = re.compile(
    r"\bafter\s+which\b|\bfollowed\s+by\s+movement\s+from\b",
    re.IGNORECASE,
)
_SUBORDINATE_CONTEXT_START = re.compile(
    r"^\s*(?:while|when|because|although)\b",
    re.IGNORECASE,
)
_MOVEMENT_PREDICATE = re.compile(
    r"\b(?:moved|moving|marched|marching|movement)\b",
    re.IGNORECASE,
)
_SUBJECT_BEFORE_PREDICATE = re.compile(
    r"((?:[A-Z][A-Za-z'\u2019-]+(?:\s+[A-Z][A-Za-z'\u2019-]+){0,3})\s+)$",
)
_SUBJECT_STOPWORDS = frozenset({
    "and", "or", "then", "while", "when", "from", "to", "movement", "trace", "show", "follow", "reconstruct",
})
_CLAUSE_BOUNDARY = re.compile(
    r"\b(?:before|after|while|when|because|although|following|once|prior\s+to|"
    r"until|upon|subsequently|afterward|afterwards|"
    r"during|in|throughout|for)\b|(?:,\s*(?:trace|show|reconstruct|follow)\b)|[,;:\u2014]|[.;]|$",
    re.IGNORECASE,
)
_EPISODE_MARKER = re.compile(r"\b(?:during|in|throughout|for)\b", re.IGNORECASE)
_EPISODE_NAMED_CAMPAIGN = re.compile(
    r"\b(?:during|in|throughout|for)\s+(?:the\s+)?Campaign\s+([A-Z][A-Za-z'\u2019-]*)\b",
    re.IGNORECASE,
)
_EPISODE_WAR = re.compile(
    r"\b(?:during|in|throughout|for)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z'\u2019-]+(?:\s+(?:the\s+)?[A-Z][A-Za-z'\u2019-]+){0,4})\s+war\b",
    re.IGNORECASE,
)
_EPISODE_EXPEDITION = re.compile(
    r"\b(?:during|in|throughout|for)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z'\u2019-]+(?:\s+(?:the\s+)?[A-Z][A-Za-z'\u2019-]+){0,4})\s+expedition\b",
    re.IGNORECASE,
)
_EPISODE_CAMPAIGN = re.compile(
    r"\b(?:during|in|throughout|for)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z'\u2019-]+(?:\s+(?:the\s+)?[A-Z][A-Za-z'\u2019-]+){0,4})\s+campaign\b",
    re.IGNORECASE,
)
_EPISODE_TAIL_CANDIDATE = re.compile(
    r"(?:[,;]\s*)?"
    r"(?:"
    r"(?:and|or|versus|vs|before|after|then|followed\s+by|subsequently)\s+(?:the\s+)?"
    r"|/\s*"
    r")"
    r"Campaign\s+(?P<name>[A-Z][A-Za-z'\u2019-]*)\b",
    re.IGNORECASE,
)
_CONTEXT_MARKER = re.compile(
    r"\b(?:before|after|while|when|because|although|following|once|prior\s+to|"
    r"until|upon|subsequently|afterward|afterwards|followed\s+by)\b",
    re.IGNORECASE,
)
_LEADING_CONTEXT_END = re.compile(
    r",\s*(?:trace|show|reconstruct|follow)\b|[;]|$",
    re.IGNORECASE,
)
_PRIMARY_REQUEST = re.compile(r"\b(?:trace|show|reconstruct|follow)\b", re.IGNORECASE)


class QuerySpanRole(str, Enum):
    SUBJECT = "SUBJECT"
    ORIGIN = "ORIGIN"
    VIA = "VIA"
    DESTINATION = "DESTINATION"
    EPISODE = "EPISODE"
    CONTEXT = "CONTEXT"
    TIME = "TIME"
    UNCLAIMED = "UNCLAIMED"


class SpanStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"


class _FrameGroup(str, Enum):
    SINGLE_PRIMARY = "SINGLE_PRIMARY"
    COMPETING = "COMPETING"
    PARTIAL_CONTINUATION = "PARTIAL_CONTINUATION"


@dataclass(frozen=True)
class _RouteFrameCandidate:
    origin: str
    destination: str
    via: str | None
    origin_span: tuple[int, int]
    via_span: tuple[int, int] | None
    dest_span: tuple[int, int]
    endpoint_strict: bool
    in_context: bool
    competing: bool
    frame_subject: str | None


_ROUTE_BLOCKING = frozenset({
    QuerySpanRole.SUBJECT,
    QuerySpanRole.ORIGIN,
    QuerySpanRole.VIA,
    QuerySpanRole.DESTINATION,
})
_CONTEXT_BLOCKING = frozenset({QuerySpanRole.CONTEXT})


@dataclass(frozen=True)
class QueryRoleSpan:
    role: QuerySpanRole
    start: int
    end: int
    text: str
    status: SpanStatus = SpanStatus.CONFIRMED


@dataclass(frozen=True)
class QueryScopeParse:
    subject: str | None
    origin: str | None
    destination: str | None
    episode: str | None
    has_episode_constraint: bool
    spans: tuple[QueryRoleSpan, ...]
    episode_ambiguous: bool = False
    endpoint_strict: bool = True
    has_endpoint_constraint: bool = False
    temporal_start: int | None = None
    temporal_end: int | None = None
    temporal_precision: TemporalPrecision | None = None
    has_temporal_constraint: bool = False


class _SpanRegistry:
    def __init__(self, text: str) -> None:
        self.text = text
        self.spans: list[QueryRoleSpan] = []

    def overlaps(self, start: int, end: int, *, blocking: frozenset[QuerySpanRole]) -> bool:
        return any(
            span.role in blocking and start < span.end and end > span.start
            for span in self.spans
        )

    def claim(
        self,
        role: QuerySpanRole,
        start: int,
        end: int,
        *,
        status: SpanStatus = SpanStatus.CONFIRMED,
        blocking: frozenset[QuerySpanRole] | None = None,
    ) -> bool:
        block = blocking if blocking is not None else _ROUTE_BLOCKING
        if role == QuerySpanRole.EPISODE:
            if self.overlaps(start, end, blocking=_ROUTE_BLOCKING):
                return False
        elif self.overlaps(start, end, blocking=block):
            return False
        text = self.text[start:end].strip(" ,.;:\u2014")
        if not text:
            return False
        self.spans.append(QueryRoleSpan(role, start, end, text, status))
        return True

    def inside_claimed(self, start: int, end: int, *, blocking: frozenset[QuerySpanRole]) -> bool:
        return any(
            span.role in blocking and span.start <= start and span.end >= end
            for span in self.spans
        )


def _clean(value: str) -> str:
    return value.strip(" ,.;:\u2014")


def _clause_boundary(text: str, start: int) -> int:
    match = _CLAUSE_BOUNDARY.search(text, start)
    return match.start() if match else len(text)


def _has_route_continuation_after(text: str, dest_end: int) -> bool:
    tail = text[dest_end:]
    if re.search(r"\s*/\s*(?:the\s+)?Campaign\b", tail, re.IGNORECASE):
        return False
    if re.search(r"\b(?:and|or)\s+(?:the\s+)?Campaign\b", tail, re.IGNORECASE):
        return False
    return _ROUTE_CONTINUATION.search(tail) is not None


def _later_movement_frame_start(text: str, dest_start: int, limit: int) -> int | None:
    """Start of a later from–to candidate, including optional subject + movement predicate."""
    search_to = min(limit, len(text))
    for from_match in _FROM_MARKER.finditer(text, dest_start, search_to):
        if _DEST_MARKER.search(text, from_match.end()) is None:
            continue
        prefix = text[dest_start:from_match.start()]
        start = from_match.start()
        predicate = None
        for match in _MOVEMENT_PREDICATE.finditer(prefix):
            predicate = match
        if predicate is not None:
            start = dest_start + predicate.start()
            subject = _SUBJECT_BEFORE_PREDICATE.search(prefix[:predicate.start()])
            if subject is not None:
                start = dest_start + subject.start(1)
        if start > dest_start:
            return start
    return None


def _trim_destination_text(value: str) -> str:
    tokens = _clean(value).split()
    while len(tokens) > 1 and tokens[-1][:1].islower():
        tokens.pop()
    return " ".join(tokens)


def _next_additional_dest_marker(text: str, dest_start: int, end: int) -> re.Match[str] | None:
    pos = dest_start
    while pos < end:
        match = _DEST_MARKER.search(text, pos, end)
        if match is None:
            return None
        prefix = text[max(0, match.start() - 6):match.start()]
        if prefix.casefold().rstrip().endswith("prior"):
            pos = match.end()
            continue
        return match
    return None


def _destination_span_end(text: str, dest_start: int, limit: int | None = None) -> int:
    end = _clause_boundary(text, dest_start)
    later_frame = _later_movement_frame_start(text, dest_start, end)
    if later_frame is not None and dest_start < later_frame < end:
        end = later_frame
    if limit is not None and limit < end:
        end = limit
    later_dest = _next_additional_dest_marker(text, dest_start, end)
    if later_dest is not None and later_dest.start() < end:
        end = later_dest.start()
    for pattern in (_PARTIAL_ROUTE_CONTINUATION, _ROUTE_CONTINUATION):
        match = pattern.search(text, dest_start, end)
        if match and match.start() < end:
            end = match.start()
    return end


def _episode_scan_limit(text: str, marker_end: int) -> int:
    restart = re.search(
        r"[,;]\s*(?:trace|show|reconstruct|follow)\b",
        text[marker_end:],
        re.IGNORECASE,
    )
    if restart:
        return marker_end + restart.start()
    return len(text)


def _episode_labels_extent_end(text: str, marker_start: int, scan_limit: int) -> int:
    fragment = text[marker_start:scan_limit]
    marker = _EPISODE_MARKER.search(fragment)
    if marker is None:
        return scan_limit
    last_end = marker_start + marker.end()
    named = _EPISODE_NAMED_CAMPAIGN.match(fragment[marker.start():])
    if named:
        last_end = marker_start + marker.start() + named.end()
    tail = fragment[marker.end():]
    tail_base = marker_start + marker.end()
    for match in _EPISODE_TAIL_CANDIDATE.finditer(tail):
        end = tail_base + match.end()
        if end > last_end:
            last_end = end
    return last_end


def _episode_span_end(text: str, marker_start: int, marker_end: int) -> int:
    """Episode spans include all structured candidates before the next primary request."""
    scan_limit = _episode_scan_limit(text, marker_end)
    labels = _episode_labels_in_span(text, marker_start, scan_limit)
    if len(labels) >= 2:
        return _episode_labels_extent_end(text, marker_start, scan_limit)
    pos = marker_end
    while True:
        boundary = _CLAUSE_BOUNDARY.search(text, pos)
        if boundary is None:
            return scan_limit
        marker_at = boundary.start()
        after = text[marker_at:]
        if _EPISODE_TAIL_CANDIDATE.match(after):
            pos = marker_at + 1
            continue
        return marker_at


def _context_span_end(text: str, marker_start: int, marker_end: int, *, subject_start: int | None) -> int:
    primary = _PRIMARY_REQUEST.search(text)
    primary_start = primary.start() if primary else None
    anchor = subject_start if subject_start is not None else primary_start
    if anchor is not None and marker_start < anchor:
        match = _LEADING_CONTEXT_END.search(text, marker_end)
        return match.start() if match else len(text)
    return len(text)


def _claim_subject(registry: _SpanRegistry) -> str | None:
    text = registry.text
    for pattern in (_BROAD_POSSESSIVE_ROUTE_SUBJECT, _POSSESSIVE_ROUTE_SUBJECT, _TRACE_PERSON_SUBJECT):
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        names = {
            _clean(match.group(1)).title()
            for match in matches
            if _clean(match.group(1)).casefold() != "movement"
        }
        if len(names) != 1:
            continue
        name = next(iter(names))
        match = matches[0]
        registry.claim(QuerySpanRole.SUBJECT, match.start(1), match.end(1), blocking=frozenset())
        return name
    return None


def _context_span_for_position(registry: _SpanRegistry, pos: int) -> QueryRoleSpan | None:
    for span in registry.spans:
        if span.role is QuerySpanRole.CONTEXT and span.start <= pos < span.end:
            return span
    return None


def _context_route_is_competing(text: str, context_span: QueryRoleSpan) -> bool:
    fragment = text[context_span.start:context_span.end]
    if _SUBORDINATE_CONTEXT_START.match(fragment):
        return False
    has_complete_frame = (
        _FROM_MARKER.search(fragment) is not None
        and _DEST_MARKER.search(fragment) is not None
    )
    if has_complete_frame and _MOVEMENT_PREDICATE.search(fragment):
        return True
    return _COMPETING_CONTEXT_ROUTE.search(fragment) is not None


def _normalize_subject_name(name: str) -> str | None:
    parts = name.split()
    cleaned: list[str] = []
    for part in parts:
        if part.casefold() in _SUBJECT_STOPWORDS:
            break
        cleaned.append(part)
    if not cleaned:
        return None
    return " ".join(cleaned).title()


def _subject_before_from(text: str, from_start: int) -> str | None:
    if text[from_start:from_start + 4].casefold() != "from":
        return None
    local = text[max(0, from_start - 80):from_start + 4]
    for match in reversed(list(re.finditer(
        r"\b(?:(?:and|or|then)\s+)?((?:[A-Z][A-Za-z'\u2019-]+(?:\s+[A-Z][A-Za-z'\u2019-]+){0,2}))\s+from\b",
        local,
        re.IGNORECASE,
    ))):
        name = _normalize_subject_name(_clean(match.group(1)))
        if name:
            return name
    segment = text[:from_start + 4]
    trace_matches = list(_TRACE_PERSON_SUBJECT.finditer(segment))
    if trace_matches:
        name = _normalize_subject_name(_clean(trace_matches[-1].group(1)))
        if name:
            return name
    return None


def _frame_subjects_from_candidates(candidates: list[_RouteFrameCandidate]) -> frozenset[str]:
    subjects: set[str] = set()
    for candidate in candidates:
        if candidate.in_context or not candidate.frame_subject:
            continue
        subjects.add(candidate.frame_subject.casefold())
    return frozenset(subjects)


def _build_route_frame_candidate(
    text: str,
    from_match: re.Match[str],
    *,
    next_from_start: int,
    registry: _SpanRegistry,
) -> _RouteFrameCandidate | None:
    start = from_match.end()
    start_match = _OPTIONAL_THE.match(text, start)
    origin_start = start_match.end() if start_match else start
    context_span = _context_span_for_position(registry, from_match.start())
    in_context = context_span is not None
    if not in_context and registry.inside_claimed(origin_start, origin_start + 1, blocking=_ROUTE_BLOCKING | _CONTEXT_BLOCKING):
        return None
    if not in_context and registry.overlaps(origin_start, origin_start + 1, blocking=_ROUTE_BLOCKING | _CONTEXT_BLOCKING):
        return None
    via_match = _VIA_MARKER.search(text, origin_start, next_from_start)
    dest_match = _DEST_MARKER.search(text, origin_start, next_from_start)
    if dest_match is None:
        return None
    frame_subject = _subject_before_from(text, from_match.start())
    endpoint_strict = _STRICT_DEST_MARKER.search(text, origin_start, next_from_start) is not None
    if via_match and via_match.start() < dest_match.start():
        origin_text = _clean(text[origin_start:via_match.start()])
        via_start = via_match.end()
        dest_after_via = _DEST_MARKER.search(text, via_start, next_from_start)
        if dest_after_via is None:
            return None
        via_text = _clean(text[via_start:dest_after_via.start()])
        dest_start = dest_after_via.end()
        dest_end = _destination_span_end(text, dest_start, limit=next_from_start)
        dest_text = _trim_destination_text(text[dest_start:dest_end])
        if not (origin_text and via_text and dest_text):
            return None
        endpoint_strict = _STRICT_DEST_MARKER.search(text, via_start, next_from_start) is not None
        competing = not in_context or (context_span is not None and _context_route_is_competing(text, context_span))
        return _RouteFrameCandidate(
            origin=origin_text,
            destination=dest_text,
            via=via_text,
            origin_span=(origin_start, via_match.start()),
            via_span=(via_start, dest_after_via.start()),
            dest_span=(dest_start, dest_end),
            endpoint_strict=endpoint_strict,
            in_context=in_context,
            competing=competing,
            frame_subject=frame_subject,
        )
    if via_match and via_match.start() > dest_match.start():
        origin_text = _clean(text[origin_start:dest_match.start()])
        dest_start = dest_match.end()
        dest_end = via_match.start()
        via_start = via_match.end()
        via_end = _clause_boundary(text, via_start)
        dest_text = _trim_destination_text(text[dest_start:dest_end])
        via_text = _clean(text[via_start:via_end])
        if not (origin_text and dest_text and via_text):
            return None
        competing = not in_context or (context_span is not None and _context_route_is_competing(text, context_span))
        return _RouteFrameCandidate(
            origin=origin_text,
            destination=dest_text,
            via=via_text,
            origin_span=(origin_start, dest_match.start()),
            via_span=(via_start, via_end),
            dest_span=(dest_start, dest_end),
            endpoint_strict=endpoint_strict,
            in_context=in_context,
            competing=competing,
            frame_subject=frame_subject,
        )
    origin_text = _clean(text[origin_start:dest_match.start()])
    dest_start = dest_match.end()
    dest_end = _destination_span_end(text, dest_start, limit=next_from_start)
    dest_text = _trim_destination_text(text[dest_start:dest_end])
    if not (origin_text and dest_text):
        return None
    competing = not in_context or (context_span is not None and _context_route_is_competing(text, context_span))
    return _RouteFrameCandidate(
        origin=origin_text,
        destination=dest_text,
        via=None,
        origin_span=(origin_start, dest_match.start()),
        via_span=None,
        dest_span=(dest_start, dest_end),
        endpoint_strict=endpoint_strict,
        in_context=in_context,
        competing=competing,
        frame_subject=frame_subject,
    )


def _discover_route_frame_candidates(registry: _SpanRegistry) -> list[_RouteFrameCandidate]:
    text = registry.text
    from_matches = list(_FROM_MARKER.finditer(text))
    candidates: list[_RouteFrameCandidate] = []
    for index, from_match in enumerate(from_matches):
        next_start = from_matches[index + 1].start() if index + 1 < len(from_matches) else len(text)
        candidate = _build_route_frame_candidate(
            text,
            from_match,
            next_from_start=next_start,
            registry=registry,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _has_partial_route_continuation(text: str, candidate: _RouteFrameCandidate) -> bool:
    dest_end = candidate.dest_span[1]
    tail = text[dest_end:]
    if _PARTIAL_ROUTE_CONTINUATION.search(tail):
        return True
    if _has_route_continuation_after(text, dest_end):
        return True
    rest_end = _clause_boundary(text, dest_end)
    extra_dest = _next_additional_dest_marker(text, dest_end, rest_end)
    if extra_dest is not None and _FROM_MARKER.search(text, dest_end, rest_end) is None:
        return True
    if (
        re.search(r"\s+(?:toward|towards|into)\b", tail, re.IGNORECASE)
        and _FROM_MARKER.search(tail) is None
    ):
        return True
    return False


def _classify_route_frame_group(candidates: list[_RouteFrameCandidate], text: str) -> _FrameGroup:
    competing = [candidate for candidate in candidates if candidate.competing]
    if len(competing) >= 2:
        return _FrameGroup.COMPETING
    primary = [candidate for candidate in candidates if not candidate.in_context]
    if len(primary) == 1 and _has_partial_route_continuation(text, primary[0]):
        return _FrameGroup.PARTIAL_CONTINUATION
    if len(primary) >= 2:
        return _FrameGroup.COMPETING
    return _FrameGroup.SINGLE_PRIMARY


def _claim_route_frame(
    registry: _SpanRegistry,
) -> tuple[str | None, str | None, str | None, bool, frozenset[str], bool]:
    candidates = _discover_route_frame_candidates(registry)
    frame_subjects = _frame_subjects_from_candidates(candidates)
    if not candidates:
        return None, None, None, True, frame_subjects, False
    group = _classify_route_frame_group(candidates, registry.text)
    if group is not _FrameGroup.SINGLE_PRIMARY:
        return None, None, None, True, frame_subjects, True
    primary_candidates = [candidate for candidate in candidates if not candidate.in_context]
    if not primary_candidates:
        return None, None, None, True, frame_subjects, False
    primary = primary_candidates[0]
    registry.claim(QuerySpanRole.ORIGIN, primary.origin_span[0], primary.origin_span[1])
    if primary.via_span is not None:
        registry.claim(QuerySpanRole.VIA, primary.via_span[0], primary.via_span[1])
    registry.claim(QuerySpanRole.DESTINATION, primary.dest_span[0], primary.dest_span[1])
    has_endpoint_constraint = bool(primary.origin and primary.destination)
    return primary.origin, primary.via, primary.destination, primary.endpoint_strict, frame_subjects, has_endpoint_constraint


def _episode_label_at(text: str, marker_start: int) -> str | None:
    fragment = text[marker_start:]
    for pattern, formatter in (
        (_EPISODE_NAMED_CAMPAIGN, lambda match: f"Campaign {match.group(1)}"),
        (_EPISODE_WAR, lambda match: f"{match.group(1)} War"),
        (_EPISODE_EXPEDITION, lambda match: f"{match.group(1)} Expedition"),
        (_EPISODE_CAMPAIGN, lambda match: f"{match.group(1)} Campaign"),
    ):
        match = pattern.match(fragment)
        if match:
            return formatter(match)
    return None


def _episode_labels_in_span(text: str, start: int, end: int) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    primary = _episode_label_at(text, start)
    if primary:
        labels.append(primary)
        seen.add(primary.casefold())
    fragment = text[start:end]
    marker = _EPISODE_MARKER.search(fragment)
    if marker is None:
        return labels
    tail = fragment[marker.end():]
    for candidate in _EPISODE_TAIL_CANDIDATE.finditer(tail):
        candidate_label = f"Campaign {candidate.group('name')}"
        candidate_key = candidate_label.casefold()
        if candidate_key not in seen:
            seen.add(candidate_key)
            labels.append(candidate_label)
    return labels


def _claim_episodes(registry: _SpanRegistry) -> tuple[str | None, bool, bool]:
    text = registry.text
    collected: list[str] = []
    ambiguous = False
    has_constraint = False
    primary_markers: list[re.Match[str]] = []
    for marker in _EPISODE_MARKER.finditer(text):
        if registry.inside_claimed(marker.start(), marker.end(), blocking=_CONTEXT_BLOCKING):
            continue
        if registry.overlaps(marker.start(), marker.end(), blocking=_ROUTE_BLOCKING):
            continue
        if _episode_label_at(text, marker.start()) is None:
            continue
        primary_markers.append(marker)
        has_constraint = True
        tail_end = _episode_span_end(text, marker.start(), marker.end())
        span_labels = _episode_labels_in_span(text, marker.start(), tail_end)
        if len(span_labels) >= 2:
            ambiguous = True
        collected.extend(span_labels)
        registry.claim(
            QuerySpanRole.EPISODE,
            marker.start(),
            tail_end,
            status=SpanStatus.AMBIGUOUS if len(span_labels) >= 2 else SpanStatus.CONFIRMED,
            blocking=frozenset(),
        )
    if len(primary_markers) >= 2:
        ambiguous = True
    unique = list(dict.fromkeys(collected))
    if ambiguous or len(unique) >= 2:
        return None, has_constraint, True
    return (unique[0] if unique else None), has_constraint, False


def _claim_context(registry: _SpanRegistry) -> None:
    text = registry.text
    subject_span = next((span for span in registry.spans if span.role is QuerySpanRole.SUBJECT), None)
    subject_start = subject_span.start if subject_span else None
    for match in _CONTEXT_MARKER.finditer(text):
        if registry.overlaps(match.start(), match.end(), blocking=_ROUTE_BLOCKING):
            continue
        if match.group(0).casefold() == "subsequently":
            after = text[match.end():]
            if re.match(r"\s+(?:the\s+)?Campaign\s+[A-Z]", after, re.IGNORECASE):
                before = text[max(0, match.start() - 40):match.start()]
                if re.search(r"Campaign\s+[A-Z]", before, re.IGNORECASE):
                    continue
        span_end = _context_span_end(
            text,
            match.start(),
            match.end(),
            subject_start=subject_start,
        )
        registry.claim(
            QuerySpanRole.CONTEXT,
            match.start(),
            span_end,
            blocking=frozenset({QuerySpanRole.EPISODE, QuerySpanRole.SUBJECT}),
        )


def _claim_time(
    registry: _SpanRegistry,
) -> tuple[int | None, int | None, TemporalPrecision | None, bool]:
    from backend.app.routes.temporal import EvidenceTemporalResolver

    text = registry.text
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve(text, "query")
    claimed = False
    for pattern in (*EvidenceTemporalResolver._RANGE_PATTERNS, *EvidenceTemporalResolver._YEAR_PATTERNS):
        for match in pattern.finditer(text):
            start, end = match.span()
            prefix = text[:start]
            lead = re.search(r"(?:in|during)\s+$", prefix, re.IGNORECASE)
            if lead:
                start = lead.start()
            if registry.claim(QuerySpanRole.TIME, start, end, blocking=_ROUTE_BLOCKING):
                claimed = True
    has_constraint = claimed or bool(readings) or "TEMPORAL_CONFLICT" in codes
    if not has_constraint:
        return None, None, None, False
    if "TEMPORAL_CONFLICT" in codes or not readings:
        return None, None, None, True
    item = readings[0]
    if item.normalized_start is None:
        return None, None, item.precision, True
    start = int(item.normalized_start)
    end = int(item.normalized_end) if item.normalized_end is not None else start
    if start > end:
        start, end = end, start
    return start, end, item.precision, True


def parse_query_scope(text: str) -> QueryScopeParse:
    registry = _SpanRegistry(text or "")
    subject = _claim_subject(registry)
    _claim_context(registry)
    origin, _via, destination, endpoint_strict, frame_subjects, has_endpoint_constraint = _claim_route_frame(registry)
    if len(frame_subjects) > 1:
        subject = None
        registry.spans = [span for span in registry.spans if span.role is not QuerySpanRole.SUBJECT]
    temporal_start, temporal_end, temporal_precision, has_temporal = _claim_time(registry)
    episode, has_episode, episode_ambiguous = _claim_episodes(registry)
    return QueryScopeParse(
        subject=subject,
        origin=origin,
        destination=destination,
        episode=episode,
        has_episode_constraint=has_episode,
        spans=tuple(registry.spans),
        episode_ambiguous=episode_ambiguous,
        endpoint_strict=endpoint_strict if origin and destination else True,
        has_endpoint_constraint=has_endpoint_constraint,
        temporal_start=temporal_start,
        temporal_end=temporal_end,
        temporal_precision=temporal_precision,
        has_temporal_constraint=has_temporal,
    )


def parse_query_scope_spans(text: str) -> tuple[QueryRoleSpan, ...]:
    return parse_query_scope(text).spans


def build_query_route_scope(contexts: tuple[str, ...] | None) -> QueryScopeParse:
    if not contexts:
        return QueryScopeParse(None, None, None, None, False, (), False, True)
    for context in contexts:
        if not (context or "").strip():
            continue
        return parse_query_scope(context)
    return QueryScopeParse(None, None, None, None, False, (), False, True)
