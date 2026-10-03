from __future__ import annotations
import json, logging, re
from time import perf_counter
from backend.app.agent.prompts import SYSTEM_PROMPT
from backend.app.agent.tools import incomplete_movement_answer_context
from backend.app.agent.evidence_support import final_answer_support, assess_evidence_support, assess_final_answer_provenance, event_relation_supports_answer, has_unsupported_route_pattern, render_evidence_citations, validate_evidence_citations, validate_evidence_selection
from backend.app.agent.route_orchestration import (
    PRE_ROUTE_SUPPRESSION_MESSAGE,
    duplicate_attempt_payload,
    lookup_prior_resolve,
    prior_tool_attempt_key,
    route_is_ready,
    route_ready_tool_schemas,
    should_suppress_pre_route_tool,
    visible_historical_events,
)
from backend.app.models import AgentProviderCallTiming, AgentState, AgentToolHistoryEntry
from backend.app.agent.route_summary import admitted_route_summary
from backend.app.route_result_status import RouteResultStatus, derive_route_result_status

logger = logging.getLogger(__name__)
_ROUTE_TERMS = ("route", "路线", "行军", "进军", "绘制", "展示")
_GEOGRAPHY_TERMS = ("distance", "elevation", "coordinate", "距离", "高程", "坐标")
_MOVEMENT_VERBS = (
    "move", "moved", "moving", "march", "marched", "marching", "sail", "sailed", "sailing",
    "travel", "traveled", "travelled", "traveling", "travelling", "advance", "advanced", "advancing",
    "proceed", "proceeded", "withdraw", "withdrew", "retreat", "retreated", "hasten", "hastened",
)
_TRANSIT_VERBS = ("cross", "crossed", "crossing", "traverse", "traversed", "traversing")
_DIRECTIONAL_MARKERS = (
    " from ", " into ", " across ", " between ", " through ", " toward", " towards", " onto ", " over ",
    "从", "到", "进入", "越过", "翻越",
)
_DISPLAY_VERBS = ("show", "trace", "map", "follow", "reconstruct", "display")
_MOVEMENT_NOUNS = (
    "campaign movements", "campaign movement", "movements", "movement", "routes", "route",
    "journeys", "journey", "marches", "march", "advances", "advance", "retreats", "retreat",
)
_ANALYTICAL_MOVEMENT_CUES = (
    "consequences of", "why were", "why was", "why did", "what caused", "what were the",
    "explain ", "describe ", "strategy", "how important", "tell me about", "significance of",
    "impact of", "political consequences", "important were", "important was",
)
_INSUFFICIENT_TERMS = ("insufficient", "cannot build", "unable to build", "evidence is not enough", "证据不足", "无法生成", "不能生成")
COMPLETION_TOOL_BY_OUTPUT = {"historical_route": "build_historical_route"}
POST_ROUTE_UPSTREAM_TOOLS = frozenset({
    "search_historical_evidence",
    "resolve_ancient_place",
    "calculate_distance",
    "get_elevation",
    "get_elevation_profile",
})
POST_ROUTE_SUPPRESSION_MESSAGE = (
    "Historical route is already built from the current Evidence. "
    "Do not call search_historical_evidence, resolve_ancient_place, or build_historical_route again. "
    "Proceed to submit_grounded_answer with a grounded summary, or finish if route presentation is complete."
)
ROUTE_PROSE_GROUNDING_FALLBACK = (
    "A structured route was built from the current Evidence, but the generated explanation "
    "did not complete grounded-answer submission. Use the verified route nodes, fragments, and citations."
)
ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION = (
    "A structured route was built from the current Evidence, but the generated explanation "
    "did not complete grounded-answer submission. Use the verified route nodes and citations."
)
GENERIC_GROUNDING_GUARDRAIL = (
    "The current retrieved historical evidence is insufficient to support a reliable answer."
)
NO_ROUTE_TERMINAL_GUARDRAIL = (
    "The current retrieved historical evidence is insufficient to support a reliable "
    "HistoricalRoute, so the system will not add unsupported places or route details."
)


# Historical non-completion, distinct from epistemic "cannot establish a route".
_TERMINAL_NON_COMPLETION = re.compile(
    r"(?P<PREVENTED>\b(?:prevented|blocked)\b|被阻止)"
    r"|(?P<ABORTED>\b(?:aborted|abandoned)\b|\b(?:ended|stopped|failed)\b[^.!?;]{0,80}\bbefore\b|中止|放弃)"
    r"|(?P<PLANNED>\b(?:planned|intended|unexecuted)\b|计划前往)"
    r"|(?P<NEGATED>\b(?:did\s+not|never|failed\s+to)\s+(?:reach|arrive|enter|complete)\b"
    r"|\bnot\s+(?:(?:a|to|have|been)\s+)*(?:completed|complete|executed|reached|arrived)\b"
    r"|未(?:完成|执行|抵达|到达|进入))", re.IGNORECASE,
)
_TERMINAL_CLAUSES = re.compile(
    r"[.;!?\n,。！？，；]+|\b(?:but|however|nevertheless|yet|and|so|therefore|although|though|because|while)\b|但是|然而|不过|但|因此",
    re.IGNORECASE,
)


def _has_terminal_simulation(state: AgentState) -> bool:
    presentation = state.historical_route_presentation or {}
    return any(
        isinstance(feature, dict)
        and (feature.get("geometry") or {}).get("type") in {"LineString", "MultiLineString"}
        and bool((feature.get("geometry") or {}).get("coordinates"))
        and (feature.get("properties") or {}).get("segment_role") != "failed_gap"
        for feature in (presentation.get("geojson") or {}).get("features") or []
    )


def _is_simulation_clause(clause: str) -> bool:
    return bool(re.search(
        r"^\s*(?:(?:the|a|separate|displayed)\s+)*"
        r"(?:gis\s+simulation\b|simulated\s+(?:route|path|geometry)\b|GIS模拟|仿真路径|模拟路线)", clause, re.I,
    )) and not re.search(r"\b(?:historical|attested|actually|documentary)\b|史实|历史路线", clause, re.I)


def _historical_terminal_text(answer: str | None, state: AgentState) -> str | None:
    if not answer or not _has_terminal_simulation(state):
        return answer
    return ". ".join(clause for clause in _TERMINAL_CLAUSES.split(answer) if not _is_simulation_clause(clause))


def _non_completion_reply(state: AgentState) -> str | None:
    """Quote existing documentary state; candidate prose supplies no facts."""
    sources = {item.id: item for item in state.historical_evidence[:8]}
    reports = [
        (fact["source_statement"], tuple(ref for ref in fact["evidence_refs"] if ref in sources))
        for fact in incomplete_movement_answer_context(state)
    ]
    if not reports:
        reports = [
            (sentence.strip(), (item.id,)) for item in sources.values()
            for sentence in re.split(r"(?<=[.!?。！？])\s*", item.text)
            if _TERMINAL_NON_COMPLETION.search(sentence)
        ]
    reports = list(dict.fromkeys((text, refs) for text, refs in reports if text and refs))[:2]
    if not reports:
        return None
    quoted = " ".join(
        f'Retrieved evidence reports: “{text}” {render_evidence_citations(refs, list(sources.values()))}'
        for text, refs in reports
    )
    conclusion = (
        " No completed historical route is established by this result." if state.historical_route is None else
        " This non-completion does not establish an additional completed route."
    )
    return quoted + conclusion


def _negative_prose_is_state_supported(answer: str, state: AgentState) -> bool:
    # Preserve documentary clauses or generic paraphrases of known outcomes.
    # Named claims must match documentary wording rather than shared keywords.
    text = re.sub(r"\[Evidence:[^\]]*\]", "", answer, flags=re.IGNORECASE)
    clauses = _TERMINAL_CLAUSES.split(text)
    negative = [clause.strip().casefold() for clause in clauses if _TERMINAL_NON_COMPLETION.search(clause)]
    cited = set(re.findall(r"\[Evidence:\s*(.*?)\s+—", answer, re.I))
    selected = [item for item in state.historical_evidence if not cited or item.id in cited]
    sources = [item.text.casefold() for item in selected]
    documentary_clauses = {
        clause.strip() for source in sources for clause in _TERMINAL_CLAUSES.split(source)
    }
    if negative and all(clause in documentary_clauses for clause in negative):
        return True
    # Generic non-completion paraphrases may retain their wording, but named
    # actors/places require exact documentary clauses rather than keyword overlap.
    assessment = assess_final_answer_provenance(text, state.user_query or "", state.historical_evidence)
    if any(item.entity_like for item in assessment.candidate_entities):
        return False
    matches = list(_TERMINAL_NON_COMPLETION.finditer(text))
    if any(re.search(r"\b(?:not|never)\s+(?:\w+ly\s+)*$", text[:match.start()], re.I) for match in matches):
        return False
    selected_ids = {item.id for item in selected}
    supported = {fact["outcome"] for fact in incomplete_movement_answer_context(state)
                 if set(fact["evidence_refs"]).issubset(selected_ids)}
    requested = {key for match in _TERMINAL_NON_COMPLETION.finditer(text)
                 for key, value in match.groupdict().items() if value}
    if supported.intersection({"ABORTED", "PREVENTED", "FAILED_TO_REACH"}):
        supported.add("NEGATED")
    return bool(requested) and requested.issubset(supported)


def _fingerprint(name: str, arguments: dict) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"


def _has_analytical_movement_question(normalized: str) -> bool:
    padded = f" {normalized} "
    return any(cue in padded or normalized.startswith(cue.strip()) for cue in _ANALYTICAL_MOVEMENT_CUES)


def _contains_display_verb(normalized: str) -> bool:
    if "can you show" in normalized or "i want to trace" in normalized:
        return True
    padded = f" {normalized} "
    return any(
        padded.startswith(f"{verb} ") or f" {verb} " in padded or f" {verb} me " in padded
        for verb in _DISPLAY_VERBS
    )


def _contains_movement_object(normalized: str) -> bool:
    return any(re.search(rf"\b{re.escape(noun)}\b", normalized) for noun in _MOVEMENT_NOUNS)


def _contains_movement_verb(normalized: str) -> bool:
    return any(re.search(rf"\b{re.escape(verb)}\b", normalized) for verb in _MOVEMENT_VERBS + _TRANSIT_VERBS)


def _has_where_or_how_movement_question(normalized: str) -> bool:
    padded = f" {normalized} "
    if not (
        padded.startswith("where ")
        or " where " in padded
        or padded.startswith("how ")
        or " how " in padded
    ):
        return False
    if not _contains_movement_verb(normalized):
        return False
    if any(marker in padded for marker in _DIRECTIONAL_MARKERS):
        return True
    if " from " in padded and " to " in padded:
        return True
    if _contains_movement_object(normalized):
        return True
    if " during " in padded or " in the campaign" in padded:
        return True
    return False


def _has_movement_display_intent(normalized: str) -> bool:
    if _contains_display_verb(normalized) and (
        _contains_movement_object(normalized) or _contains_movement_verb(normalized)
    ):
        return True
    return _has_where_or_how_movement_question(normalized)


def _has_movement_intent(normalized: str) -> bool:
    """Detect general movement questions without requiring the literal word 'route'."""
    # An explicit movement-display request remains a route request when it also
    # asks for explanation. Auxiliary analytical clauses must not veto it.
    if _contains_display_verb(normalized) and (
        _contains_movement_object(normalized) or _contains_movement_verb(normalized)
    ):
        return True
    if _has_analytical_movement_question(normalized):
        return False
    if _has_movement_display_intent(normalized):
        return True
    padded = f" {normalized} "
    if any(verb in normalized for verb in _TRANSIT_VERBS) and (
        padded.startswith("how ") or " how " in padded or padded.startswith("where ") or " where " in padded
    ):
        return True
    if not _contains_movement_verb(normalized):
        return False
    if any(marker in padded for marker in _DIRECTIONAL_MARKERS):
        return True
    return " from " in padded and " to " in padded


def infer_requested_output(user_message: str) -> str:
    normalized = user_message.lower()
    if any(term in normalized for term in _ROUTE_TERMS):
        return "historical_route"
    # An explicit command + named movement + from-frame is a route request,
    # even when the command is normally analytical ("explain" / "describe").
    if re.search(
        r"(?i:\b(?:explain|describe)\s+)(?:[A-Z][A-Za-z'\u2019-]+\s+){1,4}"
        r"(?i:(?:movements?|marches?|journeys?|advances?|returns?|travels?|voyages?)\s+from\b)",
        user_message,
    ):
        return "historical_route"
    # Direct named-subject commands express route intent through their from/to
    # frame. This classifies requested output only; evidence still owns authority.
    if re.match(
        r"^\s*(?i:trace|reconstruct|follow)\s+(?:[A-Z][A-Za-z'\u2019-]+\s+){1,4}(?i:from\b)",
        user_message,
    ):
        from ..routes.query_scope_parser import parse_query_scope

        scope = parse_query_scope(user_message)
        if scope.origin and scope.destination:
            return "historical_route"
    if _has_movement_intent(normalized):
        return "historical_route"
    if any(term in normalized for term in _GEOGRAPHY_TERMS):
        return "geography_fact"
    return "answer"


def _explicitly_insufficient(answer: str | None) -> bool:
    normalized = (answer or "").lower()
    return any(term in normalized for term in _INSUFFICIENT_TERMS)


def _model_result(tool_name: str, payload: dict, state: AgentState, remaining_search_budget: int | None = None) -> dict:
    if tool_name == "search_historical_evidence":
        result = payload.get("result", {})
        if result.get("status") == "search_budget_exhausted":
            return {**result, "accumulated_evidence_count": len(state.historical_evidence), "remaining_search_budget": remaining_search_budget}
        visible_evidence = state.historical_evidence[:8]
        events = visible_historical_events(state)
        readiness = route_is_ready(state)
        return {
            "result_count": result.get("result_count", 0),
            "unique_authors": sorted({item.author for item in state.historical_evidence}),
            "unique_works": sorted({item.work for item in state.historical_evidence}),
            "books_covered": sorted({item.book for item in state.historical_evidence if item.book}),
            "locators_covered": sorted({item.locator for item in state.historical_evidence if item.locator})[:8],
            "accumulated_evidence_count": len(state.historical_evidence),
            "remaining_search_budget": remaining_search_budget,
            "evidence_support_status": state.evidence_support_status,
            "relevant_evidence_count": state.relevant_evidence_count,
            "matched_subject_terms": state.matched_subject_terms,
            "missing_subject_terms": state.missing_subject_terms,
            "evidence": [{"id": item.id, "author": item.author, "work": item.work, "locator": item.locator, "excerpt": item.excerpt[:360]} for item in visible_evidence],
            "historical_events": events,
            "incomplete_movement_facts": incomplete_movement_answer_context(state),
            "route_ready": readiness,
            "movement_event_count": sum(1 for event in events if str(event.get("event_type", "")).upper() == "MOVEMENT"),
        }
    if tool_name == "build_historical_route":
        route = state.historical_route
        return {"route": None} if route is None else {
            "route_points": [{"name": point.historical_place.canonical_name, "evidence_refs": point.evidence_refs, "coordinate_role": point.coordinate_role} for point in route.ordered_points],
            "unresolved_mentions": [mention.normalized_name for mention in route.unresolved_mentions],
        }
    return payload.get("result", {})


def _route_is_built(state: AgentState) -> bool:
    route = state.historical_route
    if route is None:
        return False
    return bool(route.ordered_points or route.route_components or route.branch_relations)


def _should_suppress_post_route_tool(tool_name: str, state: AgentState) -> bool:
    if not _route_is_built(state):
        return False
    if tool_name == "build_historical_route":
        return True
    return tool_name in POST_ROUTE_UPSTREAM_TOOLS


class BoundedAgentLoop:
    def __init__(self, provider, tools, max_steps: int = 8, max_tool_executions: int = 10, max_rag_search_executions: int = 4, max_completion_corrections: int = 1, max_completion_tool_executions: int = 1, max_grounding_corrections: int = 1):
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps
        self.max_tool_executions = max_tool_executions
        self.max_rag_search_executions = max_rag_search_executions
        self.max_completion_corrections = max_completion_corrections
        self.max_completion_tool_executions = max_completion_tool_executions
        self.max_grounding_corrections = max_grounding_corrections

    @staticmethod
    def _is_completion_critical_tool(tool_name: str, state: AgentState) -> bool:
        return state.historical_route is None and COMPLETION_TOOL_BY_OUTPUT.get(state.requested_output) == tool_name

    @staticmethod
    def _route_builder_attempted(state: AgentState) -> bool:
        return bool(state.tool_execution_stats.get("route_builder_attempted"))

    def _should_attempt_deterministic_route_builder(self, state: AgentState, *, require_route_ready: bool = False) -> bool:
        ready = (
            state.requested_output == "historical_route"
            and bool(state.historical_evidence)
            and not self._route_builder_attempted(state)
        )
        if not ready:
            return False
        if require_route_ready:
            return route_is_ready(state)
        return True

    @staticmethod
    def _default_route_builder_arguments(state: AgentState) -> dict:
        event = state.historical_events[0] if state.historical_events else None
        period = (event.period if event and event.period else None) or state.historical_period or "unspecified"
        return {
            "event_id": event.id if event else "evidence-driven-route",
            "name": event.name if event else "Evidence-supported historical route",
            "period": period,
        }

    def _attempt_deterministic_route_builder(self, state: AgentState) -> None:
        """One bounded route-builder attempt. Does not invent a route if construction fails closed."""
        arguments = self._default_route_builder_arguments(state)
        completion_critical = self._is_completion_critical_tool("build_historical_route", state)
        general_exhausted = state.tool_execution_stats["general_tool_executions"] >= self.max_tool_executions
        budget_source = "general"
        if general_exhausted and completion_critical and state.tool_execution_stats["completion_reserved_executions"] < self.max_completion_tool_executions:
            budget_source = "completion_reserved"
        elif general_exhausted:
            state.tool_execution_stats["budget_rejected"] += 1
            if completion_critical:
                state.tool_execution_stats["completion_budget_rejected"] += 1
            else:
                state.tool_execution_stats["general_budget_rejected"] += 1
            state.tool_history.append(AgentToolHistoryEntry(
                tool_name="build_historical_route", arguments=arguments, success=False,
                result_summary="deterministic route builder skipped; tool execution budget reached",
                duration_ms=0, outcome="completion_budget_rejected" if completion_critical else "budget_rejected",
                budget_source="completion_reserved" if completion_critical else "general",
            ))
            return
        logger.info("deterministic_route_builder_started budget_source=%s", budget_source)
        payload, summary = self.tools.execute("build_historical_route", arguments, state)
        logger.info("deterministic_route_builder_completed success=%s duration_ms=%s", payload["success"], payload["duration_ms"])
        state.tool_execution_stats["actual_tool_executions"] += 1
        state.tool_execution_stats["tool_requests"] += 1
        if budget_source == "completion_reserved":
            state.tool_execution_stats["completion_reserved_executions"] += 1
        else:
            state.tool_execution_stats["general_tool_executions"] += 1
        if not payload["success"]:
            state.tool_execution_stats["tool_failures"] += 1
        state.tool_history.append(AgentToolHistoryEntry(
            tool_name="build_historical_route", arguments=arguments, success=payload["success"],
            result_summary=summary, duration_ms=payload["duration_ms"],
            outcome="success" if payload["success"] else "failure", budget_source=budget_source,
        ))
        state.tool_execution_stats["route_builder_attempted"] = 1

    def _refresh_evidence_support(self, state: AgentState):
        assessment = assess_evidence_support(state.user_query or "", state.requested_output, state.historical_evidence)
        state.evidence_support_status = assessment.status
        state.relevant_evidence_count = assessment.relevant_count
        state.total_evidence_count = assessment.total_count
        state.matched_subject_terms = list(assessment.matched_subject_terms)
        state.missing_subject_terms = list(assessment.missing_subject_terms)
        return assessment

    @staticmethod
    def _has_audited_geography(state: AgentState) -> bool:
        return any(entry.success and entry.tool_name in {
            "resolve_ancient_place", "calculate_distance", "get_elevation", "get_elevation_profile",
        } for entry in state.tool_history)

    @staticmethod
    def _record_grounding_assessment(state: AgentState, assessment) -> None:
        entity_assessments = assessment.candidate_entities
        state.detected_phrase_count = len(entity_assessments)
        state.detected_entity_count = sum(item.entity_like for item in entity_assessments)
        state.evidence_grounded_entity_count = sum(
            item.entity_like and item.source == "evidence" for item in entity_assessments
        )
        state.query_context_entity_count = sum(
            item.entity_like and item.source == "query" for item in entity_assessments
        )
        state.detected_work_titles = list(assessment.detected_work_titles)
        state.evidence_grounded_claim_count = len(assessment.provenance.evidence_grounded_claims)
        state.unverified_suggestion_count = len(assessment.provenance.unverified_suggestions)
        state.unverified_suggestion_terms = [
            item.text for item in assessment.provenance.unverified_suggestions
        ]
        state.unsupported_fact_terms = list(assessment.unsupported_fact_terms)
        state.unsupported_fact_claim_count = len(assessment.unsupported_fact_terms)
        state.ignored_non_entity_terms = list(assessment.ignored_non_entity_terms)
        if assessment.candidate_explosion:
            state.warnings.append("provenance_candidate_explosion")

    def _route_preserving_guardrail_reply(self, state: AgentState) -> str:
        summary = admitted_route_summary(state)
        if summary:
            return summary
        if state.requested_output == "historical_route" and state.historical_route_presentation is not None:
            return ROUTE_PROSE_GROUNDING_FALLBACK
        if state.requested_output == "historical_route" and state.historical_route is not None:
            return ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION
        return GENERIC_GROUNDING_GUARDRAIL

    def _should_apply_route_terminal_closure(self, state: AgentState) -> bool:
        return (
            state.requested_output == "historical_route"
            and (state.historical_route is not None or state.historical_route_presentation is not None)
        )

    def _finish_route_preserving_guardrail(
        self,
        state: AgentState,
        started: float,
        *,
        reason_warning: str,
    ) -> tuple[str, AgentState]:
        summary = admitted_route_summary(state)
        if summary:
            state.status = "completed"
            state.final_grounding_status = "route_state_summary"
            return self._finish(summary, state, started)
        state.status = "completed_with_guardrail"
        state.final_grounding_status = "guardrail_fallback"
        state.warnings.append(reason_warning)
        if state.historical_route is not None:
            state.warnings.append("prose_grounding_discarded_route_preserved")
        return self._finish(self._route_preserving_guardrail_reply(state), state, started)

    def _finish_route_terminal_closure(self, state: AgentState, started: float) -> tuple[str, AgentState]:
        return self._finish_route_preserving_guardrail(
            state,
            started,
            reason_warning="route_terminal_submission_missing",
        )

    def _grounding_guardrail_reply(self, state: AgentState) -> str:
        return self._route_preserving_guardrail_reply(state)

    def _finish_grounding_guardrail(self, state: AgentState, started: float) -> tuple[str, AgentState]:
        return self._finish_route_preserving_guardrail(
            state,
            started,
            reason_warning="unsupported_historical_answer_discarded",
        )

    def _route_completion_action(self, response_content: str | None, state: AgentState, corrections: int) -> str:
        if state.requested_output != "historical_route" or state.historical_route is not None:
            return "finish"
        if self._should_attempt_deterministic_route_builder(state):
            self._attempt_deterministic_route_builder(state)
            if state.historical_route is not None:
                return "finish"
        if not self._route_builder_attempted(state) and not state.historical_evidence:
            return "finish_insufficient"
        if self._route_builder_attempted(state) and state.historical_route is None:
            if state.evidence_support_status in {"irrelevant", "insufficient"}:
                return "finish_insufficient"
            return "finish"
        if state.evidence_support_status in {"irrelevant", "insufficient"}:
            return "finish_insufficient"
        if corrections < self.max_completion_corrections:
            return "correct"
        return "failed_contract"

    @staticmethod
    def _record_provider_timing(
        state: AgentState, *, request_started: float, call_started: float,
        call_finished: float, agent_step: int, status: str,
    ) -> None:
        """Keep request-local timing only; prompts, responses, and secrets stay out."""
        timing = AgentProviderCallTiming(
            call_number=len(state.provider_call_timing) + 1,
            agent_step=agent_step,
            started_ms=int((call_started - request_started) * 1000),
            finished_ms=int((call_finished - request_started) * 1000),
            elapsed_ms=int((call_finished - call_started) * 1000),
            status=status,
        )
        state.provider_call_timing.append(timing)
        timings = state.provider_call_timing
        state.tool_results["provider_timing_summary"] = {
            "total_calls": len(timings),
            "successful_calls": sum(item.status == "SUCCESS" for item in timings),
            "timed_out_calls": sum(item.status == "TIMEOUT" for item in timings),
            "failed_calls": sum(item.status == "ERROR" for item in timings),
            "total_wait_ms": sum(item.elapsed_ms for item in timings),
            "longest_call_ms": max((item.elapsed_ms for item in timings), default=0),
        }

    def run(self, user_message: str, state: AgentState) -> tuple[str, AgentState]:
        started = perf_counter()
        logger.info("agent_request_started")
        state.user_query, state.status, state.final_answer = user_message, "running", None
        state.grounding_corrections, state.detected_phrase_count, state.detected_entity_count, state.evidence_grounded_entity_count, state.query_context_entity_count, state.detected_work_titles, state.evidence_grounded_claim_count, state.unverified_suggestion_count, state.unverified_suggestion_terms, state.unsupported_fact_claim_count, state.unsupported_fact_terms, state.ignored_non_entity_terms, state.final_grounding_status = 0, 0, 0, 0, 0, [], 0, 0, [], 0, [], [], None
        state.requested_output = infer_requested_output(user_message)
        state.route_intent = (
            self.tools.resolve_route_intent(user_message)
            if state.requested_output == "historical_route" else None
        )
        state.historical_route = None
        state.supporting_evidence = []
        state.historical_route_presentation = None
        state.historical_route_diagnostics = None
        state.historical_events = []
        state.provider_call_timing = []
        state.historical_event_diagnostics = None
        state.intent = state.route_intent.intent if state.route_intent else state.requested_output
        state.messages.append({"role": "user", "content": user_message})
        state.tool_execution_stats = {
            "llm_api_calls": 0, "tool_requests": 0, "actual_tool_executions": 0,
            "general_tool_executions": 0, "completion_reserved_executions": 0,
            "duplicate_tool_calls": 0, "tool_failures": 0, "budget_rejected": 0,
            "general_budget_rejected": 0, "completion_budget_rejected": 0,
            "rag_search_executions": 0, "rag_search_budget_rejected": 0,
            "completion_corrections": 0,
            "grounding_corrections": 0,
            "suppressed_tool_calls": 0,
            "route_builder_attempted": 0,
        }
        self._refresh_evidence_support(state)
        if (
            state.requested_output == "historical_route"
            and not state.historical_evidence
            and _has_movement_display_intent((user_message or "").lower())
        ):
            bootstrap_started = perf_counter()
            bootstrap_payload, _bootstrap_summary = self.tools.execute(
                "search_historical_evidence",
                {"query": user_message, "top_k": 10},
                state,
            )
            self._refresh_evidence_support(state)
            logger.info(
                "route_discovery_bootstrap_search success=%s elapsed_ms=%s",
                bootstrap_payload["success"],
                int((perf_counter() - bootstrap_started) * 1000),
            )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state.messages[-12:]]
        failures: dict[str, int] = {}
        attempted: dict[str, AgentToolHistoryEntry] = {}
        corrections = 0
        grounding_corrections = 0
        for step in range(1, self.max_steps + 1):
            state.step_count = step
            logger.info("llm_%s_started step=%s", "continuation" if step > 1 else "request", step)
            provider_started = perf_counter()
            try:
                schemas = self.tools.schemas
                if route_is_ready(state) and not _route_is_built(state):
                    schemas = route_ready_tool_schemas(self.tools.schemas)
                elif state.requested_output == "answer" and state.historical_evidence:
                    schemas = self.tools.schemas
                else:
                    schemas = [tool for tool in self.tools.schemas if tool["name"] != "submit_grounded_answer"]
                # Derived afresh for every call, including bootstrap retrieval.
                # Keep this request-local context out of persisted conversation history.
                incomplete_facts = incomplete_movement_answer_context(state)
                answer_messages = messages
                if incomplete_facts:
                    answer_messages = [*messages, {"role": "system", "content":
                        "The following incomplete_movement_facts are source-linked negative/non-completion "
                        "documentary context for explanation only. They are not completed movements, arrivals, "
                        "route nodes, edges, travel modes, or route completeness. Do not infer missing endpoints "
                        "or actors. Cite their Evidence IDs when using them. "
                        + json.dumps({"incomplete_movement_facts": incomplete_facts}, ensure_ascii=False)}]
                response = self.provider.complete(answer_messages, schemas)
            except Exception as exc:
                provider_finished = perf_counter()
                self._record_provider_timing(
                    state, request_started=started, call_started=provider_started,
                    call_finished=provider_finished, agent_step=step,
                    status="TIMEOUT" if any(term in str(exc).lower() for term in ("timeout", "timed out")) else "ERROR",
                )
                state.status = "provider_error"
                state.warnings.append(f"LLM provider failure: {type(exc).__name__}")
                return self._finish("The configured language-model provider is unavailable.", state, started)
            provider_finished = perf_counter()
            self._record_provider_timing(
                state, request_started=started, call_started=provider_started,
                call_finished=provider_finished, agent_step=step, status="SUCCESS",
            )
            state.tool_execution_stats["llm_api_calls"] += 1
            state.tool_results["llm_api_call_count"] = state.tool_execution_stats["llm_api_calls"]
            if response.http_status is not None:
                state.tool_results.setdefault("llm_http_statuses", []).append(response.http_status)
            if response.usage:
                totals = state.tool_results.setdefault("llm_usage", {})
                for key, value in response.usage.items():
                    totals[key] = totals.get(key, 0) + value
            logger.info("llm_response_received step=%s finish_reason=%s tool_call_count=%s", step, response.finish_reason, len(response.tool_calls))
            terminal_calls = [call for call in response.tool_calls if call.name == "submit_grounded_answer"]
            if terminal_calls and self._should_attempt_deterministic_route_builder(state):
                self._attempt_deterministic_route_builder(state)
            if terminal_calls:
                call = terminal_calls[0]
                arguments = call.arguments
                answer = arguments.get("answer")
                ids = arguments.get("evidence_ids")
                insufficient = arguments.get("insufficient_evidence", False)
                issues = []
                if len(response.tool_calls) != 1 or not isinstance(answer, str) or not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
                    issues.append("malformed_grounded_answer_submission")
                selected_ids = tuple(ids) if isinstance(ids, list) and all(isinstance(item, str) for item in ids) else ()
                assessment = assess_final_answer_provenance(_historical_terminal_text(answer if isinstance(answer, str) else None, state), state.user_query or "", state.historical_evidence)
                explicit_insufficient = bool(insufficient) and _explicitly_insufficient(answer if isinstance(answer, str) else None)
                if bool(insufficient) and not explicit_insufficient and assessment.status in {"grounded", "grounded_with_unverified_suggestions"}:
                    insufficient = False
                    explicit_insufficient = False
                issues.extend(validate_evidence_selection(selected_ids, state.historical_evidence, require_selection=not explicit_insufficient))
                event_supported = event_relation_supports_answer(answer, state.historical_events, state.historical_evidence)
                if assessment.status == "unsupported_fact" and not event_supported: issues.append("unsupported_fact")
                if bool(insufficient) and not explicit_insufficient: issues.append("invalid_insufficient_evidence_submission")
                if issues:
                    state.warnings.extend(issues)
                    if grounding_corrections < self.max_grounding_corrections and step < self.max_steps:
                        grounding_corrections += 1; state.grounding_corrections += 1; state.tool_execution_stats["grounding_corrections"] += 1
                        manifest = "; ".join(f"{item.id} ({item.author}, {item.work}, {item.locator})" for item in state.historical_evidence[:8])
                        messages.append({"role":"user","content":f"Your submit_grounded_answer validation failed: {', '.join(issues)}. Call submit_grounded_answer again with answer and evidence_ids selected only from: {manifest}. Use insufficient_evidence=true only for an explicit insufficiency answer."})
                        continue
                    return self._finish_grounding_guardrail(state, started)
                self._record_grounding_assessment(state, assessment)
                if explicit_insufficient:
                    state.final_grounding_status="insufficient_evidence"
                    return self._finish(answer, state, started)
                rendered = render_evidence_citations(selected_ids, state.historical_evidence)
                state.final_grounding_status="provenance_corrected" if grounding_corrections else "grounded"
                return self._finish(f"{answer.strip()} {rendered}".strip(), state, started)
            if not response.tool_calls:
                self._refresh_evidence_support(state)
                if state.requested_output in {"answer", "geography_fact"}:
                    has_geography = self._has_audited_geography(state)
                    if not state.historical_evidence and not has_geography:
                        if grounding_corrections < self.max_grounding_corrections and step < self.max_steps:
                            grounding_corrections += 1
                            state.grounding_corrections += 1
                            state.tool_execution_stats["grounding_corrections"] += 1
                            messages.append({
                                "role": "user",
                                "content": (
                                    "No audited factual source has been supplied. Before answering, call "
                                    "search_historical_evidence or an appropriate audited geography tool. "
                                    "Do not answer from model knowledge."
                                ),
                            })
                            logger.info("answer_evidence_correction_started count=%s", grounding_corrections)
                            continue
                        state.final_grounding_status = "insufficient_evidence"
                        state.warnings.append("answer_blocked_without_evidence")
                        return self._finish(
                            GENERIC_GROUNDING_GUARDRAIL,
                            state,
                            started,
                        )
                    try:
                        answer_assessment = assess_final_answer_provenance(
                            response.content, state.user_query or "", state.historical_evidence
                        )
                    except Exception as exc:
                        state.status = "failed_grounding"
                        state.final_grounding_status = "validator_error"
                        state.warnings.append(f"grounding_validator_error:{type(exc).__name__}")
                        return self._finish(
                            "The system could not safely validate the historical answer.",
                            state,
                            started,
                        )
                    self._record_grounding_assessment(state, answer_assessment)
                    answer_text = response.content or ""
                    provenance_acceptable = answer_assessment.status in {
                        "grounded",
                        "grounded_with_unverified_suggestions",
                    }
                    citation_issues = validate_evidence_citations(
                        answer_text, state.historical_evidence, require_citation=False,
                    )
                    if (
                        not citation_issues
                        and state.requested_output == "answer"
                        and not _explicitly_insufficient(answer_text)
                        and not provenance_acceptable
                    ):
                        citation_issues = ("missing_grounded_answer_submission",)
                    if citation_issues:
                        state.warnings.extend(citation_issues)
                    if (
                        (answer_assessment.status == "unsupported_fact" or citation_issues)
                        and grounding_corrections < self.max_grounding_corrections
                        and step < self.max_steps
                    ):
                        grounding_corrections += 1
                        state.grounding_corrections += 1
                        state.tool_execution_stats["grounding_corrections"] += 1
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Revise the answer; validation failed: {', '.join(citation_issues) or answer_assessment.status}. "
                                "Call submit_grounded_answer with answer and evidence_ids selected from this manifest; do not write citation metadata. "
                                "Allowed Evidence: " + "; ".join(f"{item.id} ({item.author}, {item.work}, {item.locator})" for item in state.historical_evidence[:8]) + ". Remove "
                                "unsupported dates, people, places, events, and routes. If the Evidence does not "
                                "answer the question, say that it is insufficient."
                            ),
                        })
                        logger.info(
                            "answer_grounding_correction_started count=%s terms=%s",
                            grounding_corrections,
                            ",".join(answer_assessment.unsupported_fact_terms),
                        )
                        continue
                    if answer_assessment.status == "unsupported_fact" or citation_issues:
                        state.status = "completed_with_guardrail"
                        state.final_grounding_status = "guardrail_fallback"
                        state.warnings.append("unsupported_historical_answer_discarded")
                        return self._finish(
                            GENERIC_GROUNDING_GUARDRAIL,
                            state,
                            started,
                        )
                    state.final_grounding_status = (
                        "provenance_corrected" if grounding_corrections else answer_assessment.status
                    )
                completion_action = self._route_completion_action(response.content, state, corrections)
                if completion_action == "finish":
                    if self._should_apply_route_terminal_closure(state):
                        return self._finish_route_terminal_closure(state, started)
                    return self._finish(response.content or "The agent completed without a final answer.", state, started)
                if completion_action == "finish_insufficient":
                    try:
                        claim_assessment = assess_final_answer_provenance(_historical_terminal_text(response.content, state), state.user_query or "", state.historical_evidence)
                    except Exception as exc:
                        state.status = "failed_grounding"
                        state.final_grounding_status = "validator_error"
                        state.warnings.append(f"grounding_validator_error:{type(exc).__name__}")
                        return self._finish("The system could not safely validate the requested HistoricalRoute response.", state, started)
                    citation_issues = validate_evidence_citations(
                        response.content or "", state.historical_evidence, require_citation=False,
                    )
                    if citation_issues:
                        state.warnings.extend(citation_issues)
                        return self._finish_grounding_guardrail(state, started)
                    entity_assessments = claim_assessment.candidate_entities
                    state.detected_phrase_count = len(entity_assessments)
                    state.detected_entity_count = sum(item.entity_like for item in entity_assessments)
                    state.evidence_grounded_entity_count = sum(item.entity_like and item.source == "evidence" for item in entity_assessments)
                    state.query_context_entity_count = sum(item.entity_like and item.source == "query" for item in entity_assessments)
                    state.detected_work_titles = list(claim_assessment.detected_work_titles)
                    state.evidence_grounded_claim_count = len(claim_assessment.provenance.evidence_grounded_claims)
                    state.unverified_suggestion_count = len(claim_assessment.provenance.unverified_suggestions)
                    state.unverified_suggestion_terms = [item.text for item in claim_assessment.provenance.unverified_suggestions]
                    state.unsupported_fact_terms = list(claim_assessment.unsupported_fact_terms)
                    state.unsupported_fact_claim_count = len(claim_assessment.unsupported_fact_terms)
                    state.ignored_non_entity_terms = list(claim_assessment.ignored_non_entity_terms)
                    if claim_assessment.candidate_explosion:
                        state.warnings.append("provenance_candidate_explosion")
                    if claim_assessment.status == "unsupported_fact" and grounding_corrections < self.max_grounding_corrections and step < self.max_steps:
                        grounding_corrections += 1
                        state.grounding_corrections += 1
                        state.tool_execution_stats["grounding_corrections"] += 1
                        correction = "Your previous answer introduced historical entities not supported by the current Evidence as established facts or route content. You may keep such entities only as clearly labeled unverified research suggestions. Do not present them as route nodes, verified events, campaign sequence, or current-Evidence conclusions. Do not call tools for unverified suggestions."
                        messages.append({"role": "user", "content": correction})
                        logger.info("grounding_correction_started count=%s terms=%s", grounding_corrections, ",".join(claim_assessment.unsupported_fact_terms))
                        continue
                    if claim_assessment.status == "unsupported_fact":
                        # The model answer is discarded. The runtime completed safely because
                        # no unsupported content, route, or GIS action reaches the user.
                        state.status = "completed_with_guardrail"
                        state.final_grounding_status = "guardrail_fallback"
                        state.warnings.append("unsupported_claims_with_insufficient_evidence")
                        return self._finish("The current retrieved historical evidence is insufficient to support a reliable HistoricalRoute, so the system will not add unsupported places or route details.", state, started)
                    state.final_grounding_status = "provenance_corrected" if grounding_corrections else claim_assessment.status
                    state.warnings.append("insufficient_relevant_evidence")
                    # Grounded evidence can establish non-completion or disagreement
                    # without using an insufficiency keyword. The final route guard
                    # still rejects any positive claim without route state.
                    preserve_answer = _explicitly_insufficient(response.content) or (
                        bool(state.historical_evidence)
                        and claim_assessment.status in {"grounded", "grounded_with_unverified_suggestions"}
                    )
                    answer = response.content if preserve_answer else "The current retrieved historical evidence is insufficient to support a reliable HistoricalRoute."
                    return self._finish(answer, state, started)
                if completion_action == "correct" and step < self.max_steps:
                    corrections += 1
                    state.tool_execution_stats["completion_corrections"] += 1
                    correction = "The user requested a Historical Route. No structured route has been produced by build_historical_route. Call build_historical_route using accumulated Evidence, or explicitly state that available Evidence is insufficient to generate a route. Do not substitute repeated place resolution or construct coordinates/GeoJSON yourself."
                    messages.append({"role": "user", "content": correction})
                    logger.info("route_completion_correction_started count=%s", corrections)
                    continue
                state.status = "failed_contract"
                state.warnings.append("historical_route_required_but_not_built")
                return self._finish("The requested HistoricalRoute was not built from the available Evidence.", state, started)
            messages.append({"role": "assistant", "content": response.content or "", "tool_calls": [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}} for call in response.tool_calls]})
            for call in response.tool_calls:
                state.tool_execution_stats["tool_requests"] += 1
                fingerprint = _fingerprint(call.name, call.arguments)
                attempt_key = prior_tool_attempt_key(call.name, call.arguments)
                budget_source = "general"
                handled = False
                if _should_suppress_post_route_tool(call.name, state):
                    summary = "route already built; upstream discovery suppressed"
                    payload = {
                        "success": False,
                        "result": {
                            "status": "route_already_built",
                            "reason": "ROUTE_ALREADY_BUILT",
                            "message": POST_ROUTE_SUPPRESSION_MESSAGE,
                            "tool_call_suppressed": True,
                            "tool": call.name,
                            "phase": "POST_ROUTE",
                        },
                        "summary": summary,
                        "duration_ms": 0,
                    }
                    outcome = "route_already_built"
                    state.tool_execution_stats["suppressed_tool_calls"] += 1
                    state.tool_results.setdefault("tool_call_suppressed", []).append({
                        "tool": call.name,
                        "step": step,
                        "phase": "POST_ROUTE",
                        "reason": "ROUTE_ALREADY_BUILT",
                    })
                    handled = True
                elif should_suppress_pre_route_tool(call.name, state):
                    summary = "route-ready state established; upstream discovery suppressed"
                    payload = {
                        "success": False,
                        "result": {
                            "status": "route_ready",
                            "reason": "ROUTE_READY",
                            "message": PRE_ROUTE_SUPPRESSION_MESSAGE,
                            "tool_call_suppressed": True,
                            "tool": call.name,
                            "phase": "PRE_ROUTE",
                        },
                        "summary": summary,
                        "duration_ms": 0,
                    }
                    outcome = "route_ready_suppressed"
                    state.tool_execution_stats["suppressed_tool_calls"] += 1
                    state.tool_results.setdefault("tool_call_suppressed", []).append({
                        "tool": call.name,
                        "step": step,
                        "phase": "PRE_ROUTE",
                        "reason": "ROUTE_READY",
                    })
                    handled = True
                elif call.name == "build_historical_route" and self._route_builder_attempted(state):
                    prior = next(
                        entry for entry in reversed(state.tool_history)
                        if entry.tool_name == "build_historical_route"
                    )
                    if prior.success or state.historical_route is not None:
                        summary = "route builder already attempted in this run"
                        payload = {
                            "success": state.historical_route is not None,
                            "result": {
                                "status": "duplicate",
                                "message": "This exact tool call already succeeded earlier in this run.",
                                "previous_result_summary": prior.result_summary,
                            },
                            "summary": summary,
                            "duration_ms": 0,
                        }
                        outcome = "duplicate"
                        state.tool_execution_stats["duplicate_tool_calls"] += 1
                        handled = True
                elif attempt_key in attempted and attempted[attempt_key].success:
                    prior = attempted[attempt_key]
                    payload, summary = duplicate_attempt_payload(prior)
                    outcome = "duplicate"
                    state.tool_execution_stats["duplicate_tool_calls"] += 1
                    handled = True
                elif call.name == "resolve_ancient_place":
                    cached = lookup_prior_resolve(call.arguments, state)
                    if cached is not None:
                        result, summary = cached
                        payload = {"success": bool(result.get("found")), "result": result, "summary": summary, "duration_ms": 0}
                        outcome = "duplicate_resolve"
                        state.tool_execution_stats["duplicate_tool_calls"] += 1
                        handled = True
                    elif attempt_key in attempted:
                        prior = attempted[attempt_key]
                        payload, summary = duplicate_attempt_payload(prior)
                        outcome = "duplicate"
                        state.tool_execution_stats["duplicate_tool_calls"] += 1
                        handled = True
                if not handled:
                    if call.name == "search_historical_evidence" and state.tool_execution_stats["rag_search_executions"] >= self.max_rag_search_executions:
                        summary = "historical evidence search budget reached; use existing evidence or state insufficiency"
                        payload = {"success": False, "result": {"status": "search_budget_exhausted", "message": "Historical evidence search budget reached. Use the evidence already retrieved to answer or state that evidence is insufficient."}, "summary": summary, "duration_ms": 0}
                        outcome = "search_budget_rejected"
                        state.tool_execution_stats["rag_search_budget_rejected"] += 1
                    elif failures.get(fingerprint, 0) >= 2:
                        state.status = "tool_failure"
                        state.warnings.append(f"Repeated failing tool call blocked: {call.name}")
                        return self._finish("The agent stopped after repeated tool failures.", state, started)
                    else:
                        completion_critical = self._is_completion_critical_tool(call.name, state)
                        general_exhausted = state.tool_execution_stats["general_tool_executions"] >= self.max_tool_executions
                        if general_exhausted and completion_critical and state.tool_execution_stats["completion_reserved_executions"] < self.max_completion_tool_executions:
                            budget_source = "completion_reserved"
                        elif general_exhausted and completion_critical:
                            summary = "completion tool reservation already used; no further reserved execution is available"
                            payload = {"success": False, "result": {"status": "completion_budget_rejected", "message": "Completion tool reservation has been exhausted."}, "summary": summary, "duration_ms": 0}
                            outcome = "completion_budget_rejected"
                            budget_source = "completion_reserved"
                            state.tool_execution_stats["budget_rejected"] += 1
                            state.tool_execution_stats["completion_budget_rejected"] += 1
                        elif general_exhausted:
                            summary = "general tool execution budget reached; use existing results to finish"
                            payload = {"success": False, "result": {"status": "budget_rejected", "message": "General tool execution budget reached."}, "summary": summary, "duration_ms": 0}
                            outcome = "budget_rejected"
                            budget_source = "general"
                            state.tool_execution_stats["budget_rejected"] += 1
                            state.tool_execution_stats["general_budget_rejected"] += 1
                        else:
                            budget_source = "general"
                        if not general_exhausted or (completion_critical and state.tool_execution_stats["completion_reserved_executions"] < self.max_completion_tool_executions):
                            logger.info("tool_execution_started tool=%s budget_source=%s", call.name, budget_source)
                            payload, summary = self.tools.execute(call.name, call.arguments, state)
                            logger.info("tool_execution_completed tool=%s success=%s duration_ms=%s", call.name, payload["success"], payload["duration_ms"])
                            outcome = "success" if payload["success"] else "failure"
                            state.tool_execution_stats["actual_tool_executions"] += 1
                            if budget_source == "completion_reserved":
                                state.tool_execution_stats["completion_reserved_executions"] += 1
                            else:
                                state.tool_execution_stats["general_tool_executions"] += 1
                            if call.name == "search_historical_evidence":
                                state.tool_execution_stats["rag_search_executions"] += 1
                                self._refresh_evidence_support(state)
                            if call.name == "build_historical_route":
                                state.tool_execution_stats["route_builder_attempted"] = 1
                            if not payload["success"]:
                                failures[fingerprint] = failures.get(fingerprint, 0) + 1
                                state.tool_execution_stats["tool_failures"] += 1
                history_entry = AgentToolHistoryEntry(tool_name=call.name, arguments=call.arguments, success=payload["success"], result_summary=summary, duration_ms=payload["duration_ms"], outcome=outcome, budget_source=budget_source)
                attempted[attempt_key] = history_entry
                state.tool_history.append(history_entry)
                logger.info("agent_step=%s tool=%s outcome=%s", step, call.name, outcome)
                remaining_search_budget = max(0, self.max_rag_search_executions - state.tool_execution_stats["rag_search_executions"])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"tool": call.name, "outcome": outcome, "success": payload["success"], "summary": summary, "evidence_count": len(state.historical_evidence), "route_points": len(state.historical_route.ordered_points) if state.historical_route else 0, "result": _model_result(call.name, payload, state, remaining_search_budget)}, ensure_ascii=False)})
        if self._should_attempt_deterministic_route_builder(state):
            self._attempt_deterministic_route_builder(state)
        if self._should_apply_route_terminal_closure(state):
            return self._finish_route_terminal_closure(state, started)
        state.status = "max_steps"
        state.warnings.append("Maximum agent steps reached")
        return self._finish("The agent reached its safe step limit and returned a partial result.", state, started)

    def _finish(self, answer: str, state: AgentState, started: float) -> tuple[str, AgentState]:
        state.status = "completed" if state.status == "running" else state.status
        route_status = derive_route_result_status(state)
        has_simulation = _has_terminal_simulation(state)
        unsafe_route = self._final_answer_asserts_unsupported_route(answer, allow_simulation=has_simulation)
        if state.historical_route is None and unsafe_route:
            state.warnings.append("Final answer mentioned a route without route state")
            answer = _non_completion_reply(state) or self._no_route_terminal_guardrail_reply(state)
            state.final_grounding_status = "guardrail_fallback"
        elif route_status is RouteResultStatus.PARTIAL and unsafe_route:
            # Preserve admitted fragments without letting prose complete a missing leg.
            answer = admitted_route_summary(state) or self._no_route_terminal_guardrail_reply(state)
            state.warnings.append("Final route prose limited to admitted partial route")
        elif _TERMINAL_NON_COMPLETION.search(answer):
            if not _negative_prose_is_state_supported(answer, state):
                answer = (_non_completion_reply(state) or admitted_route_summary(state)
                          or self._no_route_terminal_guardrail_reply(state))
                state.warnings.append("Final non-completion prose replaced with documentary state")
        summary_refs = ()
        if state.historical_route is not None and answer == admitted_route_summary(state):
            summary_refs = tuple(ref for ref in state.historical_route.evidence_refs if ref)[:4]
        state.supporting_evidence = final_answer_support(
            answer, state.historical_evidence, validated_reference_ids=summary_refs,
        )
        state.final_answer = answer
        logger.info("agent_finished status=%s elapsed_ms=%s", state.status, int((perf_counter() - started) * 1000))
        state.messages.append({"role": "assistant", "content": answer})
        return answer, state

    @staticmethod
    def _final_answer_asserts_unsupported_route(answer: str, *, allow_simulation: bool = False) -> bool:
        if not (answer or "").strip():
            return False
        normalized = answer.casefold().strip()
        if normalized in {GENERIC_GROUNDING_GUARDRAIL.casefold(), NO_ROUTE_TERMINAL_GUARDRAIL.casefold()}:
            return False
        # Inspect asserted predicates, not mere route vocabulary. Infinitives
        # expressing a refused task are not assertions that a route occurred.
        clauses = _TERMINAL_CLAUSES.split(normalized)
        route_language = r"\b(?:route\w*|path|itinerary|movement|travel\w*|journey|expedition|waypoints?|geometry|origin|destination)\b"
        movement_predicate = r"\b(?:went|gone|moved|proceeded|travelled|traveled|sailed|marched|followed|passed|crossed|entered|arrived|reached)\b"
        construction_predicate = r"\b(?:draw|draws|drew|build|builds|built|reconstruct(?:ed|s|ing)?|generat\w*|establish\w*|prove\w*|confirm\w*|mention\w*)\b"
        nominal_predicate = rf"{route_language}[^.;!?]{{0,100}}?\b(?:is|are|was|were|would\s+be|could\s+be|may\s+be|might\s+be|includes?|shows?|follows?|consists?)\b"
        denial = (
            r"\b(?:no|never)\b(?:\W+\w+){0,5}\W*$"
            r"|\b(?:does|did|do)\s+not\s+(?:establish|support|show|prove|confirm|describe|document|mention)\b[^.!?;]*$"
            r"|\b(?:cannot|can't|could not|unable to)\s+(?:establish|support|show|prove|confirm|describe|document|reconstruct|build|generate)\b[^.!?;]*$"
            r"|\b(?:not|never)\s+(?:\w+ly\s+){0,2}$"
        )
        non_completion = (
            r"^.{0,80}?\b(?P<negative>"
            r"(?:cannot|can't|could not|should not|must not)\s+(?:be\s+)?(?:reconstruct\w*|draw\w*|build|built|generat\w*|establish\w*|support\w*|confirm\w*|prove\w*)"
            r"|(?:was|were|is|are)\s+not\s+(?:completed|executed|established|supported|documented|reconstructed|drawn|built|generated|available|proven|confirmed|shown|attested)"
            r"|(?:was|were|is|remains?)\s+(?:prevented|blocked|aborted|abandoned|incomplete|unexecuted|proposed|planned))\b"
        )
        for clause in clauses:
            if allow_simulation and _is_simulation_clause(clause):
                continue
            if _TERMINAL_NON_COMPLETION.search(clause):
                # Arrival inside "abandoned before ... reached" is not completed.
                clause = re.sub(r"\bbefore\b.*$", "", clause)
            assertions = list(re.finditer(movement_predicate, clause))
            if re.search(route_language, clause):
                assertions += list(re.finditer(construction_predicate, clause))
                assertions += list(re.finditer(nominal_predicate, clause))
            if not assertions:
                assertions += list(re.finditer(rf"{route_language}\s*(?::|\b(?:from|via|through)\b)", clause))
            if re.search(route_language, normalized) or re.search(r"路线|行军|旅程", normalized):
                assertions += list(re.finditer(
                    r"\b(?:likely|probably|apparently|plausibly)\b[^.!?;]{0,60}\b(?:via|through|from|along)\b", clause,
                ))
                assertions += list(re.finditer(r"(?:可能|大概|也许).{0,40}(?:经|从|到|沿|途)", clause))
            assertions += list(re.finditer(r"→|->|—>", clause))
            if (assertions or re.search(route_language, clause)) and re.search(
                r"\bnot\s+(?:impossible|unlikely)|\b(?:does|do|did)\s+not\s+(?:disprove|deny)"
                r"|\b(?:cannot|can't)\s+(?:exclude|rule out|deny|say)", clause,
            ):
                return True
            for assertion in assertions:
                prefix = clause[:assertion.start()]
                if re.search(r"\bto\s*$|\bwhether\b", prefix):
                    continue
                if re.search(denial, prefix) or re.search(
                    r"\b(?:cannot|can't|could not|unable to|does not|do not|did not)\s*$", prefix,
                ):
                    continue
                # Nominal predicates include their route subject; inspect the
                # subject's remaining clause for non-completion scope.
                nominal = re.match(route_language, assertion.group())
                mentions = list(re.finditer(route_language, prefix))
                tail = (
                    clause[assertion.start() + nominal.end():] if nominal else
                    clause[mentions[-1].end():] if mentions else clause[assertion.end():]
                )
                negative = re.search(non_completion, tail)
                if negative:
                    # A later denied action cannot cancel an earlier assertion.
                    # Locate its governing auxiliary rather than treating the
                    # entire remaining clause as negative.
                    before_denial = tail[:negative.start("negative")]
                    if not re.search(rf"{movement_predicate}|\b(?:is|are|was|were)\b", before_denial):
                        continue
                return True
            if re.search(r"路线|移动|行军|旅程|几何|进入|到达|从.+(?:经|到)", clause) and re.search(
                r"是|经过|途经|沿|包含|包括|从.+到|完成|建立|重建", clause,
            ):
                if not re.search(r"没有|无法|不能|不足|未|不应|不宜|不曾|被阻止|中止|放弃", clause):
                    return True
        return False

    @staticmethod
    def _no_route_terminal_guardrail_reply(state: AgentState) -> str:
        if state.requested_output == "historical_route":
            return NO_ROUTE_TERMINAL_GUARDRAIL
        return GENERIC_GROUNDING_GUARDRAIL
