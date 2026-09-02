from __future__ import annotations
import json, logging, re
from time import perf_counter
from backend.app.agent.prompts import SYSTEM_PROMPT
from backend.app.agent.evidence_support import assess_evidence_support, assess_final_answer_provenance, event_relation_supports_answer, render_evidence_citations, validate_evidence_citations, validate_evidence_selection
from backend.app.models import AgentProviderCallTiming, AgentState, AgentToolHistoryEntry

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
ROUTE_PROSE_GROUNDING_FALLBACK = (
    "A structured route was built from the current Evidence, but the generated explanation "
    "did not pass grounding validation. Use the verified route nodes and citations."
)
GENERIC_GROUNDING_GUARDRAIL = (
    "The current retrieved historical evidence is insufficient to support a reliable answer."
)


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
        visible_ids = {item.id for item in visible_evidence}
        events = []
        for event in state.historical_events[:8]:
            serialized = event.model_dump(mode="json")
            refs = set(serialized.get("evidence_refs", []))
            if not refs or not refs <= visible_ids:
                continue
            events.append({
                "id": serialized["id"],
                "name": serialized["name"],
                "event_type": serialized.get("event_type"),
                "summary": serialized.get("summary"),
                "period": serialized.get("period"),
                "temporal_grounding": serialized.get("temporal_grounding"),
                "place_mentions": serialized.get("place_mentions", []),
                "evidence_refs": serialized.get("evidence_refs", []),
                "grounding_status": serialized.get("grounding_status"),
                "limitations": serialized.get("limitations", []),
            })
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
        }
    if tool_name == "build_historical_route":
        route = state.historical_route
        return {"route": None} if route is None else {
            "route_points": [{"name": point.historical_place.canonical_name, "evidence_refs": point.evidence_refs, "coordinate_role": point.coordinate_role} for point in route.ordered_points],
            "unresolved_mentions": [mention.normalized_name for mention in route.unresolved_mentions],
        }
    return payload.get("result", {})


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
        return any(entry.tool_name == "build_historical_route" and entry.outcome in {"success", "failure"} for entry in state.tool_history)

    def _should_attempt_deterministic_route_builder(self, state: AgentState) -> bool:
        return (
            state.requested_output == "historical_route"
            and bool(state.historical_evidence)
            and not self._route_builder_attempted(state)
        )

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

    def _grounding_guardrail_reply(self, state: AgentState) -> str:
        if state.requested_output == "historical_route" and state.historical_route is not None:
            return ROUTE_PROSE_GROUNDING_FALLBACK
        return GENERIC_GROUNDING_GUARDRAIL

    def _finish_grounding_guardrail(self, state: AgentState, started: float) -> tuple[str, AgentState]:
        state.status = "completed_with_guardrail"
        state.final_grounding_status = "guardrail_fallback"
        state.warnings.append("unsupported_historical_answer_discarded")
        if state.requested_output == "historical_route" and state.historical_route is not None:
            state.warnings.append("prose_grounding_discarded_route_preserved")
        return self._finish(self._grounding_guardrail_reply(state), state, started)

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
        }
        self._refresh_evidence_support(state)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state.messages[-12:]]
        failures: dict[str, int] = {}
        successful: dict[str, str] = {}
        corrections = 0
        grounding_corrections = 0
        for step in range(1, self.max_steps + 1):
            state.step_count = step
            logger.info("llm_%s_started step=%s", "continuation" if step > 1 else "request", step)
            provider_started = perf_counter()
            try:
                schemas = self.tools.schemas if state.requested_output == "answer" and state.historical_evidence else [tool for tool in self.tools.schemas if tool["name"] != "submit_grounded_answer"]
                response = self.provider.complete(messages, schemas)
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
                assessment = assess_final_answer_provenance(answer if isinstance(answer, str) else None, state.user_query or "", state.historical_evidence)
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
                    return self._finish(response.content or "The agent completed without a final answer.", state, started)
                if completion_action == "finish_insufficient":
                    try:
                        claim_assessment = assess_final_answer_provenance(response.content, state.user_query or "", state.historical_evidence)
                    except Exception as exc:
                        state.status = "failed_grounding"
                        state.final_grounding_status = "validator_error"
                        state.warnings.append(f"grounding_validator_error:{type(exc).__name__}")
                        return self._finish("The system could not safely validate the requested HistoricalRoute response.", state, started)
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
                    answer = response.content if _explicitly_insufficient(response.content) else "The current retrieved historical evidence is insufficient to support a reliable HistoricalRoute."
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
                budget_source = "general"
                if fingerprint in successful:
                    summary = f"duplicate cache hit; reuse prior successful result: {successful[fingerprint]}"
                    payload = {"success": True, "result": {"status": "duplicate", "message": "This exact tool call already succeeded earlier in this run.", "previous_result_summary": successful[fingerprint]}, "summary": summary, "duration_ms": 0}
                    outcome = "duplicate"
                    state.tool_execution_stats["duplicate_tool_calls"] += 1
                elif call.name == "search_historical_evidence" and state.tool_execution_stats["rag_search_executions"] >= self.max_rag_search_executions:
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
                        if payload["success"]:
                            successful[fingerprint] = summary
                        else:
                            failures[fingerprint] = failures.get(fingerprint, 0) + 1
                            state.tool_execution_stats["tool_failures"] += 1
                state.tool_history.append(AgentToolHistoryEntry(tool_name=call.name, arguments=call.arguments, success=payload["success"], result_summary=summary, duration_ms=payload["duration_ms"], outcome=outcome, budget_source=budget_source))
                logger.info("agent_step=%s tool=%s outcome=%s", step, call.name, outcome)
                remaining_search_budget = max(0, self.max_rag_search_executions - state.tool_execution_stats["rag_search_executions"])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"tool": call.name, "outcome": outcome, "success": payload["success"], "summary": summary, "evidence_count": len(state.historical_evidence), "route_points": len(state.historical_route.ordered_points) if state.historical_route else 0, "result": _model_result(call.name, payload, state, remaining_search_budget)}, ensure_ascii=False)})
        if self._should_attempt_deterministic_route_builder(state):
            self._attempt_deterministic_route_builder(state)
        state.status = "max_steps"
        state.warnings.append("Maximum agent steps reached")
        return self._finish("The agent reached its safe step limit and returned a partial result.", state, started)

    def _finish(self, answer: str, state: AgentState, started: float) -> tuple[str, AgentState]:
        state.status = "completed" if state.status == "running" else state.status
        if state.historical_route is None and "route" in answer.lower() and "no route" not in answer.lower() and "insufficient" not in answer.lower():
            state.warnings.append("Final answer mentioned a route without route state")
        state.final_answer = answer
        logger.info("agent_finished status=%s elapsed_ms=%s", state.status, int((perf_counter() - started) * 1000))
        state.messages.append({"role": "assistant", "content": answer})
        return answer, state
