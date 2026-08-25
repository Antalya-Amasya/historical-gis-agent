from __future__ import annotations
import json, logging
from time import perf_counter
from backend.app.agent.prompts import SYSTEM_PROMPT
from backend.app.agent.evidence_support import assess_evidence_support, assess_final_answer_provenance
from backend.app.models import AgentState, AgentToolHistoryEntry

logger = logging.getLogger(__name__)
_ROUTE_TERMS = ("route", "路线", "行军", "进军", "绘制", "展示")
_GEOGRAPHY_TERMS = ("distance", "elevation", "coordinate", "距离", "高程", "坐标")
_INSUFFICIENT_TERMS = ("insufficient", "cannot build", "unable to build", "evidence is not enough", "证据不足", "无法生成", "不能生成")
COMPLETION_TOOL_BY_OUTPUT = {"historical_route": "build_historical_route"}


def _fingerprint(name: str, arguments: dict) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"


def infer_requested_output(user_message: str) -> str:
    normalized = user_message.lower()
    if any(term in normalized for term in _ROUTE_TERMS):
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
            "evidence": [{"id": item.id, "author": item.author, "work": item.work, "locator": item.locator, "excerpt": item.excerpt[:360]} for item in state.historical_evidence[:8]],
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

    def _refresh_evidence_support(self, state: AgentState):
        assessment = assess_evidence_support(state.user_query or "", state.requested_output, state.historical_evidence)
        state.evidence_support_status = assessment.status
        state.relevant_evidence_count = assessment.relevant_count
        state.total_evidence_count = assessment.total_count
        state.matched_subject_terms = list(assessment.matched_subject_terms)
        state.missing_subject_terms = list(assessment.missing_subject_terms)
        return assessment

    def _route_completion_action(self, response_content: str | None, state: AgentState, corrections: int) -> str:
        if state.requested_output != "historical_route" or state.historical_route is not None:
            return "finish"
        if state.evidence_support_status in {"irrelevant", "insufficient"}:
            return "finish_insufficient"
        if corrections < self.max_completion_corrections:
            return "correct"
        return "failed_contract"

    def run(self, user_message: str, state: AgentState) -> tuple[str, AgentState]:
        started = perf_counter()
        logger.info("agent_request_started")
        state.user_query, state.status, state.final_answer = user_message, "running", None
        state.grounding_corrections, state.detected_phrase_count, state.detected_entity_count, state.evidence_grounded_entity_count, state.query_context_entity_count, state.detected_work_titles, state.evidence_grounded_claim_count, state.unverified_suggestion_count, state.unverified_suggestion_terms, state.unsupported_fact_claim_count, state.unsupported_fact_terms, state.ignored_non_entity_terms, state.final_grounding_status = 0, 0, 0, 0, 0, [], 0, 0, [], 0, [], [], None
        state.requested_output = infer_requested_output(user_message)
        state.intent = state.requested_output
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
            try:
                response = self.provider.complete(messages, self.tools.schemas)
            except Exception as exc:
                state.status = "provider_error"
                state.warnings.append(f"LLM provider failure: {type(exc).__name__}")
                return self._finish("The configured language-model provider is unavailable.", state, started)
            state.tool_execution_stats["llm_api_calls"] += 1
            state.tool_results["llm_api_call_count"] = state.tool_execution_stats["llm_api_calls"]
            if response.http_status is not None:
                state.tool_results.setdefault("llm_http_statuses", []).append(response.http_status)
            if response.usage:
                totals = state.tool_results.setdefault("llm_usage", {})
                for key, value in response.usage.items():
                    totals[key] = totals.get(key, 0) + value
            logger.info("llm_response_received step=%s finish_reason=%s tool_call_count=%s", step, response.finish_reason, len(response.tool_calls))
            if not response.tool_calls:
                self._refresh_evidence_support(state)
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
