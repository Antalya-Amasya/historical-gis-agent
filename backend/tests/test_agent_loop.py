from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.rag.retriever import HistoricalRetriever

class Retriever(HistoricalRetriever):
    def __init__(self, items): self.items=items; self.calls=[]
    def retrieve(self, query, top_k=5, filters=None): self.calls.append((query,top_k,filters)); return self.items
class Geo:
    def __init__(self): self.calls=[]
    def call(self, tool, arguments):
        self.calls.append((tool,arguments))
        if tool == "resolve_ancient_place":
            values={"Carthago Nova":(37.6,-0.98,"265849"),"Rhodanus":(43.33,4.85,"148168")}
            if arguments["name"] not in values:return {"found":False}
            lat,lon,pid=values[arguments["name"]]
            return {"found":True,"id":"pleiades-"+pid,"canonical_name":arguments["name"],"latitude":lat,"longitude":lon,"source":"Pleiades: A Gazetteer of Past Places","source_id":pid,"source_url":"https://pleiades.stoa.org/places/"+pid,"confidence":.9,"uncertain":False,"coordinate_role":"exact_site"}
        if tool == "calculate_distance": return {"meters":1,"kilometers":.001,"method":"haversine_geodesic","source":"local"}
        return {"provider":"mock","source":"mock"}
def ev(identifier,text): return Evidence(id=identifier,author="Polybius",work="Histories",locator="Book III",excerpt=text,text=text,book="3",page_start=1,page_end=1,source_file="polybius.pdf",source_type="pdf")
def route_ev(): return [ev("move", "Synthetic test evidence: Hannibal's army marched from New Carthage to the Rhone.")]
def agent(script, evidence=None, max_steps=4, max_tool_executions=10, max_rag_search_executions=4, max_completion_corrections=1, max_completion_tool_executions=1): return HistoricalGisAgent(ScriptedLLMProvider(script),Retriever(evidence or []),Geo(),max_steps=max_steps,max_tool_executions=max_tool_executions,max_rag_search_executions=max_rag_search_executions,max_completion_corrections=max_completion_corrections,max_completion_tool_executions=max_completion_tool_executions)
def call(name,args,ident="x"): return AgentModelResponse(tool_calls=[AgentToolCall(id=ident,name=name,arguments=args)],finish_reason="tool_calls")

def test_tool_registry_has_only_allowlisted_schemas():
    names={item["name"] for item in AgentToolRegistry(Retriever([]),Geo()).schemas}
    assert names=={"search_historical_evidence","resolve_ancient_place","calculate_distance","get_elevation","get_elevation_profile","build_historical_route"}

def test_scripted_llm_search_then_route_then_finish():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Schematic route returned.")],route_ev())
    reply,state=subject.respond("show a route",AgentState(session_id="a"))
    assert state.historical_route and len(state.historical_route.ordered_points)==2
    assert [entry.tool_name for entry in state.tool_history]==["search_historical_evidence","build_historical_route"]
    assert state.final_answer==reply and state.historical_evidence

def test_scripted_llm_can_use_mcp_tool_without_bypassing_client():
    geo=Geo(); subject=HistoricalGisAgent(ScriptedLLMProvider([call("resolve_ancient_place",{"name":"Carthago Nova"}),AgentModelResponse(content="Resolved from MCP.")]),Retriever([]),geo)
    _,state=subject.respond("resolve a place",AgentState(session_id="b"))
    assert geo.calls==[("resolve_ancient_place",{"name":"Carthago Nova"})]
    assert state.resolved_places[0].canonical_name=="Carthago Nova"

def test_unknown_and_malformed_tools_are_audited_failures():
    subject=agent([call("not_allowed",{}),AgentModelResponse(content="Stopped.")])
    _,state=subject.respond("x",AgentState(session_id="c"))
    assert not state.tool_history[0].success and "Unknown agent tool" in state.tool_history[0].result_summary
    subject=agent([call("search_historical_evidence",{"top_k":"bad"}),AgentModelResponse(content="Stopped.")])
    _,state=subject.respond("x",AgentState(session_id="d"))
    assert not state.tool_history[0].success

def test_repeated_failures_and_max_steps_stop_safely():
    bad=call("not_allowed",{},"same")
    _,state=agent([bad,bad,bad]).respond("x",AgentState(session_id="e"))
    assert state.status=="tool_failure"
    looping=[call("search_historical_evidence",{"query":"x"},str(i)) for i in range(4)]
    _,state=HistoricalGisAgent(ScriptedLLMProvider(looping),Retriever([]),Geo(),max_steps=2).respond("x",AgentState(session_id="f"))
    assert state.status=="max_steps"

def test_insufficient_evidence_never_builds_route():
    subject=agent([call("search_historical_evidence",{"query":"Caesar"}),AgentModelResponse(content="Corpus evidence is insufficient; no route was built.")])
    _,state=subject.respond("show Caesar route",AgentState(session_id="g"))
    assert state.historical_evidence==[] and state.historical_route is None


def test_exact_successful_call_is_cached_and_provider_can_finish_after_duplicate():
    retriever=Retriever([ev("one","Alps")])
    script=[call("search_historical_evidence",{"query":"Alps","top_k":3},"one"),call("search_historical_evidence",{"top_k":3,"query":"Alps"},"two"),AgentModelResponse(content="Used cached evidence.")]
    subject=HistoricalGisAgent(ScriptedLLMProvider(script),retriever,Geo(),max_steps=4)
    reply,state=subject.respond("x",AgentState(session_id="dedup"))
    assert reply=="Used cached evidence." and len(retriever.calls)==1
    assert [entry.outcome for entry in state.tool_history]==["success","duplicate"]
    assert state.tool_execution_stats["duplicate_tool_calls"]==1 and state.tool_execution_stats["actual_tool_executions"]==1

def test_different_arguments_execute_again():
    retriever=Retriever([])
    script=[call("search_historical_evidence",{"query":"Alps","top_k":3},"one"),call("search_historical_evidence",{"query":"Alps baggage animals","top_k":3},"two"),AgentModelResponse(content="Done.")]
    _,state=HistoricalGisAgent(ScriptedLLMProvider(script),retriever,Geo()).respond("x",AgentState(session_id="different"))
    assert len(retriever.calls)==2 and state.tool_execution_stats["duplicate_tool_calls"]==0 and state.tool_execution_stats["rag_search_executions"]==2

def test_failed_tool_is_not_reused_as_successful_cache_hit():
    script=[call("search_historical_evidence",{"top_k":"bad"},"one"),call("search_historical_evidence",{"top_k":"bad"},"two"),AgentModelResponse(content="Stopped.")]
    _,state=agent(script).respond("x",AgentState(session_id="failed"))
    assert [entry.outcome for entry in state.tool_history]==["failure","failure"]
    assert state.tool_execution_stats["actual_tool_executions"]==2 and state.tool_execution_stats["duplicate_tool_calls"]==0

def test_duplicate_does_not_consume_budget_and_new_call_is_rejected_at_budget():
    retriever=Retriever([])
    script=[call("search_historical_evidence",{"query":"one"},"one"),call("search_historical_evidence",{"query":"one"},"two"),call("search_historical_evidence",{"query":"two"},"three"),AgentModelResponse(content="Use existing results.")]
    _,state=HistoricalGisAgent(ScriptedLLMProvider(script),retriever,Geo(),max_tool_executions=1).respond("x",AgentState(session_id="budget"))
    assert len(retriever.calls)==1
    assert [entry.outcome for entry in state.tool_history]==["success","duplicate","budget_rejected"]
    assert state.tool_execution_stats["actual_tool_executions"]==1 and state.tool_execution_stats["budget_rejected"]==1


def test_rag_search_budget_rejects_new_search_and_agent_can_finish():
    provider=ScriptedLLMProvider([call("search_historical_evidence",{"query":"first"},"one"),call("search_historical_evidence",{"query":"second"},"two"),call("search_historical_evidence",{"query":"third"},"three"),AgentModelResponse(content="Used existing evidence.")])
    subject=HistoricalGisAgent(provider,Retriever([ev("one","Alps")]),Geo(),max_rag_search_executions=2)
    reply,state=subject.respond("x",AgentState(session_id="search-budget"))
    assert reply=="Used existing evidence." and state.tool_execution_stats["actual_tool_executions"]==2
    assert [entry.outcome for entry in state.tool_history]==["success","success","search_budget_rejected"]
    assert state.tool_execution_stats["rag_search_executions"]==2 and state.tool_execution_stats["rag_search_budget_rejected"]==1
    import json
    rejected=json.loads([message for message in provider.requests[3]["messages"] if message["role"]=="tool"][-1]["content"])
    assert rejected["result"]["status"]=="search_budget_exhausted"


def test_duplicate_search_does_not_consume_rag_search_budget():
    retriever=Retriever([ev("one","Alps")])
    script=[call("search_historical_evidence",{"query":"same"},"one"),call("search_historical_evidence",{"query":"same"},"two"),call("search_historical_evidence",{"query":"other"},"three"),AgentModelResponse(content="Done.")]
    _,state=agent(script,retriever.items,max_rag_search_executions=2).respond("x",AgentState(session_id="search-duplicate"))
    assert state.tool_execution_stats["actual_tool_executions"]==2 and state.tool_execution_stats["rag_search_executions"]==2
    assert state.tool_execution_stats["duplicate_tool_calls"]==1 and state.tool_execution_stats["rag_search_budget_rejected"]==0


def test_non_rag_tool_is_not_limited_by_rag_search_budget():
    retriever=Retriever([])
    script=[call("search_historical_evidence",{"query":"one"},"one"),call("resolve_ancient_place",{"name":"Carthago Nova"},"two"),AgentModelResponse(content="Done.")]
    _,state=agent(script,max_rag_search_executions=1).respond("x",AgentState(session_id="non-rag"))
    assert state.resolved_places[0].canonical_name=="Carthago Nova"
    assert state.tool_execution_stats["rag_search_executions"]==1


def test_search_tool_result_has_sufficiency_summary_and_remaining_budget():
    provider=ScriptedLLMProvider([call("search_historical_evidence",{"query":"Alps"},"one"),AgentModelResponse(content="Done.")])
    subject=HistoricalGisAgent(provider,Retriever([ev("one","Alps"),ev("two","Snow")]),Geo(),max_rag_search_executions=4)
    subject.respond("x",AgentState(session_id="summary"))
    import json
    tool_message=next(message for message in provider.requests[1]["messages"] if message["role"]=="tool")
    result=json.loads(tool_message["content"])["result"]
    assert result["result_count"]==2 and result["accumulated_evidence_count"]==2
    assert result["unique_authors"]==["Polybius"] and result["remaining_search_budget"]==3


def test_route_final_without_route_triggers_one_completion_correction():
    provider=ScriptedLLMProvider([call("search_historical_evidence",{"query":"Hannibal"}),AgentModelResponse(content="Here is a route."),AgentModelResponse(content="Still no structured route.")])
    subject=HistoricalGisAgent(provider,Retriever([ev("n","New Carthage"),ev("h","Hannibal")]),Geo(),max_steps=4)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="route-correction"))
    assert state.requested_output=="historical_route" and state.tool_execution_stats["completion_corrections"]==1
    correction=provider.requests[2]["messages"][-1]
    assert correction["role"]=="user" and "build_historical_route" in correction["content"]


def test_route_correction_allows_llm_to_choose_builder_then_complete():
    provider=ScriptedLLMProvider([call("search_historical_evidence",{"query":"Hannibal"}),AgentModelResponse(content="I will answer without a route."),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="The structured route is available.")])
    subject=HistoricalGisAgent(provider,Retriever(route_ev()),Geo(),max_steps=5)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="route-builder"))
    assert state.status=="completed" and state.historical_route is not None
    assert state.tool_execution_stats["completion_corrections"]==1
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1


def test_route_with_no_evidence_and_explicit_insufficiency_can_finish():
    subject=agent([call("search_historical_evidence",{"query":"missing"}),AgentModelResponse(content="Available evidence is insufficient to generate a route.")])
    _,state=subject.respond("show a historical route",AgentState(session_id="route-insufficient"))
    assert state.status=="completed" and state.historical_route is None
    assert state.tool_execution_stats["completion_corrections"]==0


def test_route_correction_without_builder_stops_with_failed_contract():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal"}),AgentModelResponse(content="A route exists."),AgentModelResponse(content="I still will not call the builder.")],[ev("n","New Carthage"),ev("h","Hannibal")],max_steps=4)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="route-failed-contract"))
    assert state.status=="failed_contract" and "historical_route_required_but_not_built" in state.warnings
    assert state.tool_execution_stats["completion_corrections"]==1


def test_route_safeguard_refuses_when_builder_has_no_grounded_edge():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="An unsupported route is ready.")],[ev("bare","Hannibal crossed the Rhone.")],max_steps=3,max_completion_corrections=0)
    reply,state=subject.respond("show a Hannibal historical route",AgentState(session_id="grounded-route-required"))
    assert state.historical_route is None and state.status=="failed_contract"
    assert reply=="The requested HistoricalRoute was not built from the available Evidence."


def test_non_route_questions_do_not_require_builder():
    subject=agent([call("search_historical_evidence",{"query":"Polybius"}),AgentModelResponse(content="Grounded answer.")],[ev("one","Alps")])
    _,state=subject.respond("What does Polybius describe?",AgentState(session_id="ordinary-question"))
    assert state.requested_output=="answer" and state.status=="completed"
    assert state.tool_execution_stats["completion_corrections"]==0


def test_geography_question_does_not_require_builder():
    subject=agent([call("resolve_ancient_place",{"name":"Carthago Nova"}),AgentModelResponse(content="Resolved.")])
    _,state=subject.respond("What are the coordinates of Carthago Nova?",AgentState(session_id="geography-question"))
    assert state.requested_output=="geography_fact" and state.status=="completed"


def test_repeated_place_resolution_does_not_satisfy_route_contract():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal"}),call("resolve_ancient_place",{"name":"Carthago Nova"},"one"),call("resolve_ancient_place",{"name":"Rhodanus"},"two"),AgentModelResponse(content="The route is ready."),AgentModelResponse(content="No builder was called.")],[ev("n","New Carthage"),ev("h","Hannibal")],max_steps=6)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="resolved-not-route"))
    assert state.status=="failed_contract" and state.historical_route is None
    assert state.tool_execution_stats["completion_corrections"]==1


def test_existing_route_finishes_without_extra_correction():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Structured route complete.")],route_ev())
    _,state=subject.respond("show a historical route",AgentState(session_id="route-exists"))
    assert state.status=="completed" and state.historical_route is not None
    assert state.tool_execution_stats["completion_corrections"]==0


def test_route_builder_uses_general_budget_when_capacity_remains():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Done.")],route_ev(),max_tool_executions=3)
    _,state=subject.respond("show a historical route",AgentState(session_id="general-builder"))
    builder=next(item for item in state.tool_history if item.tool_name=="build_historical_route")
    assert state.status=="completed" and builder.budget_source=="general"
    assert state.tool_execution_stats["completion_reserved_executions"]==0 and state.tool_execution_stats["general_tool_executions"]==2


def test_route_builder_gets_one_reserved_execution_after_general_budget_exhaustion():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Done.")],route_ev(),max_tool_executions=1)
    _,state=subject.respond("show a historical route",AgentState(session_id="reserved-builder"))
    builder=next(item for item in state.tool_history if item.tool_name=="build_historical_route")
    assert state.status=="completed" and state.historical_route is not None
    assert builder.budget_source=="completion_reserved" and state.tool_execution_stats["completion_reserved_executions"]==1
    assert state.tool_execution_stats["general_tool_executions"]==1


def test_reserved_completion_slot_is_single_use_after_failure():
    bad={"event_id":"test","name":"Test route"}
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",bad,"one"),call("build_historical_route",bad,"two"),AgentModelResponse(content="Evidence is insufficient to generate a route.")],[ev("n","New Carthage")],max_tool_executions=1,max_steps=5)
    _,state=subject.respond("show a historical route",AgentState(session_id="reserved-once"))
    builders=[item for item in state.tool_history if item.tool_name=="build_historical_route"]
    assert [item.outcome for item in builders]==["failure","completion_budget_rejected"]
    assert state.tool_execution_stats["completion_reserved_executions"]==1 and state.tool_execution_stats["completion_budget_rejected"]==1


def test_non_completion_tools_remain_rejected_after_general_budget_exhaustion():
    subject=agent([call("search_historical_evidence",{"query":"one"}),call("resolve_ancient_place",{"name":"Carthago Nova"}),AgentModelResponse(content="Done.")],max_tool_executions=1)
    _,state=subject.respond("What does Polybius describe?",AgentState(session_id="general-reject"))
    assert state.tool_history[-1].outcome=="budget_rejected" and state.tool_execution_stats["completion_reserved_executions"]==0
    assert state.tool_execution_stats["general_budget_rejected"]==1


def test_non_route_request_does_not_get_builder_reservation():
    subject=agent([call("search_historical_evidence",{"query":"one"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Done.")],[ev("n","New Carthage"),ev("r","Rhone")],max_tool_executions=1)
    _,state=subject.respond("What does Polybius describe?",AgentState(session_id="non-route-builder"))
    builder=next(item for item in state.tool_history if item.tool_name=="build_historical_route")
    assert builder.outcome=="budget_rejected" and state.tool_execution_stats["completion_reserved_executions"]==0


def test_rag_search_cannot_use_completion_reservation():
    subject=agent([call("search_historical_evidence",{"query":"one"}),call("search_historical_evidence",{"query":"two"}),AgentModelResponse(content="Evidence is insufficient to generate a route.")],[ev("one","Alps")],max_tool_executions=1)
    _,state=subject.respond("show a historical route",AgentState(session_id="rag-no-reservation"))
    searches=[item for item in state.tool_history if item.tool_name=="search_historical_evidence"]
    assert [item.outcome for item in searches]==["success","budget_rejected"]
    assert state.tool_execution_stats["completion_reserved_executions"]==0


def test_duplicate_builder_does_not_consume_second_reservation():
    arguments={"event_id":"test","name":"Test route","period":"218 BCE"}
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",arguments,"one"),call("build_historical_route",arguments,"two"),AgentModelResponse(content="Done.")],[ev("n","New Carthage"),ev("r","Rhone")],max_tool_executions=1)
    _,state=subject.respond("show a historical route",AgentState(session_id="duplicate-builder"))
    builders=[item for item in state.tool_history if item.tool_name=="build_historical_route"]
    assert [item.outcome for item in builders]==["success","duplicate"]
    assert state.tool_execution_stats["completion_reserved_executions"]==1


def test_irrelevant_nonempty_evidence_allows_insufficient_route_finish_without_correction():
    subject=agent([call("search_historical_evidence",{"query":"Caesar Gaul"}),AgentModelResponse(content="Current evidence is insufficient to support a reliable route.")],[ev("one","Hannibal crossed the Alps")])
    _,state=subject.respond("show Caesar route in Gaul",AgentState(session_id="irrelevant-evidence"))
    assert state.status=="completed" and state.historical_route is None
    assert state.evidence_support_status=="irrelevant" and state.tool_execution_stats["completion_corrections"]==0
    assert "insufficient_relevant_evidence" in state.warnings


def test_unsupported_route_answer_gets_one_grounding_correction_then_can_finish():
    subject=agent([call("search_historical_evidence",{"query":"Caesar Gaul"}),AgentModelResponse(content="Caesar route goes A → B → C."),AgentModelResponse(content="Current Caesar/Gaul Evidence is insufficient to support a reliable route.")],[ev("one","Hannibal crossed the Alps")])
    _,state=subject.respond("show Caesar route in Gaul",AgentState(session_id="grounding-validator"))
    assert state.status=="completed" and state.grounding_corrections==1 and state.final_grounding_status=="provenance_corrected"
    assert state.historical_route is None


def test_repeated_unsupported_route_answer_fails_grounding_without_returning_leak():
    subject=agent([call("search_historical_evidence",{"query":"Caesar Gaul"}),AgentModelResponse(content="Key locations include A, B, C."),AgentModelResponse(content="Key locations include Foo, Bar.")],[ev("one","Hannibal crossed the Alps")])
    reply,state=subject.respond("show Caesar route in Gaul",AgentState(session_id="grounding-failed"))
    assert state.status=="completed_with_guardrail" and state.grounding_corrections==1
    assert state.final_grounding_status=="guardrail_fallback"
    assert state.unsupported_fact_claim_count > 0
    assert "unsupported_claims_with_insufficient_evidence" in state.warnings
    assert "A, B, C" not in reply and "甲、乙、丙" not in reply
    assert state.historical_route is None
    assert [item.tool_name for item in state.tool_history] == ["search_historical_evidence"]


def test_sufficient_hannibal_support_still_triggers_route_correction():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal Alps"}),AgentModelResponse(content="I will answer without a route."),AgentModelResponse(content="Still no route.")],[ev("one","Hannibal crossed the Alps")],max_steps=4)
    _,state=subject.respond("show Hannibal route over the Alps",AgentState(session_id="sufficient-route-correction"))
    assert state.evidence_support_status=="sufficient" and state.tool_execution_stats["completion_corrections"]==1


def test_ordinary_question_is_not_subject_to_route_support_gate():
    subject=agent([call("search_historical_evidence",{"query":"Caesar"}),AgentModelResponse(content="Grounded answer.")],[ev("one","Hannibal crossed the Alps")])
    _,state=subject.respond("What does Polybius describe?",AgentState(session_id="ordinary-support"))
    assert state.requested_output=="answer" and state.status=="completed"


def test_grounding_correction_allows_explicit_unverified_research_suggestions():
    provider=ScriptedLLMProvider([
        call("search_historical_evidence",{"query":"Caesar Gaul"}),
        AgentModelResponse(content="Foo was a decisive battle location."),
        AgentModelResponse(content="Current Evidence is insufficient for a route. Unverified research suggestions: Foo. This is not supported by current Evidence and is not a route node."),
    ])
    subject=HistoricalGisAgent(provider,Retriever([ev("one","Hannibal crossed the Alps")]),Geo(),max_steps=4)
    reply,state=subject.respond("show Caesar route in Gaul",AgentState(session_id="provenance-correction"))
    assert state.status=="completed" and state.grounding_corrections==1
    assert state.final_grounding_status=="provenance_corrected"
    assert state.unverified_suggestion_terms==["foo"] and state.historical_route is None
    assert "Unverified research suggestions" in reply



def test_work_title_grounding_correction_allows_corpus_gap_suggestion_without_gis():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Caesar Gaul"}),
        AgentModelResponse(content="Chronicles of Bar proves the route passed Fooport."),
        AgentModelResponse(content="Current Evidence does not include Chronicles of Bar. It may be worth consulting for further research; it is not a route node."),
    ])
    subject = HistoricalGisAgent(provider, Retriever([ev("one", "Hannibal crossed the Alps")]), Geo(), max_steps=4)
    _, state = subject.respond("show Caesar route in Gaul", AgentState(session_id="work-correction"))
    assert state.status == "completed" and state.grounding_corrections == 1
    assert state.final_grounding_status == "provenance_corrected"
    assert state.unverified_suggestion_terms == ["chronicles of bar"]
    assert state.unsupported_fact_terms == [] and state.historical_route is None
    assert [entry.tool_name for entry in state.tool_history] == ["search_historical_evidence"]


def test_repeated_model_only_work_citation_fails_grounding_without_gis():
    subject = agent([
        call("search_historical_evidence", {"query": "Caesar Gaul"}),
        AgentModelResponse(content="Annals of Bar proves the route passed Fooport."),
        AgentModelResponse(content="According to Annals of Bar, the army passed Fooport."),
    ], [ev("one", "Hannibal crossed the Alps")])
    _, state = subject.respond("show Caesar route in Gaul", AgentState(session_id="work-failed"))
    assert state.status == "completed_with_guardrail" and state.grounding_corrections == 1
    assert state.final_grounding_status == "guardrail_fallback"
    assert state.unsupported_fact_claim_count > 0 and state.historical_route is None
    assert [entry.tool_name for entry in state.tool_history] == ["search_historical_evidence"]



def test_grounding_validator_error_is_not_reported_as_guardrail_completion(monkeypatch):
    import backend.app.agent.loop as loop_module

    def raise_validator_error(*_args, **_kwargs):
        raise RuntimeError("validator unavailable")

    monkeypatch.setattr(loop_module, "assess_final_answer_provenance", raise_validator_error)
    subject = agent([
        call("search_historical_evidence", {"query": "Caesar Gaul"}),
        AgentModelResponse(content="Current Evidence is insufficient to support a route."),
    ], [ev("one", "Hannibal crossed the Alps")])
    reply, state = subject.respond("show Caesar route in Gaul", AgentState(session_id="validator-error"))
    assert state.status == "failed_grounding"
    assert state.final_grounding_status == "validator_error"
    assert "grounding_validator_error:RuntimeError" in state.warnings
    assert reply == "The system could not safely validate the requested HistoricalRoute response."
    assert state.historical_route is None


def test_final_answer_prompt_separates_user_prose_from_execution_diagnostics():
    provider = ScriptedLLMProvider([AgentModelResponse(content="## Historical answer\n\nEvidence supports this conclusion.")])
    subject = HistoricalGisAgent(provider, Retriever([]), Geo(), max_steps=1)

    reply, state = subject.respond("What happened?", AgentState(session_id="final-answer-boundary"))

    system_prompt = provider.requests[0]["messages"][0]["content"]
    assert reply == "## Historical answer\n\nEvidence supports this conclusion."
    assert state.final_answer == reply
    assert "final response is user-facing historical prose only" in system_prompt
    assert "search or tool budgets" in system_prompt
    assert state.tool_execution_stats["rag_search_budget_rejected"] == 0
