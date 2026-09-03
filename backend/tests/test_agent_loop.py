from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.evidence_support import validate_evidence_citations
from backend.app.agent.loop import ROUTE_PROSE_GROUNDING_FALLBACK, ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION, _model_result
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence, HistoricalEvent, HistoricalEventType
from backend.app.rag.retriever import HistoricalRetriever

class Retriever(HistoricalRetriever):
    def __init__(self, items): self.items=items; self.calls=[]
    def retrieve(self, query, top_k=5, filters=None): self.calls.append((query,top_k,filters)); return self.items

class QueryRetriever(HistoricalRetriever):
    def __init__(self, mapping): self.mapping=mapping; self.calls=[]
    def retrieve(self, query, top_k=5, filters=None):
        self.calls.append((query, top_k, filters))
        return self.mapping.get(query, [])
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
def terminal(answer, ids, insufficient_evidence=False): return call("submit_grounded_answer", {"answer":answer,"evidence_ids":ids,"insufficient_evidence":insufficient_evidence}, "terminal")
def agent(script, evidence=None, max_steps=4, max_tool_executions=10, max_rag_search_executions=4, max_completion_corrections=1, max_completion_tool_executions=1, max_grounding_corrections=1): return HistoricalGisAgent(ScriptedLLMProvider(script),Retriever(evidence or []),Geo(),max_steps=max_steps,max_tool_executions=max_tool_executions,max_rag_search_executions=max_rag_search_executions,max_completion_corrections=max_completion_corrections,max_completion_tool_executions=max_completion_tool_executions,max_grounding_corrections=max_grounding_corrections)
def call(name,args,ident="x"): return AgentModelResponse(tool_calls=[AgentToolCall(id=ident,name=name,arguments=args)],finish_reason="tool_calls")

def test_tool_registry_has_only_allowlisted_schemas():
    names={item["name"] for item in AgentToolRegistry(Retriever([]),Geo()).schemas}
    assert names=={"search_historical_evidence","resolve_ancient_place","calculate_distance","get_elevation","get_elevation_profile","build_historical_route","submit_grounded_answer"}

def test_scripted_llm_search_then_route_then_finish():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Schematic route returned.")],route_ev())
    reply,state=subject.respond("show a route",AgentState(session_id="a"))
    assert state.historical_route and len(state.historical_route.ordered_points)==2
    assert [entry.tool_name for entry in state.tool_history]==["search_historical_evidence","build_historical_route"]
    assert state.status=="completed_with_guardrail"
    assert reply==ROUTE_PROSE_GROUNDING_FALLBACK
    assert state.historical_evidence

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
    script=[call("search_historical_evidence",{"query":"Alps","top_k":3},"one"),call("search_historical_evidence",{"top_k":3,"query":"Alps"},"two"),terminal("Used cached evidence.",["one"])]
    subject=HistoricalGisAgent(ScriptedLLMProvider(script),retriever,Geo(),max_steps=4)
    reply,state=subject.respond("x",AgentState(session_id="dedup"))
    assert reply.startswith("Used cached evidence.") and len(retriever.calls)==1
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
    provider=ScriptedLLMProvider([call("search_historical_evidence",{"query":"first"},"one"),call("search_historical_evidence",{"query":"second"},"two"),call("search_historical_evidence",{"query":"third"},"three"),terminal("Used existing evidence.",["one"])])
    subject=HistoricalGisAgent(provider,Retriever([ev("one","Alps")]),Geo(),max_rag_search_executions=2)
    reply,state=subject.respond("x",AgentState(session_id="search-budget"))
    assert reply.startswith("Used existing evidence.") and state.tool_execution_stats["actual_tool_executions"]==2
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


def test_search_context_exposes_bounded_evidence_provenance_and_events_to_answer_step():
    provider=ScriptedLLMProvider([
        call("search_historical_evidence",{"query":"Hannibal marched"},"one"),
        terminal("Polybius records that Hannibal's army marched from New Carthage to the Rhone.",["move"]),
    ])
    subject=HistoricalGisAgent(provider,Retriever(route_ev()),Geo())
    subject.respond("What did Hannibal do?",AgentState(session_id="grounded-context"))
    import json
    tool_message=next(message for message in provider.requests[1]["messages"] if message["role"]=="tool")
    result=json.loads(tool_message["content"])["result"]
    assert result["evidence"][0] == {
        "id": "move", "author": "Polybius", "work": "Histories",
        "locator": "Book III", "excerpt": "Synthetic test evidence: Hannibal's army marched from New Carthage to the Rhone.",
    }
    assert result["historical_events"][0]["event_type"] == "MOVEMENT"
    assert result["historical_events"][0]["evidence_refs"] == ["move"]
    assert result["historical_events"][0]["summary"].endswith("New Carthage to the Rhone.")


def test_accumulated_evidence_is_extracted_against_user_request_not_last_search_subquery():
    item = ev("alesia", "Caesar fought at Alesia during the campaign.")
    state = AgentState(session_id="event-query", user_query="Caesar campaign")
    result, _ = AgentToolRegistry(Retriever([item]), Geo()).execute(
        "search_historical_evidence", {"query": "Rhine bridges"}, state,
    )
    assert result["success"]
    assert [(event.event_type.value, event.place_mentions[0].raw_text) for event in state.historical_events] == [("BATTLE", "Alesia")]


def test_sequential_retrievals_keep_prior_evidence_backed_events():
    alesia = ev("alesia", "Caesar fought at Alesia during the campaign.")
    rhine = ev("rhine", "Caesar crossed the Rhine with his army.")
    registry = AgentToolRegistry(
        QueryRetriever({"Caesar Alesia": [alesia], "Rhine bridges": [rhine]}),
        Geo(),
    )
    state = AgentState(session_id="sequential-retrieval", user_query="Caesar campaign route")
    registry.execute("search_historical_evidence", {"query": "Caesar Alesia"}, state)
    assert [(event.place_mentions[0].raw_text, event.event_type.value) for event in state.historical_events] == [("Alesia", "BATTLE")]
    registry.execute("search_historical_evidence", {"query": "Rhine bridges"}, state)
    assert [(event.event_type.value, event.place_mentions[0].raw_text if event.place_mentions else None) for event in state.historical_events] == [
        ("BATTLE", "Alesia"),
        ("MOVEMENT", "Rhine"),
    ]


def test_unrelated_subquery_does_not_erase_prior_task_relevant_events():
    alesia = ev("alesia", "Caesar fought at Alesia during the campaign.")
    registry = AgentToolRegistry(QueryRetriever({"Caesar Alesia": [alesia], "Rhine bridges": []}), Geo())
    state = AgentState(session_id="unrelated-subquery", user_query="Caesar campaign route")
    registry.execute("search_historical_evidence", {"query": "Caesar Alesia"}, state)
    registry.execute("search_historical_evidence", {"query": "Rhine bridges"}, state)
    assert [(event.event_type.value, event.place_mentions[0].raw_text) for event in state.historical_events] == [("BATTLE", "Alesia")]


def test_query_only_place_without_evidence_does_not_create_event_or_mention():
    item = ev("march", "The army marched from one camp to another.")
    state = AgentState(session_id="query-only-place", user_query="Atlantis expedition")
    _, _ = AgentToolRegistry(Retriever([item]), Geo()).execute(
        "search_historical_evidence", {"query": "Atlantis expedition"}, state,
    )
    assert state.historical_events == []
    assert all("Atlantis" not in mention.raw_text for event in state.historical_events for mention in event.place_mentions)


def test_empty_retrieval_subquery_does_not_expand_filter_context():
    alesia = ev("alesia", "Caesar fought at Alesia during the campaign.")
    registry = AgentToolRegistry(QueryRetriever({"Caesar Alesia": [alesia], "empty probe": []}), Geo())
    state = AgentState(session_id="empty-retrieval", user_query="Caesar campaign route")
    registry.execute("search_historical_evidence", {"query": "Caesar Alesia"}, state)
    registry.execute("search_historical_evidence", {"query": "empty probe"}, state)
    assert [(event.event_type.value, event.place_mentions[0].raw_text) for event in state.historical_events] == [("BATTLE", "Alesia")]


def test_cumulative_retrieval_context_is_request_local():
    alesia = ev("alesia", "Caesar fought at Alesia during the campaign.")
    registry = AgentToolRegistry(QueryRetriever({"Caesar Alesia": [alesia], "Rhine bridges": []}), Geo())
    first = AgentState(session_id="first-run", user_query="Caesar campaign")
    second = AgentState(session_id="second-run", user_query="Caesar campaign")
    registry.execute("search_historical_evidence", {"query": "Caesar Alesia"}, first)
    registry.execute("search_historical_evidence", {"query": "Rhine bridges"}, second)
    assert first.historical_events
    assert second.historical_events == []


def test_ordinary_qa_without_evidence_discards_provider_fact_claim():
    provider=ScriptedLLMProvider([AgentModelResponse(content="Atlantis was the decisive location.")])
    reply,state=HistoricalGisAgent(provider,Retriever([]),Geo()).respond(
        "What happened?",AgentState(session_id="no-evidence-answer")
    )
    assert reply == "The current retrieved historical evidence is insufficient to support a reliable answer."
    assert "Atlantis" not in reply
    assert state.status == "completed"
    assert state.grounding_corrections == 1
    assert "call search_historical_evidence" in provider.requests[1]["messages"][-1]["content"]


def test_ordinary_qa_unsupported_entity_is_corrected_against_evidence():
    provider=ScriptedLLMProvider([
        call("search_historical_evidence",{"query":"Hannibal"},"one"),
        AgentModelResponse(content="Atlantis was the decisive location."),
        terminal("Polybius records that Hannibal's army marched from New Carthage to the Rhone.",["move"]),
    ])
    reply,state=HistoricalGisAgent(provider,Retriever(route_ev()),Geo()).respond(
        "What did Hannibal do?",AgentState(session_id="answer-grounding")
    )
    assert "Atlantis" not in reply
    assert "Polybius" in reply
    assert state.grounding_corrections == 1
    assert state.final_grounding_status == "provenance_corrected"


def test_invalid_evidence_citation_is_corrected_then_validated():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal"}),
        terminal("Polybius records Hannibal.",["fabricated-999"]),
        terminal("Polybius records Hannibal.",["one"]),
    ])
    reply, state = HistoricalGisAgent(provider, Retriever([ev("one", "Hannibal marched")]), Geo()).respond("What did Hannibal do?", AgentState(session_id="citation-correct"))
    assert "fabricated-999" not in reply and state.final_grounding_status == "provenance_corrected"


def test_repeated_invalid_evidence_citation_fails_closed():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal"}),
        AgentModelResponse(content="Polybius records Hannibal. [Evidence: fabricated-999 — Polybius, Histories, Book III]"),
        AgentModelResponse(content="Polybius records Hannibal. [Evidence: fabricated-998 — Polybius, Histories, Book III]"),
    ])
    reply, state = HistoricalGisAgent(provider, Retriever([ev("one", "Hannibal marched")]), Geo()).respond("What did Hannibal do?", AgentState(session_id="citation-reject"))
    assert state.status == "completed_with_guardrail" and "fabricated" not in reply


def test_citation_metadata_must_match_the_visible_evidence_record():
    item = ev("one", "Hannibal marched")
    valid = "[Evidence: one — Polybius, Histories, Book III]"
    assert validate_evidence_citations(valid, [item], require_citation=True) == ()
    assert validate_evidence_citations("[Evidence: one — Livy, Histories, Book III]", [item], require_citation=True) == ("mismatched_evidence_author:one",)
    assert validate_evidence_citations("[Evidence: one — Polybius, Annals, Book III]", [item], require_citation=True) == ("mismatched_evidence_work:one",)
    assert validate_evidence_citations("[Evidence: one — Polybius, Histories, Book IV]", [item], require_citation=True) == ("mismatched_evidence_locator:one",)

def test_selected_evidence_ids_render_canonical_citations_and_deduplicate():
    provider=ScriptedLLMProvider([call("search_historical_evidence", {"query":"Hannibal"}), terminal("Hannibal marched.",["one","one"])])
    reply,state=HistoricalGisAgent(provider, Retriever([ev("one", "Hannibal marched")]), Geo()).respond("What did Hannibal do?", AgentState(session_id="selected-id"))
    assert reply.endswith("[Evidence: one — Polybius, Histories, Book III]") and reply.count("[Evidence: one") == 1
    assert state.final_grounding_status == "grounded"

def test_missing_or_hidden_selected_evidence_id_fails_closed_after_one_correction():
    provider=ScriptedLLMProvider([call("search_historical_evidence", {"query":"Hannibal"}), terminal("Hannibal marched.",[]), terminal("Hannibal marched.",["hidden"])])
    evidence=[ev("one", "Hannibal marched")] + [ev(str(index), "other") for index in range(2,9)] + [ev("hidden", "not visible")]
    reply,state=HistoricalGisAgent(provider, Retriever(evidence), Geo()).respond("What did Hannibal do?", AgentState(session_id="bad-selection"))
    assert state.status == "completed_with_guardrail" and "Hannibal marched" not in reply
    correction=provider.requests[2]["messages"][-1]["content"]
    assert "missing_evidence_selection" in correction and "one (Polybius, Histories, Book III)" in correction


def test_event_context_surfaces_movement_events_beyond_first_evidence_page():
    state = AgentState(session_id="event-context")
    state.historical_evidence = [ev(str(index), "Hannibal marched") for index in range(9)]
    state.historical_events = [
        HistoricalEvent(id="visible", name="Visible", summary="visible", event_type=HistoricalEventType.MOVEMENT, evidence_refs=["0"]),
        HistoricalEvent(id="hidden", name="Hidden", summary="hidden", event_type=HistoricalEventType.MOVEMENT, evidence_refs=["8"]),
        HistoricalEvent(id="mixed", name="Mixed", summary="mixed", event_type=HistoricalEventType.MOVEMENT, evidence_refs=["0", "8"]),
    ]
    result = _model_result("search_historical_evidence", {"result": {"result_count": 9}}, state)
    assert {event["id"] for event in result["historical_events"]} == {"visible", "hidden", "mixed"}


def test_geography_fact_requires_audited_source():
    reply, state = agent([AgentModelResponse(content="Alesia is in Gaul.")]).respond("What are the coordinates of Alesia?", AgentState(session_id="geo-closed"))
    assert state.final_grounding_status == "insufficient_evidence" and "Alesia" not in reply


def test_geography_fact_allows_audited_geography_tool_result():
    subject = agent([call("resolve_ancient_place", {"name": "Carthago Nova"}), AgentModelResponse(content="Carthago Nova is resolved by the audited geography source.")])
    reply, state = subject.respond("What are the coordinates of Carthago Nova?", AgentState(session_id="geo-audited"))
    assert state.status == "completed" and "Carthago Nova" in reply


def test_route_intent_with_evidence_attempts_builder_without_waiting_for_llm():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal"}),AgentModelResponse(content="Here is a route.")],[ev("n","New Carthage"),ev("h","Hannibal")])
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="route-correction"))
    assert state.requested_output=="historical_route"
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1
    assert state.tool_execution_stats["completion_corrections"]==0


def test_route_intent_does_not_require_a_successful_route():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal"}),AgentModelResponse(content="Available evidence is insufficient to generate a route.")],[ev("n","New Carthage"),ev("h","Hannibal")])
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="route-fail-closed"))
    assert state.historical_route is None and state.status=="completed"
    assert state.historical_route_diagnostics is not None
    assert state.historical_route_diagnostics.get("reason_codes")


def test_route_builder_is_not_invoked_twice_once_attempted():
    provider=ScriptedLLMProvider([call("search_historical_evidence",{"query":"Hannibal"}),AgentModelResponse(content="I will answer without a route."),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="The structured route is available.")])
    subject=HistoricalGisAgent(provider,Retriever(route_ev()),Geo(),max_steps=5)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="route-builder"))
    assert state.status=="completed_with_guardrail" and state.historical_route is not None
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1
    assert state.tool_execution_stats["completion_corrections"]==0


def test_route_with_no_evidence_and_explicit_insufficiency_can_finish():
    subject=agent([call("search_historical_evidence",{"query":"missing"}),AgentModelResponse(content="Available evidence is insufficient to generate a route.")])
    _,state=subject.respond("show a historical route",AgentState(session_id="route-insufficient"))
    assert state.status=="completed" and state.historical_route is None
    assert state.tool_execution_stats["completion_corrections"]==0
    assert all(item.tool_name != "build_historical_route" for item in state.tool_history)


def test_submit_grounded_answer_cannot_skip_route_builder_on_route_intent():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal"}),terminal("Evidence is insufficient to generate a route.",["h"],True)],[ev("h","Hannibal campaigned in Spain.")])
    _,state=subject.respond("展示汉尼拔路线",AgentState(session_id="route-terminal"))
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1
    assert state.historical_route is None


def test_route_intent_without_builder_call_still_gets_one_deterministic_attempt():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal"}),AgentModelResponse(content="A route exists.")],[ev("n","New Carthage"),ev("h","Hannibal")],max_steps=4)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="route-failed-contract"))
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1
    assert state.historical_route is None
    assert "historical_route_required_but_not_built" not in state.warnings


def test_route_safeguard_refuses_when_builder_has_no_grounded_edge():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="The retrieved evidence was insufficient to construct a historical route.")],[ev("bare","Hannibal crossed the Rhone.")],max_steps=3,max_completion_corrections=0)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="grounded-route-required"))
    assert state.historical_route is None and state.status=="completed"
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1


def test_non_route_questions_do_not_require_builder():
    subject=agent([call("search_historical_evidence",{"query":"Polybius"}),terminal("Grounded answer.",["one"])],[ev("one","Alps")])
    _,state=subject.respond("What does Polybius describe?",AgentState(session_id="ordinary-question"))
    assert state.requested_output=="answer" and state.status=="completed"
    assert state.tool_execution_stats["completion_corrections"]==0
    assert all(item.tool_name != "build_historical_route" for item in state.tool_history)


def test_geography_question_does_not_require_builder():
    subject=agent([call("resolve_ancient_place",{"name":"Carthago Nova"}),AgentModelResponse(content="Resolved.")])
    _,state=subject.respond("What are the coordinates of Carthago Nova?",AgentState(session_id="geography-question"))
    assert state.requested_output=="geography_fact" and state.status=="completed"


def test_repeated_place_resolution_does_not_replace_route_builder():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal"}),call("resolve_ancient_place",{"name":"Carthago Nova"},"one"),call("resolve_ancient_place",{"name":"Rhodanus"},"two"),AgentModelResponse(content="The retrieved evidence was insufficient to construct a historical route.")],[ev("n","New Carthage"),ev("h","Hannibal")],max_steps=6)
    _,state=subject.respond("show a Hannibal historical route",AgentState(session_id="resolved-not-route"))
    assert state.historical_route is None
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1
    assert state.tool_execution_stats["completion_corrections"]==0


def test_existing_route_finishes_without_extra_correction():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Structured route complete.")],route_ev())
    _,state=subject.respond("show a historical route",AgentState(session_id="route-exists"))
    assert state.status=="completed_with_guardrail" and state.historical_route is not None
    assert state.tool_execution_stats["completion_corrections"]==0


def test_route_builder_uses_general_budget_when_capacity_remains():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Done.")],route_ev(),max_tool_executions=3)
    _,state=subject.respond("show a historical route",AgentState(session_id="general-builder"))
    builder=next(item for item in state.tool_history if item.tool_name=="build_historical_route")
    assert state.status=="completed_with_guardrail" and builder.budget_source=="general"
    assert state.tool_execution_stats["completion_reserved_executions"]==0 and state.tool_execution_stats["general_tool_executions"]==2


def test_route_builder_gets_one_reserved_execution_after_general_budget_exhaustion():
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",{"event_id":"test","name":"Test route","period":"218 BCE"}),AgentModelResponse(content="Done.")],route_ev(),max_tool_executions=1)
    _,state=subject.respond("show a historical route",AgentState(session_id="reserved-builder"))
    builder=next(item for item in state.tool_history if item.tool_name=="build_historical_route")
    assert state.status=="completed_with_guardrail" and state.historical_route is not None
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
    builder=next(item for item in state.tool_history if item.tool_name=="build_historical_route")
    assert [item.outcome for item in searches]==["success","budget_rejected"]
    assert builder.budget_source=="completion_reserved" and state.tool_execution_stats["completion_reserved_executions"]==1


def test_duplicate_builder_does_not_consume_second_reservation():
    arguments={"event_id":"test","name":"Test route","period":"218 BCE"}
    subject=agent([call("search_historical_evidence",{"query":"route"}),call("build_historical_route",arguments,"one"),call("build_historical_route",arguments,"two"),AgentModelResponse(content="Done.")],[ev("n","New Carthage"),ev("r","Rhone")],max_tool_executions=1)
    _,state=subject.respond("show a historical route",AgentState(session_id="duplicate-builder"))
    builders=[item for item in state.tool_history if item.tool_name=="build_historical_route"]
    assert [item.outcome for item in builders]==["success","duplicate"]
    assert state.tool_execution_stats["completion_reserved_executions"]==1


def test_irrelevant_nonempty_evidence_still_attempts_builder_then_fail_closed():
    subject=agent([call("search_historical_evidence",{"query":"Caesar Gaul"}),AgentModelResponse(content="Current evidence is insufficient to support a reliable route.")],[ev("one","Hannibal crossed the Alps")])
    _,state=subject.respond("show Caesar route in Gaul",AgentState(session_id="irrelevant-evidence"))
    assert state.status=="completed" and state.historical_route is None
    assert state.evidence_support_status=="irrelevant"
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1
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
    assert [item.tool_name for item in state.tool_history] == ["search_historical_evidence", "build_historical_route"]


def test_sufficient_hannibal_support_attempts_builder_instead_of_correction():
    subject=agent([call("search_historical_evidence",{"query":"Hannibal Alps"}),AgentModelResponse(content="I will answer without a route.")],[ev("one","Hannibal crossed the Alps")],max_steps=4)
    _,state=subject.respond("show Hannibal route over the Alps",AgentState(session_id="sufficient-route-correction"))
    assert state.evidence_support_status=="sufficient"
    assert [item.tool_name for item in state.tool_history].count("build_historical_route")==1
    assert state.tool_execution_stats["completion_corrections"]==0


def test_ordinary_question_is_not_subject_to_route_support_gate():
    subject=agent([call("search_historical_evidence",{"query":"Caesar"}),terminal("Grounded answer.",["one"])],[ev("one","Hannibal crossed the Alps")])
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
    assert [entry.tool_name for entry in state.tool_history] == ["search_historical_evidence", "build_historical_route"]


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
    assert [entry.tool_name for entry in state.tool_history] == ["search_historical_evidence", "build_historical_route"]



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
    assert reply == "The current retrieved historical evidence is insufficient to support a reliable answer."
    assert state.final_answer == reply
    assert "final response is user-facing historical prose only" in system_prompt
    assert "search or tool budgets" in system_prompt
    assert state.tool_execution_stats["rag_search_budget_rejected"] == 0
    assert state.final_grounding_status == "insufficient_evidence"
