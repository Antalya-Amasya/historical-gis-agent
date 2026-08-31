from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence, EventPlaceRole, HistoricalEventType, TemporalGroundingStatus
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor


def evidence(identifier: str, text: str, period: str | None = None) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text, period=period)


def extract(*items: Evidence):
    return EvidenceGroundedHistoricalEventExtractor().extract(list(items))


def test_reform_event_preserves_explicit_statement_and_unresolved_temporal_text():
    events, diagnostics = extract(evidence("reform", "The reformer proposed a land law in Forum during his tribunate."))

    assert len(events) == 1 and events[0].event_type is HistoricalEventType.REFORM
    assert events[0].summary == "The reformer proposed a land law in Forum during his tribunate."
    assert events[0].place_mentions[0].raw_text == "Forum"
    assert events[0].place_mentions[0].canonical_hint is None
    assert events[0].temporal_grounding.status is TemporalGroundingStatus.UNRESOLVED
    assert diagnostics["reason_codes"] == ["EVENT_EXTRACTED", "TEMPORAL_UNRESOLVED"]


def test_battle_and_assassination_keep_site_mentions_with_evidence_refs():
    events, _ = extract(
        evidence("battle", "The armies fought at Place B."),
        evidence("assassination", "The leader was assassinated at Place C."),
    )

    assert [event.event_type for event in events] == [HistoricalEventType.BATTLE, HistoricalEventType.ASSASSINATION]
    assert [event.place_mentions[0].raw_text for event in events] == ["Place B", "Place C"]
    assert [event.place_mentions[0].evidence_refs for event in events] == [["battle"], ["assassination"]]
    assert all(not event.places for event in events)


def test_movement_event_is_not_a_route_or_coordinate_claim():
    events, _ = extract(evidence("movement", "The army marched from Place A to Place B."))

    assert len(events) == 1 and events[0].event_type is HistoricalEventType.MOVEMENT
    assert [item.raw_text for item in events[0].place_mentions] == ["Place A", "Place B"]
    assert not events[0].places and events[0].evidence == []
    assert "route inference" in events[0].limitations[0]


def test_same_sentence_anaphoric_movement_preserves_explicit_regional_origin():
    events, _ = extract(evidence(
        "anaphoric-movement",
        "The commander led his army into the country of the Arverni; "
        "and after marching from it to Oppidum B, he established a camp.",
    ))

    movement = next(event for event in events if event.event_type is HistoricalEventType.MOVEMENT)
    roles = {mention.raw_text: mention.role for mention in movement.place_mentions}
    assert roles["Arverni"] is EventPlaceRole.ORIGIN
    assert roles["Oppidum B"] is EventPlaceRole.DESTINATION
    assert movement.summary.startswith("The commander led his army into the country of the Arverni;")


def test_anaphoric_origin_is_not_carried_across_a_sentence_boundary():
    events, _ = extract(evidence(
        "separate-movement",
        "The commander entered the country of the Arverni. "
        "Later the army marched from it to Oppidum B.",
    ))

    movement = next(event for event in events if event.event_type is HistoricalEventType.MOVEMENT)
    assert all(mention.role is not EventPlaceRole.ORIGIN for mention in movement.place_mentions)


def test_non_event_evidence_cannot_fabricate_an_event():
    events, diagnostics = extract(evidence("plain", "The source names a person and a city without describing an action."))

    assert events == [] and diagnostics["reason_codes"] == ["NO_EVENT_EVIDENCE", "INSUFFICIENT_GROUNDING"]


def test_independent_statements_produce_multiple_events_without_merging():
    events, _ = extract(evidence("multiple", "The assembly elected an official. Later the army besieged Place D."))

    assert [event.event_type for event in events] == [HistoricalEventType.ELECTION, HistoricalEventType.SIEGE]
    assert len({event.id for event in events}) == 2
    assert events[0].evidence_refs == events[1].evidence_refs == ["multiple"]


def test_document_period_metadata_is_not_treated_as_event_time():
    events, _ = extract(evidence("dated", "The senate issued a decree.", "133 BCE"))

    temporal = events[0].temporal_grounding
    assert temporal.raw_expression is None
    assert temporal.normalized_start is None and temporal.normalized_end is None
    assert temporal.status is TemporalGroundingStatus.UNRESOLVED


def test_query_relevance_keeps_asserted_event_without_inventing_query_facts():
    item = evidence("asserted", "Commander A captured Town B in Province C.")

    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([item], query="Commander A in Province C")

    assert len(events) == 1
    assert events[0].summary == item.text
    assert events[0].event_type is HistoricalEventType.MILITARY
    assert events[0].evidence_refs == ["asserted"]


def test_retrospective_reference_is_not_classified_as_the_asserted_battle():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([
        evidence("retrospective", "After the defeat at Battlefield D, Settlement E revolted from the alliance."),
    ], query="Battlefield D")

    assert len(events) == 1
    assert events[0].event_type is HistoricalEventType.REBELLION
    assert events[0].summary.startswith("After the defeat at Battlefield D")


def test_hypothetical_and_reported_actions_do_not_become_completed_events():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([
        evidence("planned", "Commander A planned to attack Town B."),
        evidence("speech", 'The senator said that Commander C should march to Town D.'),
    ], query="Commander")

    assert events == []


def test_query_normalization_handles_ligatures_without_hard_coded_event_names():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([
        evidence("ligature", "Cæsar was assassinated at Forum E."),
        evidence("place", "At Cannæ the armies fought a battle."),
    ], query="Caesar Cannae")

    assert [event.event_type for event in events] == [HistoricalEventType.ASSASSINATION, HistoricalEventType.BATTLE]


def test_query_relevance_filters_unrelated_context_but_preserves_multiple_events():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([
        evidence("capture", "Commander A captured Town B in Province C."),
        evidence("siege", "Commander A besieged Fortress D in Province C."),
        evidence("context", "General Z captured Town Y in Province Q."),
    ], query="Commander A military actions in Province C")

    assert [event.evidence_refs for event in events] == [["capture"], ["siege"]]


def test_navigation_topic_is_not_treated_as_a_primary_source_statement():
    item = evidence("topic", "The queen was honoured by the city.")
    item.topic = "A famous battle"

    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([item], query="famous battle")

    assert events == []


def test_navigation_heading_is_not_promoted_to_an_event_statement():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([
        evidence("heading", "How a ruler was murdered (chapters 19-22)."),
    ], query="ruler murder")

    assert events == []


class Retriever(HistoricalRetriever):
    def retrieve(self, *_args, **_kwargs):
        return [evidence("agent-event", "The assembly elected an official in Place E.")]


class Geography:
    def call(self, *_args, **_kwargs):
        return {"found": False}


def test_agent_state_carries_answer_only_events_without_a_route():
    provider = ScriptedLLMProvider([
        AgentModelResponse(tool_calls=[AgentToolCall(id="search", name="search_historical_evidence", arguments={"query": "political event"})]),
        AgentModelResponse(content="The assembly elected an official."),
    ])
    agent = HistoricalGisAgent(provider, Retriever(), Geography(), max_steps=2)

    _, state = agent.respond("What political event occurred?", AgentState(session_id="event-answer-only"))

    assert state.historical_route is None and state.historical_route_presentation is None
    assert [event.event_type for event in state.historical_events] == [HistoricalEventType.ELECTION]
    assert state.historical_event_diagnostics["extraction"]["reason_codes"] == ["EVENT_EXTRACTED", "TEMPORAL_UNRESOLVED"]


def test_agent_state_carries_statement_grounded_normalized_time_without_creating_route_data():
    class DatedRetriever(HistoricalRetriever):
        def retrieve(self, *_args, **_kwargs):
            return [evidence("dated-agent", "The assembly elected an official at Place E in 133 BCE.")]

    provider = ScriptedLLMProvider([
        AgentModelResponse(tool_calls=[AgentToolCall(id="search", name="search_historical_evidence", arguments={"query": "political event"})]),
        AgentModelResponse(content="The assembly elected an official."),
    ])
    _, state = HistoricalGisAgent(provider, DatedRetriever(), Geography(), max_steps=2).respond(
        "What political event occurred?", AgentState(session_id="dated-event-answer-only"),
    )

    assert state.historical_events[0].temporal_grounding.normalized_start == "-133"
    assert state.historical_events[0].temporal_grounding.evidence_refs == ["dated-agent"]
    assert state.historical_route is None and state.historical_route_presentation is None
