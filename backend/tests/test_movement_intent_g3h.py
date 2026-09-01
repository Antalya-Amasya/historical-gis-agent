"""G3H: general movement-intent classification and subject extraction."""

from backend.app.agent.evidence_support import assess_evidence_support, extract_subject_terms
from backend.app.agent.loop import infer_requested_output
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.rag.retriever import HistoricalRetriever


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Plutarch",
        work="Life of Sulla",
        locator="1",
        excerpt=text,
        text=text,
    )


class Retriever(HistoricalRetriever):
    def __init__(self, items):
        self.items = items

    def retrieve(self, query, top_k=5, filters=None):
        return list(self.items)


class RegistryGeography:
    def call(self, tool, arguments):
        from geography_mcp.service import GeographyService

        return GeographyService().resolve_ancient_place_payload(arguments["name"])


def call(name, args, ident="x"):
    return AgentModelResponse(
        tool_calls=[AgentToolCall(id=ident, name=name, arguments=args)],
        finish_reason="tool_calls",
    )


def test_movement_from_to_question_requests_historical_route():
    assert infer_requested_output("How did Sulla move from Greece to Italy?") == "historical_route"


def test_explicit_route_question_still_requests_historical_route():
    assert infer_requested_output("Show the route from Greece to Italy") == "historical_route"


def test_crossing_question_requests_historical_route():
    assert infer_requested_output("How did Hannibal cross the Alps?") == "historical_route"


def test_march_into_question_requests_historical_route():
    assert infer_requested_output("How did the army march into Italy?") == "historical_route"


def test_ordinary_question_stays_answer():
    assert infer_requested_output("What does Polybius describe?") == "answer"


def test_geography_question_stays_geography_fact():
    assert infer_requested_output("What are the coordinates of Carthago Nova?") == "geography_fact"


def test_place_lookup_stays_answer():
    assert infer_requested_output("Where is Rome?") == "answer"


def test_metaphorical_move_without_direction_stays_answer():
    assert infer_requested_output("How did Roman politics move forward in 133 BCE?") == "answer"


def test_bridge_person_subject_term_extraction():
    terms = extract_subject_terms("How did Sulla move from Greece to Italy?")
    assert "sulla" in terms


def test_movement_route_subject_support_with_matching_evidence():
    text = (
        "Sulla left them and sailed for Greece, and thence passed on to Italy "
        "with the greater part of his army."
    )
    result = assess_evidence_support(
        "How did Sulla move from Greece to Italy?",
        "historical_route",
        [evidence("sulla-sail", text)],
    )
    assert result.status == "sufficient"
    assert "sulla" in result.matched_subject_terms


def test_movement_question_triggers_route_builder_with_sulla_evidence():
    text = (
        "Sulla left them and sailed for Greece, and thence passed on to Italy "
        "with the greater part of his army."
    )
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Sulla Greece Italy"}, "search"),
        AgentModelResponse(content="The movement is documented in the retrieved evidence."),
    ])
    agent = HistoricalGisAgent(
        provider,
        Retriever([evidence("sulla-sail", text)]),
        RegistryGeography(),
        max_steps=4,
    )
    _, state = agent.respond(
        "How did Sulla move from Greece to Italy?",
        AgentState(session_id="g3h-movement-route"),
    )
    assert state.requested_output == "historical_route"
    assert any(entry.tool_name == "build_historical_route" for entry in state.tool_history)
    assert state.historical_route is not None
    assert len(state.historical_route.ordered_points) == 2
    assert state.historical_route_presentation is not None
