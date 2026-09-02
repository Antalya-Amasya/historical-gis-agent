"""G4G-1: movement-display intent routing hardening."""

import pytest

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.loop import infer_requested_output
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.rag.retriever import HistoricalRetriever


ROUTE_CASES = [
    "Show Caesar's movements during the Gallic Wars.",
    "Trace Caesar's movements during the Gallic Wars.",
    "Show Hannibal's route through the Alps.",
    "Map Sulla's march on Rome.",
    "Where did Scipio move during the campaign?",
    "Follow the Roman army's advance through Italy.",
    "Show the movements of Caesar during the Gallic Wars.",
    "Trace the campaign movements of Hannibal.",
    "How did Sulla move from Greece to Italy?",
    "Show Caesar's route.",
    "Can you show me where Caesar moved during the Gallic Wars?",
    "I want to trace Hannibal's movements.",
]

ANSWER_CASES = [
    "What were the political consequences of Caesar's movements?",
    "Why were Hannibal's movements important?",
    "Explain Sulla's march on Rome.",
    "What caused Hannibal to cross the Alps?",
    "Describe Caesar's strategy during the Gallic Wars.",
    "How important was Roman military mobility?",
    "Tell me about Hannibal's movements.",
]


@pytest.mark.parametrize("query", ROUTE_CASES)
def test_movement_display_queries_request_historical_route(query: str):
    assert infer_requested_output(query) == "historical_route"


@pytest.mark.parametrize("query", ANSWER_CASES)
def test_analytical_movement_queries_stay_answer(query: str):
    assert infer_requested_output(query) == "answer"


def test_caesar_gallic_wars_movements_regression():
    assert (
        infer_requested_output("Show Caesar's movements during the Gallic Wars.")
        == "historical_route"
    )


class Retriever(HistoricalRetriever):
    def retrieve(self, query, top_k=5, filters=None):
        return [
            Evidence(
                id="one",
                author="Caesar",
                work="Commentaries",
                locator="Book 1",
                excerpt="Caesar marched through Gaul.",
                text="Caesar marched through Gaul.",
            )
        ]


def test_caesar_movements_enters_route_capable_agent_loop():
    provider = ScriptedLLMProvider(
        [AgentModelResponse(content="Searching evidence for Caesar's movements.")]
    )
    agent = HistoricalGisAgent(provider, Retriever(), max_steps=1)
    _, state = agent.respond(
        "Show Caesar's movements during the Gallic Wars.",
        AgentState(session_id="g4g1-caesar-route-intent"),
    )
    assert state.requested_output == "historical_route"
    assert state.intent == "historical_route"
