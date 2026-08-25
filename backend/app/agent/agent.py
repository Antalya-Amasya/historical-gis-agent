import logging
from backend.app.agent.loop import BoundedAgentLoop, infer_requested_output
from backend.app.agent.tools import AgentToolRegistry
from backend.app.geography.mcp_client import GeographyMcpClient
from backend.app.models import AgentState
from backend.app.rag.retriever import HistoricalRetriever

logger = logging.getLogger(__name__)


class HistoricalGisAgent:
    def __init__(self, provider, evidence_retriever: HistoricalRetriever, geography_client=None, max_steps: int=8, max_tool_executions: int=10, max_rag_search_executions: int=4, max_completion_corrections: int=1, max_completion_tool_executions: int=1, max_grounding_corrections: int=1, model_router=None, provider_factory=None):
        self.provider = provider
        self.model_router = model_router
        self.provider_factory = provider_factory
        self.tools = AgentToolRegistry(evidence_retriever, geography_client or GeographyMcpClient())
        self.max_steps = max_steps
        self.max_tool_executions = max_tool_executions
        self.max_rag_search_executions = max_rag_search_executions
        self.max_completion_corrections = max_completion_corrections
        self.max_completion_tool_executions = max_completion_tool_executions
        self.max_grounding_corrections = max_grounding_corrections

    def respond(self, message: str, state: AgentState):
        provider = self.provider
        if self.model_router is not None:
            requested_output = infer_requested_output(message)
            quality_mode = state.assumptions.get("quality_mode") if state.assumptions else None
            selected = self.model_router.select_model(requested_output, requested_output, quality_mode=quality_mode)
            state.selected_model_tier = selected.tier
            state.selected_model_id = selected.model_id
            state.model_policy = selected.policy
            state.quality_mode = quality_mode
            logger.info("model_selected tier=%s policy=%s reason=%s", selected.tier, selected.policy, selected.reason)
            provider = self.provider_factory(selected.model_id)
        loop = BoundedAgentLoop(provider, self.tools, self.max_steps, self.max_tool_executions, self.max_rag_search_executions, self.max_completion_corrections, self.max_completion_tool_executions, self.max_grounding_corrections)
        return loop.run(message, state)
