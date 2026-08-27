import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.deepseek import DeepSeekLLMProvider
from backend.app.agent.llm.fake import RuleBasedFakeLLMProvider
from backend.app.agent.model_router import ModelRouter
from backend.app.agent.mock_agent import MockAgent
from backend.app.core.config import settings
from backend.app.memory.store import InMemorySessionStore
from backend.app.models import AgentState, ChatRequest, ChatResponse, RagSearchRequest, RagSearchResponse
from backend.app.rag.http_store import build_production_retriever
from backend.app.routes.evidence import SemanticRouteEvidenceRetriever
from backend.app.candidate_routes.presentation import HistoricalRouteResponse
from backend.app.historical_route_presentation_service import (
    HistoricalRoutePresentationReadService, PresentationContractError, PresentationNotFoundError,
)
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Historical Military GIS Agent", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.backend_cors_origins.split(",")], allow_methods=["*"], allow_headers=["*"])
store = InMemorySessionStore()
def build_agent():
    if settings.agent_mode == "mock": return MockAgent(evidence_retriever=SemanticRouteEvidenceRetriever())
    kwargs = {"max_steps": settings.agent_max_steps, "max_tool_executions": settings.agent_max_tool_executions, "max_rag_search_executions": settings.agent_max_rag_search_executions, "max_completion_corrections": settings.agent_max_completion_corrections, "max_completion_tool_executions": settings.agent_max_completion_tool_executions, "max_grounding_corrections": settings.agent_max_grounding_corrections}
    if settings.agent_llm_provider != "deepseek": return HistoricalGisAgent(RuleBasedFakeLLMProvider(), SemanticRouteEvidenceRetriever(), **kwargs)
    router = ModelRouter(settings.deepseek_model_flash, settings.deepseek_model_pro, settings.deepseek_model, settings.agent_model_policy)
    def provider_factory(model_id: str): return DeepSeekLLMProvider(settings.deepseek_api_key, settings.deepseek_base_url, model_id, timeout_s=settings.deepseek_read_timeout_s, connect_timeout_s=settings.deepseek_connect_timeout_s)
    return HistoricalGisAgent(None, SemanticRouteEvidenceRetriever(), model_router=router, provider_factory=provider_factory, **kwargs)
agent = build_agent()
historical_route_presentation_service = HistoricalRoutePresentationReadService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "mock" if settings.agent_mode == "mock" else "bounded", "provider": settings.agent_llm_provider if settings.agent_mode != "mock" else "mock"}


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = store.get(request.session_id) or AgentState(session_id=request.session_id)
    logger.info("agent_mode=%s session_id=%s", settings.agent_mode, request.session_id)
    reply, state = agent.respond(request.message, state)
    store.save(state)
    return ChatResponse(session_id=request.session_id, reply=reply, state=state)


@app.get("/api/v1/historical-routes/{route_id}/presentation", response_model=HistoricalRouteResponse)
def historical_route_presentation(route_id: str) -> HistoricalRouteResponse:
    try:
        return historical_route_presentation_service.get_presentation(route_id)
    except PresentationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Historical route presentation not found") from exc
    except PresentationContractError as exc:
        raise HTTPException(status_code=422, detail="Historical route presentation contract is invalid") from exc


@app.post("/api/v1/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest) -> RagSearchResponse:
    retriever = build_production_retriever(settings)
    return RagSearchResponse(query=request.query, evidence=retriever.retrieve(request.query, request.top_k, request.filters))
