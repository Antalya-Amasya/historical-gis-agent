import logging
import sqlite3
from contextlib import closing

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.deepseek import DeepSeekLLMProvider
from backend.app.agent.llm.fake import RuleBasedFakeLLMProvider
from backend.app.agent.llm.zhipu import ZhipuLLMProvider
from backend.app.agent.model_router import ModelRouter
from backend.app.agent.mock_agent import MockAgent
from backend.app.core.config import settings
from backend.app.memory.store import InMemorySessionStore
from backend.app.models import AgentState, ChatRequest, ChatResponse, RagSearchRequest, RagSearchResponse
from backend.app.route_result_status import derive_route_result_status
from backend.app.rag.http_store import build_production_retriever
from backend.app.routes.evidence import SemanticRouteEvidenceRetriever
from backend.app.candidate_routes.presentation import HistoricalRouteResponse
from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.candidate_routes.geographic import GeographicCandidateRouteService
from backend.app.candidate_routes.terrain import MosaicDEMProvider
from backend.app.roads.itiner_e import RomanRoadGraph
from backend.app.gis.natural_earth_surface import NaturalEarthAvailability, NaturalEarthSurfaceClassifier
from backend.app.geography.place_registry import _connect_read_only, _index_path, _metadata
from pathlib import Path
from backend.app.historical_route_presentation_service import (
    HistoricalRoutePresentationReadService, PresentationContractError, PresentationNotFoundError,
)
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Historical Military GIS Agent", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.backend_cors_origins.split(",")], allow_methods=["*"], allow_headers=["*"])
store = InMemorySessionStore()
def build_agent(*, roman_road_orchestrator=None):
    if settings.agent_mode == "mock":
        return MockAgent(evidence_retriever=SemanticRouteEvidenceRetriever())
    kwargs = {
        "max_steps": settings.agent_max_steps,
        "max_tool_executions": settings.agent_max_tool_executions,
        "max_rag_search_executions": settings.agent_max_rag_search_executions,
        "max_completion_corrections": settings.agent_max_completion_corrections,
        "max_completion_tool_executions": settings.agent_max_completion_tool_executions,
        "max_grounding_corrections": settings.agent_max_grounding_corrections,
    }
    retriever = SemanticRouteEvidenceRetriever()
    if settings.agent_llm_provider == "deepseek":
        router = ModelRouter(
            settings.deepseek_model_flash,
            settings.deepseek_model_pro,
            settings.deepseek_model,
            settings.agent_model_policy,
        )
        def provider_factory(model_id: str):
            return DeepSeekLLMProvider(
                settings.deepseek_api_key,
                settings.deepseek_base_url,
                model_id,
                timeout_s=settings.deepseek_read_timeout_s,
                connect_timeout_s=settings.deepseek_connect_timeout_s,
            )
        return HistoricalGisAgent(
            None,
            retriever,
            model_router=router,
            provider_factory=provider_factory,
            roman_road_orchestrator=roman_road_orchestrator,
            **kwargs,
        )
    if settings.agent_llm_provider == "zhipu":
        router = ModelRouter(
            settings.zhipu_model_flash,
            settings.zhipu_model_pro,
            settings.zhipu_model,
            settings.agent_model_policy,
        )
        def provider_factory(model_id: str):
            return ZhipuLLMProvider(
                settings.zhipu_api_key,
                settings.zhipu_base_url,
                model_id,
                timeout_s=settings.zhipu_read_timeout_s,
                connect_timeout_s=settings.zhipu_connect_timeout_s,
            )
        return HistoricalGisAgent(
            None,
            retriever,
            model_router=router,
            provider_factory=provider_factory,
            roman_road_orchestrator=roman_road_orchestrator,
            **kwargs,
        )
    return HistoricalGisAgent(
        RuleBasedFakeLLMProvider(),
        retriever,
        roman_road_orchestrator=roman_road_orchestrator,
        **kwargs,
    )
agent = build_agent()
historical_route_presentation_service = HistoricalRoutePresentationReadService()


def maritime_surface_from_settings(data_root: str | None = None):
    """Load a manifest-backed Natural Earth classifier once; never bypass R3-B1 audit."""
    root = settings.maritime_surface_data_root if data_root is None else data_root
    if not root:
        return None
    dataset = Path(root)
    provider = NaturalEarthSurfaceClassifier.from_manifest(dataset, dataset / "surface-manifest.json")
    if provider.audit.availability is not NaturalEarthAvailability.AVAILABLE:
        return None
    return provider


@app.on_event("startup")
def compose_roman_road_capability() -> None:
    """Load independently configured GIS assets once per application lifecycle."""
    global agent
    index_path, _ = _index_path()
    pleiades_status = "UNAVAILABLE"
    if index_path is not None and index_path.is_file():
        try:
            with closing(_connect_read_only(index_path)) as connection:
                _metadata(connection)
            pleiades_status = "ACTIVE"
        except (OSError, RuntimeError, sqlite3.Error):
            logger.warning("Pleiades index could not be verified", exc_info=True)
    road_service = None
    if settings.roman_road_enabled:
        path = Path(settings.roman_road_geojson_path)
        if not path.is_file():
            raise RuntimeError(f"roman-road capability is enabled but dataset is unavailable: {path}")
        road_service = RomanRoadCandidateService(RomanRoadGraph.load(path))
    terrain_service = None
    if settings.dem_hgt_dir:
        dem_path = Path(settings.dem_hgt_dir)
        if not dem_path.is_dir():
            raise RuntimeError(f"configured SRTM terrain directory is unavailable: {dem_path}")
        terrain_service = GeographicCandidateRouteService(MosaicDEMProvider(dem_path))
    maritime_surface = maritime_surface_from_settings()
    if road_service is not None or maritime_surface is not None:
        agent = build_agent(roman_road_orchestrator=RomanRoadRouteOrchestrator(
            road_service,
            terrain_route_service=terrain_service,
            maritime_surface=maritime_surface,
        ))
    app.state.gis_assets = {
        "pleiades": pleiades_status,
        "srtm": "ACTIVE" if terrain_service is not None else "UNAVAILABLE",
        "itiner_e": "ACTIVE" if road_service is not None else "UNAVAILABLE",
        "natural_earth": "ACTIVE" if maritime_surface is not None else "UNAVAILABLE",
    }
    logger.info("GIS assets: %s", app.state.gis_assets)


@app.get("/health")
def health() -> dict[str, str]:
    payload = {
        "status": "ok",
        "agent": "mock" if settings.agent_mode == "mock" else "bounded",
        "provider": settings.agent_llm_provider if settings.agent_mode != "mock" else "mock",
    }
    payload.update(getattr(app.state, "gis_assets", {}))
    if settings.agent_mode != "mock" and settings.agent_llm_provider == "zhipu":
        payload["model_policy"] = settings.agent_model_policy
        payload["default_model"] = settings.zhipu_model_flash or settings.zhipu_model or ""
        payload["pro_model"] = settings.zhipu_model_pro or ""
    elif settings.agent_mode != "mock" and settings.agent_llm_provider == "deepseek":
        payload["model_policy"] = settings.agent_model_policy
        payload["default_model"] = settings.deepseek_model_flash or settings.deepseek_model or ""
        payload["pro_model"] = settings.deepseek_model_pro or ""
    return payload


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = store.get(request.session_id) or AgentState(session_id=request.session_id)
    logger.info("agent_mode=%s session_id=%s", settings.agent_mode, request.session_id)
    reply, state = agent.respond(request.message, state)
    store.save(state)
    status = derive_route_result_status(state)
    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        state=state,
        route_result_status=status.value if status is not None else None,
    )


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
