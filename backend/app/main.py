import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.agent.mock_agent import MockAgent
from backend.app.core.config import settings
from backend.app.memory.store import InMemorySessionStore
from backend.app.models import AgentState, ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Historical Military GIS Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.backend_cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
store = InMemorySessionStore()
agent = MockAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "mock" if settings.mock_agent else settings.llm_provider}


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = store.get(request.session_id) or AgentState(session_id=request.session_id)
    logger.info("agent_decision=mock_agent session_id=%s", request.session_id)
    reply, state = agent.respond(request.message, state)
    store.save(state)
    return ChatResponse(session_id=request.session_id, reply=reply, state=state)

