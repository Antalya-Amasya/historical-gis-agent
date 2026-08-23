from abc import ABC, abstractmethod

from backend.app.models import AgentState


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> AgentState | None: ...

    @abstractmethod
    def save(self, state: AgentState) -> None: ...


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}

    def get(self, session_id: str) -> AgentState | None:
        return self._states.get(session_id)

    def save(self, state: AgentState) -> None:
        self._states[state.session_id] = state

