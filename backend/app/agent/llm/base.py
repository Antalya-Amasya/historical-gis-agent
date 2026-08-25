from abc import ABC, abstractmethod
from backend.app.models import AgentModelResponse

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict[str, str]], tools: list[dict]) -> AgentModelResponse: ...
