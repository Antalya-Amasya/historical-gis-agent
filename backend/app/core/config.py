from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mock_agent: bool = True
    llm_provider: str = "mock"
    geography_mcp_url: str = "http://127.0.0.1:8001"
    backend_cors_origins: str = "http://127.0.0.1:5173"
    rag_chroma_path: str = "data/chroma"


settings = Settings()

