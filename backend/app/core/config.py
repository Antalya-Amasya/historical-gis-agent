from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mock_agent: bool = True
    llm_provider: str = "mock"
    geography_mcp_url: str = "http://127.0.0.1:8001"
    geoapify_api_key: str | None = None
    opentopography_api_key: str | None = None
    geography_provider_mode: str = "mock"
    backend_cors_origins: str = "http://127.0.0.1:5173"
    rag_chroma_path: str = "data/chroma"
    rag_embedding_provider: str = "semantic"
    rag_embedding_model: str = "intfloat/multilingual-e5-small"
    rag_embedding_device: str = "auto"
    rag_embedding_batch_size: int = 16


settings = Settings()

