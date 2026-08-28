from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mock_agent: bool = True  # legacy fallback flag
    agent_mode: str = "llm"
    llm_provider: str = "fake"
    agent_llm_provider: str = "fake"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str | None = None
    deepseek_model_flash: str | None = None
    deepseek_model_pro: str | None = None
    agent_model_policy: str = "flash_first"
    deepseek_connect_timeout_s: float = 10
    deepseek_read_timeout_s: float = 30
    llm_api_key: str | None = None
    llm_model: str | None = None
    agent_max_steps: int = 8
    agent_max_tool_executions: int = 10
    agent_max_rag_search_executions: int = 4
    agent_max_completion_corrections: int = 1
    agent_max_completion_tool_executions: int = 1
    agent_max_grounding_corrections: int = 1
    geography_mcp_url: str = "http://127.0.0.1:8001"
    geoapify_api_key: str | None = None
    opentopography_api_key: str | None = None
    geography_provider_mode: str = "mock"
    dem_hgt_dir: str | None = None
    dem_manifest_path: str | None = None
    dem_tile_cache_size: int = 16
    route_cell_size_m: float = 5_000.0
    backend_cors_origins: str = "http://127.0.0.1:5173"
    rag_chroma_path: str = "data/chroma"
    rag_embedding_provider: str = "semantic"
    rag_embedding_model: str = "intfloat/multilingual-e5-small"
    rag_embedding_device: str = "auto"
    rag_embedding_batch_size: int = 16
    rag_chroma_host: str = "127.0.0.1"
    rag_chroma_port: int = 8002
    rag_collection: str = "roman_republic_primary_sources_v2"
    rag_top_k: int = 5
    rag_query_bridge_enabled: bool = True
    roman_road_geojson_path: str = "data/raw/itiner_e/itinere_roads_zenodo_17122148.geojson"
    roman_road_enabled: bool = False


settings = Settings()

