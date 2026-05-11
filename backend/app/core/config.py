"""
Core application settings — single source of truth for all configuration.
All values loaded from environment with sensible production defaults.
"""
from functools import lru_cache
from typing import Literal
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "Multi-Agent Research Assistant"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Server ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    api_prefix: str = "/api/v1"

    # ── Security ─────────────────────────────────────────────────
    secret_key: str = Field(..., min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    cors_origins: list[str] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────────────
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/research_db"
    )
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # ── Redis ────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = 50
    cache_ttl_seconds: int = 3600

    # ── Anthropic / LLM ──────────────────────────────────────────
    anthropic_api_key: str = Field(...)
    orchestrator_model: str = "claude-opus-4-5"
    researcher_model: str = "claude-sonnet-4-5"
    critic_model: str = "claude-sonnet-4-5"
    synthesizer_model: str = "claude-sonnet-4-5"
    max_tokens: int = 8192
    temperature: float = 0.1
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 3

    # ── Agent pipeline ───────────────────────────────────────────
    max_researcher_concurrency: int = 3
    max_refinement_iterations: int = 2
    research_timeout_seconds: int = 300
    min_sources_per_subtopic: int = 2
    confidence_threshold: float = 0.6
    max_subtopics: int = 6

    # ── Search tools ─────────────────────────────────────────────
    brave_api_key: str = ""
    serpapi_key: str = ""
    semantic_scholar_api_key: str = ""

    # ── Observability ────────────────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    sentry_dsn: str = ""
    prometheus_enabled: bool = True
    otel_exporter_endpoint: str = ""

    # ── Rate limiting ────────────────────────────────────────────
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    # ── Vector store ─────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dim: int = 384
    similarity_top_k: int = 10

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
