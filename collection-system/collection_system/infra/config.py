"""Application configuration via pydantic-settings + TOML files."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the repo-root .env regardless of cwd. The layout is:
#   <repo>/collection-system/collection_system/infra/config.py
#                                              ^^^^ parents[3] == <repo>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ):
        # .env wins over process env — a stale shell-level OPENAI_API_KEY
        # (left over in the user profile) must not shadow the .env value.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # LLM
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    llm_provider: str = "groq"
    groq_model: str = "llama-3.3-70b-versatile"
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    cerebras_base_url: str = Field(default="https://api.cerebras.ai", alias="CEREBRAS_BASE_URL")
    cerebras_model: str = "llama-3.3-70b"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")  # blank = official API
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_fallback_provider: str = ""
    ollama_model: str = "qwen2.5:7b"

    # Database
    database_url: str = Field(default="", alias="DATABASE_URL")

    # Runtime
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    config_file: Path = Field(default=Path("configs/default.toml"), alias="CONFIG_FILE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    metrics_port: int = 8001

    # Query agent
    max_depth: int = 5
    max_queries: int = 10000
    relevance_threshold: float = 0.4
    llm_batch_size: int = 20

    # Search
    cc_cdx_index: str = "CC-MAIN-2026-12"
    searxng_base_url: str = "http://searxng:8888"
    searxng_rate_per_second: float = 2.0

    # Scraper
    scraper_concurrency: int = 40
    per_url_timeout_s: int = 45

    # Reliability
    global_timeout_s: int = 5400
    circuit_breaker_failures: int = 5
    circuit_breaker_cooldown_s: int = 60


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(**kwargs: object) -> None:
    """Used in tests to inject settings without environment variables."""
    global _settings
    _settings = Settings(**kwargs)  # type: ignore[arg-type]
