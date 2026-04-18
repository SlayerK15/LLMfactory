"""Pydantic-settings config — overrides CleaningConfig defaults from env/TOML."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLEAN_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    log_level: str = "INFO"

    # Default model IDs (overridable per-run via CleaningConfig)
    perplexity_model_id: str = "Qwen/Qwen2.5-0.5B"
    relevance_model_id: str = "BAAI/bge-m3"

    # Stage defaults
    near_dup_threshold: float = 0.8
    target_language: str = "en"
    relevance_threshold: float = 0.30
    perplexity_low_pct: float = 0.05
    perplexity_high_pct: float = 0.10

    # Feature flags
    enable_perplexity: bool = True
    enable_relevance: bool = True
    enable_trafilatura: bool = True


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
