"""Environment-driven configuration loading."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ForgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FORGE_", extra="ignore")

    data_dir: Path = Path("./data")
    groq_api_key: str = ""
    default_tokenizer: str = "BAAI/bge-m3"
    default_embed_model: str = "BAAI/bge-m3"
