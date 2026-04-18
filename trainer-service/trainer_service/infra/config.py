"""Environment-driven settings."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TRAINER_", extra="ignore")

    workspace: Path = Path("./data/trainer")
    aws_region: str = "ap-south-1"
    s3_bucket: str = ""
    s3_key_prefix: str = "trainer"
    modal_app_name: str = "collection-system-trainer"
