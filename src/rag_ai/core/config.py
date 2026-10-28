"""Typed configuration loaded from the environment."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG-AI API"
    database_url: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
