"""Typed configuration loaded from the environment."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG-AI API"
    app_env: Literal["development", "test", "production"] = "production"
    enable_document_parse_endpoint: bool = False
    database_url: SecretStr | None = None
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_pdf_pages: int = Field(default=500, ge=1)
    max_extracted_characters: int = Field(default=2_000_000, ge=1)

    @model_validator(mode="after")
    def reject_parse_endpoint_in_production(self) -> "Settings":
        if self.app_env == "production" and self.enable_document_parse_endpoint:
            raise ValueError("Document parsing cannot be enabled in production.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
