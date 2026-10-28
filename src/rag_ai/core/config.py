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
    chunk_size: int = Field(default=1000, ge=64, le=10_000)
    chunk_overlap: int = Field(default=150, ge=0, le=9_999)
    max_chunks_per_document: int = Field(default=5000, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> "Settings":
        if self.app_env == "production" and self.enable_document_parse_endpoint:
            raise ValueError("Document parsing cannot be enabled in production.")
        if self.chunk_overlap > self.chunk_size // 2:
            raise ValueError("CHUNK_OVERLAP must not exceed half of CHUNK_SIZE.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
