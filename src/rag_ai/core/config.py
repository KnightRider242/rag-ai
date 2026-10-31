"""Typed configuration loaded from the environment."""

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def validate_local_ollama_url(value: str) -> str:
    """Accept only an explicit loopback HTTP endpoint, without credentials."""
    try:
        parsed = urlparse(value)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.port is not None
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Ollama endpoint must be a loopback HTTP URL with an explicit port.")
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG-AI API"
    app_env: Literal["development", "test", "production"] = "production"
    enable_document_parse_endpoint: bool = False
    enable_document_index_endpoint: bool = False
    enable_document_retrieval_endpoint: bool = False
    enable_rag_answer_endpoint: bool = False
    database_url: SecretStr | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "embeddinggemma:300m"
    generation_model: str = "qwen3:4b-instruct"
    generation_num_ctx: int = Field(default=4096, ge=2048, le=8192)
    generation_num_predict: int = Field(default=512, ge=64, le=1024)
    generation_temperature: float = Field(default=0, ge=0, le=1)
    generation_max_response_bytes: int = Field(default=32768, ge=4096, le=131072)
    rag_max_chunks: int = Field(default=5, ge=1, le=5)
    rag_max_chunk_characters: int = Field(default=1200, ge=64, le=2000)
    rag_max_context_characters: int = Field(default=4000, ge=256, le=8000)
    max_embedding_dimensions: int = Field(default=4096, ge=1, le=16_000)
    max_embedding_elements: int = Field(default=1_000_000, ge=1, le=2_000_000)
    max_index_chunks_per_document: int = Field(default=500, ge=1, le=5000)
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
        if self.app_env == "production" and self.enable_document_index_endpoint:
            raise ValueError("Document indexing cannot be enabled in production.")
        if self.app_env == "production" and self.enable_document_retrieval_endpoint:
            raise ValueError("Document retrieval cannot be enabled in production.")
        if self.app_env == "production" and self.enable_rag_answer_endpoint:
            raise ValueError("RAG answer endpoint cannot be enabled in production.")
        try:
            validate_local_ollama_url(self.ollama_base_url)
        except ValueError as exc:
            raise ValueError("OLLAMA_BASE_URL must be a loopback HTTP URL with an explicit port.") from exc
        if not self.embedding_model.strip() or len(self.embedding_model) > 200:
            raise ValueError("EMBEDDING_MODEL must be a nonempty model name.")
        if not self.generation_model.strip() or len(self.generation_model) > 200:
            raise ValueError("GENERATION_MODEL must be a nonempty model name.")
        if self.rag_max_context_characters < self.rag_max_chunk_characters:
            raise ValueError("RAG_MAX_CONTEXT_CHARACTERS must cover one configured chunk.")
        if self.chunk_overlap > self.chunk_size // 2:
            raise ValueError("CHUNK_OVERLAP must not exceed half of CHUNK_SIZE.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
