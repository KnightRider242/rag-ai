"""FastAPI application entry point."""

from fastapi import FastAPI

from rag_ai.api.health import router as health_router
from rag_ai.api.documents import index_router, router as documents_router
from rag_ai.api.retrieval import router as retrieval_router
from rag_ai.api.answers import router as answer_router
from rag_ai.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the HTTP application."""
    if settings is None:
        settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.state.settings = settings
    app.include_router(health_router)
    if settings.app_env == "development" and settings.enable_document_parse_endpoint:
        app.include_router(documents_router)
        if settings.enable_document_index_endpoint:
            app.include_router(index_router)
    if settings.app_env == "development" and settings.enable_document_retrieval_endpoint:
        app.include_router(retrieval_router)
    if settings.app_env == "development" and settings.enable_rag_answer_endpoint:
        app.include_router(answer_router)
    return app


app = create_app()
