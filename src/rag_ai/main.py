"""FastAPI application entry point."""

from fastapi import FastAPI

from rag_ai.api.health import router as health_router
from rag_ai.core.config import get_settings


def create_app() -> FastAPI:
    """Create the HTTP application."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(health_router)
    return app


app = create_app()
