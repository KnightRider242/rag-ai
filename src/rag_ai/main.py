"""FastAPI application entry point."""

from fastapi import FastAPI

from rag_ai.api.health import router as health_router
from rag_ai.api.documents import index_router, router as documents_router
from rag_ai.api.retrieval import router as retrieval_router
from rag_ai.api.answers import router as answer_router
from rag_ai.api.visual import router as visual_router
from rag_ai.api.vision import pdf_router as vision_pdf_router, image_router as vision_image_router
from rag_ai.api.visual_elements import router as visual_elements_router
from rag_ai.api.interpretations import router as interpretations_router
from rag_ai.api.visual_retrieval import router as visual_retrieval_router
from rag_ai.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the HTTP application."""
    if settings is None:
        settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.state.settings = settings
    app.include_router(health_router)
    # Route registration is a local-development guard, not caller or
    # collection authorization; the server must also bind to loopback.
    if settings.app_env == "development" and settings.enable_document_parse_endpoint:
        app.include_router(documents_router)
        if settings.enable_document_index_endpoint:
            app.include_router(index_router)
    if settings.app_env == "development" and settings.enable_document_visual_inspection_endpoint:
        app.include_router(visual_router)
    if settings.app_env == "development" and settings.enable_document_visual_analysis_endpoint:
        app.include_router(vision_pdf_router)
        app.include_router(vision_image_router)
    if settings.app_env == "development" and settings.enable_visual_element_storage_endpoint:
        app.include_router(visual_elements_router)
    if settings.app_env == "development" and settings.enable_visual_interpretation_endpoint:
        app.include_router(interpretations_router)
    if settings.app_env == "development" and settings.enable_visual_retrieval_endpoint:
        app.include_router(visual_retrieval_router)
    if settings.app_env == "development" and settings.enable_document_retrieval_endpoint:
        app.include_router(retrieval_router)
    if settings.app_env == "development" and settings.enable_rag_answer_endpoint:
        app.include_router(answer_router)
    return app


app = create_app()
