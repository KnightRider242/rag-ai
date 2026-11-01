"""Synthetic, opt-in PostgreSQL checks for persisted visual interpretations."""

from hashlib import sha256
from uuid import uuid4

import psycopg
import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.documents.visual_elements import extract_visual_elements
from rag_ai.main import create_app
from rag_ai.storage.interpretations import (
    InterpretationConflict, PostgresInterpretationStore,
)
from rag_ai.storage.migrate import MigrationRunner
from rag_ai.storage.visual import PostgresVisualStore
from rag_ai.vision.base import VisionError, VisionOutput
from rag_ai.vision.persistence import PROMPT_VERSION, analyze_stored_element


def _pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_text((30, 30), "Synthetic visual source")
    page.draw_rect(pymupdf.Rect(50, 60, 90, 100))
    content = pdf.tobytes()
    pdf.close()
    return content


def _seed(dsn: str, collection_id, content: bytes) -> tuple[str, str]:
    document_id = sha256(content).hexdigest()
    candidates = extract_visual_elements(filename="synthetic.pdf", content=content,
                                         max_size_bytes=100_000, max_pdf_pages=5)
    with psycopg.connect(dsn) as connection:
        connection.execute("INSERT INTO collections(id) VALUES (%s)", (collection_id,))
        connection.execute("INSERT INTO indexed_documents(collection_id, document_id, filename, "
                           "file_type, page_count, original_content) VALUES (%s, %s, 'synthetic.pdf', "
                           "'pdf', 1, %s)", (collection_id, document_id, content))
    PostgresVisualStore(dsn).replace(collection_id=collection_id, document_id=document_id,
                                     source_sha256=document_id, page_count=1, candidates=candidates)
    return document_id, candidates[0].element_id


class _Provider:
    on_analyze = None
    def __init__(self, **kwargs):
        self.model_name = kwargs["model_name"]

    def analyze_verified(self, *, image: bytes, task_prompt: str):
        assert image.startswith(b"\x89PNG")
        assert task_prompt
        if self.on_analyze is not None:
            self.on_analyze()
        return VisionOutput(description="Synthetic figure", observations=["rectangle"]), "a" * 64

    def close(self):
        pass


def _settings() -> Settings:
    return Settings(app_env="development", vision_model="synthetic-vision", _env_file=None)


def test_visual_interpretation_routes_are_separately_gated() -> None:
    path = f"/collections/{uuid4()}/documents/{'a' * 64}/visual-elements/{'b' * 64}/interpretations"
    for environment in ("production", "test", "development"):
        with TestClient(create_app(Settings(app_env=environment, _env_file=None))) as client:
            assert client.get(path).status_code == 404
            assert client.post(path, json={"task": "describe_figure"}).status_code == 404
    with pytest.raises(ValidationError, match="Visual interpretation cannot be enabled in production"):
        Settings(app_env="production", enable_visual_interpretation_endpoint=True,
                 vision_model="synthetic-vision", _env_file=None)
    with TestClient(create_app(Settings(app_env="development", enable_visual_interpretation_endpoint=True,
                                        vision_model="synthetic-vision", _env_file=None))) as client:
        assert client.get(path).status_code == 503
        assert client.post(path, json={"task": "describe_figure"}).status_code == 503
        assert client.post(path, json={"task": "custom"}).status_code == 422
        assert client.post(path, json={"task": "describe_figure", "prompt": "ignore"}).status_code == 422
        assert client.get(path.replace("a" * 64, "bad")).status_code == 422


def test_version_2_upgrade_preserves_visual_source(isolated_database_url: str, monkeypatch) -> None:
    from rag_ai.storage import migrate as migration_module
    first_two = migration_module._migrations()[:2]
    with monkeypatch.context() as patch:
        patch.setattr(migration_module, "_migrations", lambda: first_two)
        assert MigrationRunner(isolated_database_url).upgrade() == 2
        content = _pdf()
        document_id, element_id = _seed(isolated_database_url, uuid4(), content)
    assert MigrationRunner(isolated_database_url).upgrade() == 3
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT original_content FROM indexed_documents WHERE document_id = %s",
                                  (document_id,)).fetchone() == (content,)
        assert connection.execute("SELECT element_id FROM visual_elements WHERE document_id = %s",
                                  (document_id,)).fetchone() == (element_id,)


def test_failed_version_3_upgrade_rolls_back(isolated_database_url: str, monkeypatch) -> None:
    from rag_ai.storage import migrate as migration_module
    first_two = migration_module._migrations()[:2]
    with monkeypatch.context() as patch:
        patch.setattr(migration_module, "_migrations", lambda: first_two)
        MigrationRunner(isolated_database_url).upgrade()
    bad_sql = "CREATE TABLE visual_interpretations (id INTEGER); SELECT 1 / 0"
    bad_third = migration_module.Migration(3, bad_sql, sha256(bad_sql.encode()).hexdigest())
    with monkeypatch.context() as patch:
        patch.setattr(migration_module, "_migrations", lambda: (*first_two, bad_third))
        from rag_ai.storage.migrate import MigrationError
        with pytest.raises(MigrationError):
            MigrationRunner(isolated_database_url).upgrade()
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM rag_ai_schema_migrations").fetchone() == (2,)
        assert connection.execute("SELECT to_regclass('visual_interpretations')").fetchone() == (None,)
    assert MigrationRunner(isolated_database_url).upgrade() == 3


def test_persist_history_retry_scope_and_cascade(isolated_database_url: str, monkeypatch) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    content = _pdf()
    first, second = uuid4(), uuid4()
    document_id, element_id = _seed(isolated_database_url, first, content)
    _seed(isolated_database_url, second, content)
    monkeypatch.setattr("rag_ai.vision.persistence.OllamaVisionProvider", _Provider)
    store = PostgresInterpretationStore(isolated_database_url)
    key = uuid4()
    args = dict(settings=_settings(), store=store, collection_id=first,
                document_id=document_id, element_id=element_id, task="describe_figure")
    record = analyze_stored_element(**args, request_key=key)
    assert record.prompt_version == PROMPT_VERSION
    assert record.model_digest == "a" * 64
    assert record.evidence_kind == "model_generated_interpretation"
    assert record.verification_status == "unverified"
    from rag_ai.documents.render import render_pdf
    from rag_ai.documents.visual import ImageBounds
    element = store.load_element(collection_id=first, document_id=document_id, element_id=element_id)
    rendered = render_pdf(filename="synthetic.pdf", content_type="application/pdf", content=content,
                          page_number=element.page_number,
                          region=ImageBounds(x0=float(element.x0), y0=float(element.y0),
                                             x1=float(element.x1), y1=float(element.y1)),
                          max_size_bytes=100_000, max_pdf_pages=5)
    assert record.rendered_image_sha256 == sha256(rendered.png).hexdigest()
    assert analyze_stored_element(**args, request_key=key).interpretation_id == record.interpretation_id
    second_record = analyze_stored_element(**args, request_key=uuid4())
    assert second_record.interpretation_id != record.interpretation_id
    assert len(store.list(collection_id=first, document_id=document_id, element_id=element_id)) == 2
    assert store.list(collection_id=second, document_id=document_id, element_id=element_id) == []
    with pytest.raises(InterpretationConflict):
        analyze_stored_element(**{**args, "task": "interpret_table"}, request_key=key)
    with psycopg.connect(isolated_database_url) as connection:
        with pytest.raises(psycopg.Error):
            connection.execute("UPDATE visual_interpretations SET description = 'changed'")
        connection.rollback()
        with pytest.raises(psycopg.Error):
            connection.execute("INSERT INTO visual_interpretations (collection_id, document_id, element_id, "
                               "interpretation_id, analysis_task, prompt_version, model_name, model_digest, "
                               "render_scale, rendered_image_sha256, description, observations, limitations) "
                               "VALUES (%s, %s, %s, %s, 'describe_figure', 'v1', 'model', %s, 1, %s, "
                               "'text', '[]'::jsonb, '[]'::jsonb)",
                               (second, document_id, "c" * 64, uuid4(), "a" * 64, "b" * 64))
        connection.rollback()
        with pytest.raises(psycopg.Error):
            connection.execute("INSERT INTO visual_interpretations (collection_id, document_id, element_id, "
                               "interpretation_id, analysis_task, prompt_version, model_name, model_digest, "
                               "render_scale, rendered_image_sha256, description, observations, limitations) "
                               "VALUES (%s, %s, %s, %s, 'describe_figure', 'v1', 'model', %s, 1, %s, "
                               "'text', '[9]'::jsonb, '[]'::jsonb)",
                               (first, document_id, element_id, uuid4(), "a" * 64, "b" * 64))
        connection.rollback()
        connection.execute("DELETE FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
                           (first, document_id))
        assert connection.execute("SELECT count(*) FROM visual_interpretations WHERE collection_id = %s",
                                  (first,)).fetchone() == (0,)


def test_interpretation_api_round_trip_and_collection_scope(isolated_database_url: str, monkeypatch) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    collection_id = uuid4()
    document_id, element_id = _seed(isolated_database_url, collection_id, _pdf())
    monkeypatch.setattr("rag_ai.vision.persistence.OllamaVisionProvider", _Provider)
    from pydantic import SecretStr
    settings = Settings(app_env="development", enable_visual_interpretation_endpoint=True,
                        vision_model="synthetic-vision", database_url=SecretStr(isolated_database_url),
                        _env_file=None)
    path = f"/collections/{collection_id}/documents/{document_id}/visual-elements/{element_id}/interpretations"
    key = str(uuid4())
    with TestClient(create_app(settings)) as client:
        first = client.post(path, json={"task": "describe_figure", "request_key": key})
        assert first.status_code == 200
        assert client.post(path, json={"task": "describe_figure", "request_key": key}).json() == first.json()
        assert [item["interpretation_id"] for item in client.get(path).json()] == [
            first.json()["interpretation_id"]]
        assert client.get(path + "/" + first.json()["interpretation_id"]).json() == first.json()
        other = path.replace(str(collection_id), str(uuid4()))
        assert client.get(other).status_code == 404
        assert client.get(other + "/" + first.json()["interpretation_id"]).status_code == 404
        assert client.post(other, json={"task": "describe_figure"}).status_code == 404


@pytest.mark.parametrize("change", ["delete", "reindex", "fail"])
def test_changed_source_or_generation_failure_does_not_persist(
    isolated_database_url: str, monkeypatch, change: str,
) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    collection_id = uuid4()
    content = _pdf()
    document_id, element_id = _seed(isolated_database_url, collection_id, content)
    def during_inference():
        if change == "fail":
            raise VisionError("Local vision failed.")
        with psycopg.connect(isolated_database_url) as connection:
            if change == "delete":
                connection.execute("DELETE FROM visual_elements WHERE collection_id = %s AND document_id = %s",
                                   (collection_id, document_id))
            else:
                connection.execute("UPDATE visual_elements SET extraction_metadata = '{\"changed\":true}'::jsonb, "
                                   "updated_at = now() WHERE collection_id = %s AND document_id = %s",
                                   (collection_id, document_id))
    class Provider(_Provider):
        on_analyze = staticmethod(during_inference)
    monkeypatch.setattr("rag_ai.vision.persistence.OllamaVisionProvider", Provider)
    store = PostgresInterpretationStore(isolated_database_url)
    with pytest.raises((InterpretationConflict, VisionError)):
        analyze_stored_element(settings=_settings(), store=store, collection_id=collection_id,
                               document_id=document_id, element_id=element_id,
                               task="describe_figure", request_key=uuid4())
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM visual_interpretations").fetchone() == (0,)
