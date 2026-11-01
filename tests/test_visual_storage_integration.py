"""Phase 9A.1 checks in an opt-in disposable PostgreSQL schema only."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from hashlib import sha256
from importlib.resources import files
from threading import Event
from time import monotonic
from uuid import uuid4

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from rag_ai.core.config import Settings
from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.parser import parse_document
from rag_ai.documents.visual_elements import VisualElementCandidate, element_identity, extract_visual_elements
from rag_ai.embeddings.base import EmbeddingBatch
from rag_ai.storage import migrate as migration_module
from rag_ai.storage.migrate import MigrationError, MigrationRunner
from rag_ai.storage.postgres import PostgresIndexStore
from rag_ai.storage.visual import (
    PostgresVisualStore, VisualDocumentNotFound, VisualSourceInvalid, VisualStoreError,
)
from rag_ai.main import create_app


def _source() -> tuple[object, bytes, object]:
    pdf = pymupdf.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_text((30, 30), "Synthetic stored source")
    page.draw_rect(pymupdf.Rect(50, 60, 90, 100))
    content = pdf.tobytes()
    pdf.close()
    document = parse_document(filename="synthetic.pdf", content_type="application/pdf",
                              content=content, max_size_bytes=100_000, max_pdf_pages=5,
                              max_extracted_characters=100_000)
    return document, content, chunk_document(document)


def _save(dsn: str, collection_id, source: tuple[object, bytes, object]) -> None:
    document, content, chunked = source
    PostgresIndexStore(dsn).save(collection_id=collection_id, document=document,
                                 original_content=content, chunked=chunked,
                                 first_batch=EmbeddingBatch(model_name="synthetic-model",
                                                            model_revision="synthetic-revision",
                                                            dimensions=2,
                                                            vectors=[[0.1, 0.2] for _ in chunked.chunks]))


def test_upgrade_from_version_1_preserves_text_and_vectors(isolated_database_url: str) -> None:
    script = files("rag_ai.storage.migrations").joinpath("0001_initial.sql").read_text(encoding="utf-8")
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute(script)
        connection.execute("CREATE TABLE rag_ai_schema_migrations (version INTEGER PRIMARY KEY CHECK (version > 0), "
                           "checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        connection.execute("INSERT INTO rag_ai_schema_migrations(version, checksum) VALUES (1, %s)",
                           (sha256(script.encode()).hexdigest(),))
    runner = MigrationRunner(isolated_database_url)
    assert runner.status() == "version 1"
    with psycopg.connect(isolated_database_url) as connection:
        with pytest.raises(MigrationError, match="required"):
            from rag_ai.storage.migrate import require_current_schema
            require_current_schema(connection)
    source = _source()
    collection_id = uuid4()
    # Seed existing data directly because the new application rejects stale schemas.
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute("INSERT INTO collections(id) VALUES (%s)", (collection_id,))
        connection.execute("INSERT INTO indexed_documents(collection_id, document_id, filename, file_type, "
                           "page_count, original_content) VALUES (%s, %s, %s, 'pdf', 1, %s)",
                           (collection_id, source[0].document_id, "synthetic.pdf", source[1]))
        connection.execute("INSERT INTO embedding_spaces(collection_id, model_name, model_revision, dimensions) "
                           "VALUES (%s, 'synthetic-model', 'synthetic-revision', 2)", (collection_id,))
        chunk = source[2].chunks[0]
        connection.execute("INSERT INTO indexed_chunks(collection_id, chunk_id, document_id, chunk_index, "
                           "page_number, content, chunk_size, chunk_overlap) "
                           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                           (collection_id, chunk.chunk_id, source[0].document_id, chunk.chunk_index,
                            chunk.page_number, chunk.text, source[2].chunk_size, source[2].chunk_overlap))
        connection.execute("INSERT INTO chunk_embeddings(collection_id, chunk_id, model_name, dimensions, embedding) "
                           "VALUES (%s, %s, 'synthetic-model', 2, '[0.1,0.2]'::vector)",
                           (collection_id, chunk.chunk_id))
    assert runner.upgrade() == 3
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT original_content FROM indexed_documents WHERE collection_id = %s",
                                  (collection_id,)).fetchone() == (source[1],)
        assert connection.execute("SELECT content FROM indexed_chunks WHERE collection_id = %s",
                                  (collection_id,)).fetchone() == (chunk.text,)
        assert connection.execute("SELECT vector_dims(embedding) FROM chunk_embeddings "
                                  "WHERE collection_id = %s", (collection_id,)).fetchone() == (2,)


def test_failed_version_2_upgrade_rolls_back_schema_and_history(
    isolated_database_url: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = migration_module._migrations()[0]
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute(first.script)
        connection.execute("CREATE TABLE rag_ai_schema_migrations (version INTEGER PRIMARY KEY CHECK (version > 0), "
                           "checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        connection.execute("INSERT INTO rag_ai_schema_migrations(version, checksum) VALUES (1, %s)",
                           (first.checksum,))
    bad_script = "CREATE TABLE visual_elements (id INTEGER); SELECT 1 / 0"
    failed_second = migration_module.Migration(2, bad_script, sha256(bad_script.encode()).hexdigest())
    with monkeypatch.context() as patch:
        patch.setattr(migration_module, "_migrations", lambda: (first, failed_second))
        with pytest.raises(MigrationError, match="migration failed"):
            MigrationRunner(isolated_database_url).upgrade()
    assert MigrationRunner(isolated_database_url).status() == "version 1"
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT version, checksum FROM rag_ai_schema_migrations").fetchall() == [
            (1, first.checksum),
        ]
        assert connection.execute(
            "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname = current_schema() "
            "AND tablename = 'visual_elements'"
        ).fetchone() is None


def test_atomic_reindex_collection_scope_and_cascade(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    first, second = uuid4(), uuid4()
    _save(isolated_database_url, first, source)
    _save(isolated_database_url, second, source)
    store = PostgresVisualStore(isolated_database_url)
    candidates = extract_visual_elements(filename="synthetic.pdf", content=source[1],
                                         max_size_bytes=100_000, max_pdf_pages=5)
    assert candidates
    for collection in (first, second):
        filename, content, pages = store.load_pdf(collection_id=collection,
                                                  document_id=source[0].document_id,
                                                  max_size_bytes=100_000)
        assert (filename, content, pages) == ("synthetic.pdf", source[1], 1)
        store.replace(collection_id=collection, document_id=source[0].document_id,
                      source_sha256=source[0].document_id, page_count=1, candidates=candidates)
    original = store.list_elements(collection_id=first, document_id=source[0].document_id)
    assert len(original) == len(candidates)
    store.replace(collection_id=first, document_id=source[0].document_id,
                  source_sha256=source[0].document_id, page_count=1, candidates=candidates)
    repeated = store.list_elements(collection_id=first, document_id=source[0].document_id)
    assert [row.element_id for row in original] == [row.element_id for row in repeated]
    assert [row.created_at for row in original] == [row.created_at for row in repeated]
    _save(isolated_database_url, first, source)
    assert len(store.list_elements(collection_id=first, document_id=source[0].document_id)) == len(candidates)
    bad = candidates[0].model_copy(update={"x0": candidates[0].x0 + Decimal("1")})
    bad = bad.model_copy(update={"element_id": element_identity(bad)})
    with pytest.raises(VisualStoreError):
        store.replace(collection_id=first, document_id=source[0].document_id,
                      source_sha256=source[0].document_id, page_count=1, candidates=[candidates[0], bad])
    assert len(store.list_elements(collection_id=first, document_id=source[0].document_id)) == len(candidates)
    shifted = candidates[0].model_copy(update={"x0": candidates[0].x0 + Decimal("1")})
    shifted = shifted.model_copy(update={"element_id": element_identity(shifted)})
    store.replace(collection_id=first, document_id=source[0].document_id,
                  source_sha256=source[0].document_id, page_count=1, candidates=[shifted])
    assert [row.element_id for row in store.list_elements(
        collection_id=first, document_id=source[0].document_id)] == [shifted.element_id]
    newer = extract_visual_elements(filename="synthetic.pdf", content=source[1],
                                    max_size_bytes=100_000, max_pdf_pages=5, extractor_version="2")
    store.replace(collection_id=first, document_id=source[0].document_id,
                  source_sha256=source[0].document_id, page_count=1,
                  candidates=newer, extractor_version="2")
    assert {row.extractor_version for row in store.list_elements(
        collection_id=first, document_id=source[0].document_id)} == {"2"}
    with pytest.raises(VisualDocumentNotFound):
        store.load_pdf(collection_id=uuid4(), document_id=source[0].document_id,
                       max_size_bytes=100_000)
    PostgresIndexStore(isolated_database_url).delete_document(collection_id=first,
                                                               document_id=source[0].document_id)
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM visual_elements WHERE collection_id = %s",
                                  (first,)).fetchone() == (0,)
    assert len(store.list_elements(collection_id=second, document_id=source[0].document_id)) == len(candidates)


def test_parallel_replacement_serializes_on_document(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    collection_id = uuid4()
    _save(isolated_database_url, collection_id, source)
    store = PostgresVisualStore(isolated_database_url)
    candidates = extract_visual_elements(filename="synthetic.pdf", content=source[1],
                                         max_size_bytes=100_000, max_pdf_pages=5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(store.replace, collection_id=collection_id,
                               document_id=source[0].document_id,
                               source_sha256=source[0].document_id,
                               page_count=1, candidates=candidates) for _ in range(2)]
        for future in futures:
            assert future.result(timeout=20).element_count == len(candidates)
    assert len(store.list_elements(collection_id=collection_id,
                                   document_id=source[0].document_id)) == len(candidates)


def test_contended_visual_replacements_publish_one_complete_set(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    collection_id = uuid4()
    _save(isolated_database_url, collection_id, source)
    candidate = extract_visual_elements(filename="synthetic.pdf", content=source[1],
                                        max_size_bytes=100_000, max_pdf_pages=5)[0]

    def pair(first_shift: int) -> list[VisualElementCandidate]:
        result: list[VisualElementCandidate] = []
        for occurrence_index, shift in enumerate((first_shift, first_shift + 1)):
            changed = candidate.model_copy(update={
                "occurrence_index": occurrence_index,
                "x0": candidate.x0 + Decimal(shift),
                "x1": candidate.x1 + Decimal(shift),
            })
            result.append(changed.model_copy(update={"element_id": element_identity(changed)}))
        return result

    initial, first_set, second_set = pair(0), pair(2), pair(4)
    store = PostgresVisualStore(isolated_database_url)
    store.replace(collection_id=collection_id, document_id=source[0].document_id,
                  source_sha256=source[0].document_id, page_count=1, candidates=initial)
    names = [f"visual_lock_test_{uuid4().hex}" for _ in range(2)]
    base_options = conninfo_to_dict(isolated_database_url)["options"]
    worker_stores = [PostgresVisualStore(make_conninfo(
        isolated_database_url, application_name=name,
        options=f"{base_options} -c lock_timeout=8s",
    )) for name in names]

    with ThreadPoolExecutor(max_workers=2) as pool:
        with psycopg.connect(isolated_database_url) as holder:
            with holder.transaction():
                holder_pid = holder.execute("SELECT pg_backend_pid()").fetchone()[0]
                holder.execute(
                    "SELECT 1 FROM indexed_documents WHERE collection_id = %s "
                    "AND document_id = %s FOR UPDATE",
                    (collection_id, source[0].document_id),
                )
                futures = [pool.submit(worker.replace, collection_id=collection_id,
                                       document_id=source[0].document_id,
                                       source_sha256=source[0].document_id,
                                       page_count=1, candidates=items)
                           for worker, items in zip(worker_stores, (first_set, second_set), strict=True)]
                with psycopg.connect(isolated_database_url, autocommit=True) as observer:
                    deadline = monotonic() + 5
                    blocked: set[str] = set()
                    holder_blocks_queue = False
                    while monotonic() < deadline:
                        rows = observer.execute(
                            "SELECT application_name, wait_event_type, pg_blocking_pids(pid) "
                            "FROM pg_stat_activity WHERE datname = current_database() "
                            "AND application_name IN (%s, %s)",
                            names,
                        ).fetchall()
                        blocked = {name for name, wait, _ in rows if wait == "Lock"}
                        holder_blocks_queue = any(holder_pid in blockers for _, _, blockers in rows)
                        if blocked == set(names) and holder_blocks_queue:
                            break
                        Event().wait(0.05)
                    assert blocked == set(names) and holder_blocks_queue, (
                        "Both replacements must wait in the document lock queue"
                    )
                    assert {row.element_id for row in store.list_elements(
                        collection_id=collection_id, document_id=source[0].document_id,
                    )} == {item.element_id for item in initial}
        for future in futures:
            assert future.result(timeout=12).element_count == 2

    final_ids = {row.element_id for row in store.list_elements(
        collection_id=collection_id, document_id=source[0].document_id,
    )}
    assert final_ids in ({item.element_id for item in first_set},
                         {item.element_id for item in second_set})


def test_text_and_visual_reindex_share_document_lock(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    collection_id = uuid4()
    _save(isolated_database_url, collection_id, source)
    store = PostgresVisualStore(isolated_database_url)
    candidates = extract_visual_elements(filename="synthetic.pdf", content=source[1],
                                         max_size_bytes=100_000, max_pdf_pages=5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        visual = pool.submit(store.replace, collection_id=collection_id,
                             document_id=source[0].document_id, source_sha256=source[0].document_id,
                             page_count=1, candidates=candidates)
        text = pool.submit(_save, isolated_database_url, collection_id, source)
        assert visual.result(timeout=20).element_count == len(candidates)
        text.result(timeout=20)
    assert len(store.list_elements(collection_id=collection_id,
                                   document_id=source[0].document_id)) == len(candidates)


def test_source_changed_after_read_cannot_commit_visual_set(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    collection_id = uuid4()
    _save(isolated_database_url, collection_id, source)
    store = PostgresVisualStore(isolated_database_url)
    _, content, pages = store.load_pdf(collection_id=collection_id,
                                       document_id=source[0].document_id, max_size_bytes=100_000)
    candidates = extract_visual_elements(filename="synthetic.pdf", content=content,
                                         max_size_bytes=100_000, max_pdf_pages=5)
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute("UPDATE indexed_documents SET original_content = %s "
                           "WHERE collection_id = %s AND document_id = %s",
                           (b"corrupt synthetic bytes", collection_id, source[0].document_id))
    with pytest.raises(VisualSourceInvalid):
        store.replace(collection_id=collection_id, document_id=source[0].document_id,
                      source_sha256=source[0].document_id, page_count=pages, candidates=candidates)
    assert store.list_elements(collection_id=collection_id, document_id=source[0].document_id) == []


@pytest.mark.parametrize("coordinate", ["NaN", "Infinity", "-Infinity", "-1", "301"])
def test_postgresql_rejects_special_or_out_of_page_numeric_geometry(
    isolated_database_url: str, coordinate: str,
) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    collection_id = uuid4()
    _save(isolated_database_url, collection_id, source)
    with psycopg.connect(isolated_database_url) as connection:
        with pytest.raises(psycopg.Error):
            connection.execute(
                "INSERT INTO visual_elements (collection_id, document_id, element_id, source_sha256, "
                "page_number, element_type, occurrence_index, extractor_name, extractor_version, "
                "coordinate_system, page_width, page_height, page_rotation, x0, y0, x1, y1, "
                "extraction_metadata) VALUES (%s, %s, %s, %s, 1, 'vector_path', 0, "
                "'synthetic', '1', 'displayed_pdf_points', 300, 200, 0, %s::numeric, 1, 5, 5, '{}'::jsonb)",
                (collection_id, source[0].document_id, "a" * 64,
                 source[0].document_id, coordinate),
            )


def test_database_rejects_cross_collection_or_unknown_document_reference(
    isolated_database_url: str,
) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    _save(isolated_database_url, uuid4(), source)
    with psycopg.connect(isolated_database_url) as connection:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(
                "INSERT INTO visual_elements (collection_id, document_id, element_id, source_sha256, "
                "page_number, element_type, occurrence_index, extractor_name, extractor_version, "
                "coordinate_system, page_width, page_height, page_rotation, x0, y0, x1, y1, "
                "extraction_metadata) VALUES (%s, %s, %s, %s, 1, 'vector_path', 0, "
                "'synthetic', '1', 'displayed_pdf_points', 300, 200, 0, 1, 1, 5, 5, '{}'::jsonb)",
                (uuid4(), source[0].document_id, "a" * 64, source[0].document_id),
            )


def test_development_api_indexes_and_lists_metadata_only(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    source = _source()
    collection_id = uuid4()
    _save(isolated_database_url, collection_id, source)
    path = f"/collections/{collection_id}/documents/{source[0].document_id}/visual-elements"
    app = create_app(Settings(app_env="development", enable_visual_element_storage_endpoint=True,
                              database_url=SecretStr(isolated_database_url), _env_file=None))
    with TestClient(app) as client:
        indexed = client.post(path + "/index")
        assert indexed.status_code == 200
        assert indexed.json()["element_count"] > 0
        listed = client.get(path, params={"limit": 1})
        assert listed.status_code == 200 and len(listed.json()) == 1
        assert "original_content" not in listed.text
        assert "Synthetic stored source" not in listed.text
        assert client.get(path, params={"limit": 101}).status_code == 422
        missing = client.get(f"/collections/{uuid4()}/documents/{source[0].document_id}/visual-elements")
        assert missing.status_code == 404
