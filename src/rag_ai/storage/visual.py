"""Collection-scoped, atomic persistence of visual source occurrences."""

from hashlib import sha256
import json
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from pydantic import ValidationError

from rag_ai.documents.visual_elements import (
    EXTRACTOR_NAME, EXTRACTOR_VERSION, MAX_VISUAL_ELEMENTS,
    StoredVisualElement, VisualElementCandidate, VisualIndexResult, element_identity,
)
from rag_ai.storage.migrate import MigrationError, require_current_schema


class VisualStoreError(RuntimeError):
    """Database operation failed; suppress SQL and source content in public errors."""


class VisualDocumentNotFound(ValueError):
    """The collection has no document with this identity."""


class VisualSourceInvalid(ValueError):
    """The stored source cannot safely be indexed."""


class PostgresVisualStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def load_pdf(self, *, collection_id: UUID, document_id: str,
                 max_size_bytes: int) -> tuple[str, bytes, int]:
        """Read only one indexed PDF and verify its content-derived identity."""
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                require_current_schema(connection)
                row = connection.execute(
                    "SELECT filename, file_type, page_count, octet_length(original_content) "
                    "FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
                    (collection_id, document_id),
                ).fetchone()
                if row is None:
                    raise VisualDocumentNotFound("Indexed document was not found.")
                filename, file_type, page_count, length = row
                if file_type != "pdf" or page_count is None or page_count < 1 or length > max_size_bytes:
                    raise VisualSourceInvalid("Indexed source is not an eligible PDF.")
                content = connection.execute(
                    "SELECT original_content FROM indexed_documents "
                    "WHERE collection_id = %s AND document_id = %s",
                    (collection_id, document_id),
                ).fetchone()
                if content is None or sha256(content[0]).hexdigest() != document_id:
                    raise VisualSourceInvalid("Indexed source identity is invalid.")
                return filename, content[0], page_count
        except (psycopg.Error, MigrationError) as exc:
            raise VisualStoreError("Visual storage is unavailable.") from exc

    def replace(self, *, collection_id: UUID, document_id: str, source_sha256: str,
                page_count: int, candidates: list[VisualElementCandidate],
                extractor_version: str = EXTRACTOR_VERSION) -> VisualIndexResult:
        """Replace one extractor's complete set under the parent document lock."""
        if len(candidates) > MAX_VISUAL_ELEMENTS or len({item.element_id for item in candidates}) != len(candidates):
            raise ValueError("Duplicate or excessive visual elements.")
        if any(item.source_sha256 != source_sha256 or item.page_number > page_count
               or item.extractor_name != EXTRACTOR_NAME or item.extractor_version != extractor_version
               or item.element_id != element_identity(item)
               for item in candidates):
            raise ValueError("Visual elements do not match the indexed source or extractor.")
        if source_sha256 != document_id:
            raise ValueError("Visual source identity mismatch.")
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    require_current_schema(connection)
                    row = connection.execute(
                        "SELECT original_content, file_type, page_count FROM indexed_documents "
                        "WHERE collection_id = %s AND document_id = %s FOR UPDATE",
                        (collection_id, document_id),
                    ).fetchone()
                    if row is None:
                        raise VisualDocumentNotFound("Indexed document was not found.")
                    if row[1] != "pdf" or row[2] != page_count or sha256(row[0]).hexdigest() != source_sha256:
                        raise VisualSourceInvalid("Indexed source changed during visual indexing.")
                    # Remove changed occurrence identities before inserting new
                    # ones; the complete replacement remains one transaction.
                    connection.execute(
                        "DELETE FROM visual_elements WHERE collection_id = %s AND document_id = %s "
                        "AND extractor_name = %s AND NOT (element_id = ANY(%s::text[]))",
                        (collection_id, document_id, EXTRACTOR_NAME,
                         [item.element_id for item in candidates]),
                    )
                    for item in candidates:
                        connection.execute(
                            "INSERT INTO visual_elements (collection_id, document_id, element_id, source_sha256, "
                            "page_number, element_type, occurrence_index, extractor_name, extractor_version, "
                            "coordinate_system, page_width, page_height, page_rotation, x0, y0, x1, y1, "
                            "extraction_metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                            "%s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                            "ON CONFLICT (collection_id, document_id, element_id) DO UPDATE SET "
                            "extraction_metadata = EXCLUDED.extraction_metadata, "
                            "updated_at = now() WHERE visual_elements.extraction_metadata "
                            "IS DISTINCT FROM EXCLUDED.extraction_metadata",
                            (collection_id, document_id, item.element_id, item.source_sha256,
                             item.page_number, item.element_type, item.occurrence_index,
                             item.extractor_name, item.extractor_version, item.coordinate_system,
                             item.page_width, item.page_height, item.page_rotation,
                             item.x0, item.y0, item.x1, item.y1,
                             json.dumps(item.extraction_metadata, ensure_ascii=False, allow_nan=False)),
                        )
            return VisualIndexResult(collection_id=collection_id, document_id=document_id,
                                     extractor_name=EXTRACTOR_NAME, extractor_version=extractor_version,
                                     element_count=len(candidates))
        except (psycopg.Error, MigrationError) as exc:
            raise VisualStoreError("Visual storage is unavailable.") from exc

    def list_elements(self, *, collection_id: UUID, document_id: str,
                      limit: int = 100, offset: int = 0) -> list[StoredVisualElement]:
        """Return only bounded metadata for one collection/document."""
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Invalid visual listing range.")
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                require_current_schema(connection)
                connection.row_factory = dict_row
                exists = connection.execute(
                    "SELECT 1 FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
                    (collection_id, document_id),
                ).fetchone()
                if exists is None:
                    raise VisualDocumentNotFound("Indexed document was not found.")
                rows = connection.execute(
                    "SELECT collection_id, document_id, element_id, source_sha256, page_number, "
                    "element_type, occurrence_index, extractor_name, extractor_version, "
                    "coordinate_system, page_width, page_height, page_rotation, x0, y0, x1, y1, "
                    "extraction_metadata, created_at, updated_at FROM visual_elements "
                    "WHERE collection_id = %s AND document_id = %s "
                    "ORDER BY page_number, element_type, occurrence_index, element_id LIMIT %s OFFSET %s",
                    (collection_id, document_id, limit, offset),
                ).fetchall()
                return [StoredVisualElement.model_validate(row) for row in rows]
        except (psycopg.Error, MigrationError, ValidationError) as exc:
            raise VisualStoreError("Visual storage is unavailable.") from exc
