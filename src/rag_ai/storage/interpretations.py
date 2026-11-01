"""Collection-scoped immutable local visual interpretations."""

from datetime import datetime
from hashlib import sha256
import json
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_ai.documents.visual_elements import StoredVisualElement
from rag_ai.storage.migrate import MigrationError, require_current_schema
from rag_ai.storage.visual import VisualStoreError
from rag_ai.vision.base import VisionOutput

_ELEMENT_COLUMNS = ("collection_id, document_id, element_id, source_sha256, page_number, "
                    "element_type, occurrence_index, extractor_name, extractor_version, "
                    "coordinate_system, page_width, page_height, page_rotation, x0, y0, x1, y1, "
                    "extraction_metadata, created_at, updated_at")
_INTERPRETATION_COLUMNS = ("collection_id, document_id, element_id, interpretation_id, request_key, "
                           "analysis_task, prompt_version, model_name, model_digest, render_scale, "
                           "rendered_image_sha256, description, observations, limitations, "
                           "evidence_kind, verification_status, created_at")


class InterpretationNotFound(ValueError):
    """No interpretation or element exists within the requested collection."""


class InterpretationConflict(ValueError):
    """The source changed or a request key was reused for another task."""


class StoredInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: UUID
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    element_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    interpretation_id: UUID
    request_key: UUID | None
    analysis_task: str
    prompt_version: str
    model_name: str
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    render_scale: float = Field(ge=0.5, le=2)
    rendered_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    description: str = Field(min_length=1, max_length=1200)
    observations: list[str] = Field(max_length=8)
    limitations: list[str] = Field(max_length=5)
    evidence_kind: str
    verification_status: str
    created_at: datetime


class PostgresInterpretationStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def load_element(self, *, collection_id: UUID, document_id: str,
                     element_id: str) -> StoredVisualElement:
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                require_current_schema(connection)
                connection.row_factory = dict_row
                row = connection.execute(
                    f"SELECT {_ELEMENT_COLUMNS} FROM visual_elements WHERE collection_id = %s "
                    "AND document_id = %s AND element_id = %s",
                    (collection_id, document_id, element_id),
                ).fetchone()
                if row is None:
                    raise InterpretationNotFound("Visual element was not found.")
                return StoredVisualElement.model_validate(row)
        except (psycopg.Error, MigrationError, ValidationError, ValueError) as exc:
            if isinstance(exc, InterpretationNotFound):
                raise
            raise VisualStoreError("Visual interpretation storage is unavailable.") from exc

    def get(self, *, collection_id: UUID, document_id: str, element_id: str,
            interpretation_id: UUID | None = None, request_key: UUID | None = None,
            ) -> StoredInterpretation | None:
        if (interpretation_id is None) == (request_key is None):
            raise ValueError("Specify exactly one interpretation identity.")
        field = "interpretation_id" if interpretation_id is not None else "request_key"
        identity = interpretation_id if interpretation_id is not None else request_key
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                require_current_schema(connection)
                connection.row_factory = dict_row
                row = connection.execute(
                    f"SELECT {_INTERPRETATION_COLUMNS} FROM visual_interpretations "
                    f"WHERE collection_id = %s AND document_id = %s AND element_id = %s AND {field} = %s",
                    (collection_id, document_id, element_id, identity),
                ).fetchone()
                return StoredInterpretation.model_validate(row) if row else None
        except (psycopg.Error, MigrationError, ValidationError) as exc:
            raise VisualStoreError("Visual interpretation storage is unavailable.") from exc

    def list(self, *, collection_id: UUID, document_id: str, element_id: str,
             limit: int = 50, offset: int = 0) -> list[StoredInterpretation]:
        if not 1 <= limit <= 50 or offset < 0:
            raise ValueError("Invalid interpretation listing range.")
        self.load_element(collection_id=collection_id, document_id=document_id, element_id=element_id)
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                require_current_schema(connection)
                connection.row_factory = dict_row
                rows = connection.execute(
                    f"SELECT {_INTERPRETATION_COLUMNS} FROM visual_interpretations "
                    "WHERE collection_id = %s AND document_id = %s AND element_id = %s "
                    "ORDER BY created_at DESC, interpretation_id LIMIT %s OFFSET %s",
                    (collection_id, document_id, element_id, limit, offset),
                ).fetchall()
                return [StoredInterpretation.model_validate(row) for row in rows]
        except (psycopg.Error, MigrationError, ValidationError) as exc:
            raise VisualStoreError("Visual interpretation storage is unavailable.") from exc

    def insert(self, *, element: StoredVisualElement, source_sha256: str, page_count: int,
               request_key: UUID | None, task: str, prompt_version: str, model_name: str,
               model_digest: str, render_scale: float, rendered_image_sha256: str,
               output: VisionOutput) -> StoredInterpretation:
        """Lock parent and element, reject changed state, and insert exactly once."""
        key = (element.collection_id, element.document_id, element.element_id)
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    require_current_schema(connection)
                    connection.row_factory = dict_row
                    parent = connection.execute(
                        "SELECT original_content, file_type, page_count FROM indexed_documents "
                        "WHERE collection_id = %s AND document_id = %s FOR UPDATE",
                        key[:2],
                    ).fetchone()
                    if (parent is None or parent["file_type"] != "pdf" or parent["page_count"] != page_count
                            or sha256(parent["original_content"]).hexdigest() != source_sha256):
                        raise InterpretationConflict("Indexed source changed during inference.")
                    row = connection.execute(
                        f"SELECT {_ELEMENT_COLUMNS} FROM visual_elements WHERE collection_id = %s "
                        "AND document_id = %s AND element_id = %s FOR UPDATE", key,
                    ).fetchone()
                    if row is None or StoredVisualElement.model_validate(row) != element:
                        raise InterpretationConflict("Visual element changed during inference.")
                    if request_key is not None:
                        existing = connection.execute(
                            f"SELECT {_INTERPRETATION_COLUMNS} FROM visual_interpretations WHERE "
                            "collection_id = %s AND document_id = %s AND element_id = %s AND request_key = %s",
                            (*key, request_key),
                        ).fetchone()
                        if existing is not None:
                            result = StoredInterpretation.model_validate(existing)
                            if result.analysis_task != task or result.prompt_version != prompt_version:
                                raise InterpretationConflict("Request key was used for a different task.")
                            return result
                    interpretation_id = uuid4()
                    row = connection.execute(
                        "INSERT INTO visual_interpretations (collection_id, document_id, element_id, "
                        "interpretation_id, request_key, analysis_task, prompt_version, model_name, "
                        "model_digest, render_scale, rendered_image_sha256, description, observations, limitations) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb) "
                        f"RETURNING {_INTERPRETATION_COLUMNS}",
                        (*key, interpretation_id, request_key, task, prompt_version, model_name,
                         model_digest, render_scale, rendered_image_sha256, output.description,
                         json.dumps(output.observations, ensure_ascii=False),
                         json.dumps(output.limitations, ensure_ascii=False)),
                    ).fetchone()
                    return StoredInterpretation.model_validate(row)
        except (psycopg.Error, MigrationError, ValidationError) as exc:
            raise VisualStoreError("Visual interpretation storage is unavailable.") from exc
