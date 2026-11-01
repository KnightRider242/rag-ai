"""Collection-scoped pgvector persistence for visual interpretation text."""

import json
import math
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from rag_ai.embeddings.base import EmbeddingBatch
from rag_ai.retrieval.visual import RetrievedVisualInterpretation
from rag_ai.storage.interpretations import PostgresInterpretationStore, StoredInterpretation
from rag_ai.storage.migrate import MigrationError, require_current_schema
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch, StorageError,
)
from rag_ai.storage.visual import VisualStoreError


class VisualInterpretationNotFound(ValueError):
    """The requested interpretation is absent from the selected collection."""


class VisualIndexSourceChanged(ValueError):
    """The interpretation or its source changed before the embedding write."""


class PostgresVisualRetrievalStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def load_interpretation(self, *, collection_id: UUID, document_id: str,
                            element_id: str, interpretation_id: UUID) -> StoredInterpretation:
        try:
            result = PostgresInterpretationStore(self.dsn).get(
                collection_id=collection_id, document_id=document_id,
                element_id=element_id, interpretation_id=interpretation_id,
            )
        except VisualStoreError as exc:
            raise StorageError("Visual retrieval storage is unavailable.") from exc
        if result is None:
            raise VisualInterpretationNotFound("Visual interpretation was not found.")
        return result

    def index_interpretation(self, *, interpretation: StoredInterpretation, text_sha256: str,
                             text_format_version: str, batch: EmbeddingBatch) -> bool:
        """Recheck the source under locks and insert one compatible derived vector."""
        if (len(batch.vectors) != 1 or len(batch.vectors[0]) != batch.dimensions
                or any(not math.isfinite(value) for value in batch.vectors[0])
                or not math.isfinite(math.hypot(*batch.vectors[0]))
                or math.hypot(*batch.vectors[0]) == 0):
            raise ValueError("Invalid visual embedding vector.")
        key = (interpretation.collection_id, interpretation.document_id,
               interpretation.element_id, interpretation.interpretation_id)
        vector_json = json.dumps(batch.vectors[0], allow_nan=False)
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    require_current_schema(connection)
                    if connection.execute(
                        "SELECT 1 FROM indexed_documents WHERE collection_id = %s AND document_id = %s FOR UPDATE",
                        key[:2],
                    ).fetchone() is None:
                        raise VisualIndexSourceChanged("Visual source was deleted during indexing.")
                    if connection.execute(
                        "SELECT 1 FROM visual_elements WHERE collection_id = %s AND document_id = %s "
                        "AND element_id = %s FOR UPDATE", key[:3],
                    ).fetchone() is None:
                        raise VisualIndexSourceChanged("Visual element was deleted during indexing.")
                    connection.row_factory = dict_row
                    current = connection.execute(
                        "SELECT * FROM visual_interpretations WHERE collection_id = %s "
                        "AND document_id = %s AND element_id = %s AND interpretation_id = %s FOR UPDATE",
                        key,
                    ).fetchone()
                    if current is None or StoredInterpretation.model_validate(current) != interpretation:
                        raise VisualIndexSourceChanged("Visual interpretation changed during indexing.")
                    connection.execute(
                        "INSERT INTO embedding_spaces(collection_id, model_name, model_revision, dimensions) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (key[0], batch.model_name, batch.model_revision, batch.dimensions),
                    )
                    space = connection.execute(
                        "SELECT model_revision, dimensions FROM embedding_spaces "
                        "WHERE collection_id = %s AND model_name = %s FOR UPDATE",
                        (key[0], batch.model_name),
                    ).fetchone()
                    if space is None or space["dimensions"] != batch.dimensions:
                        raise ModelDimensionMismatch("Embedding dimensions differ from this collection's model.")
                    if space["model_revision"] != batch.model_revision:
                        raise ModelRevisionMismatch("Embedding revision differs from this collection's model.")
                    inserted = connection.execute(
                        "INSERT INTO visual_interpretation_embeddings "
                        "(collection_id, document_id, element_id, interpretation_id, model_name, "
                        "model_revision, dimensions, text_format_version, text_sha256, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector) "
                        "ON CONFLICT DO NOTHING RETURNING text_sha256",
                        (*key, batch.model_name, batch.model_revision, batch.dimensions,
                         text_format_version, text_sha256, vector_json),
                    ).fetchone()
                    if inserted is not None:
                        return False
                    existing = connection.execute(
                        "SELECT model_revision, dimensions, text_sha256 FROM visual_interpretation_embeddings "
                        "WHERE collection_id = %s AND document_id = %s AND element_id = %s "
                        "AND interpretation_id = %s AND model_name = %s AND text_format_version = %s",
                        (*key, batch.model_name, text_format_version),
                    ).fetchone()
                    if (existing is None or existing["model_revision"] != batch.model_revision
                            or existing["dimensions"] != batch.dimensions
                            or existing["text_sha256"] != text_sha256):
                        raise VisualIndexSourceChanged("Existing visual embedding has different provenance.")
                    return True
        except (psycopg.Error, MigrationError) as exc:
            raise StorageError("Visual embedding storage is unavailable.") from exc

    def search(self, *, collection_id: UUID, query_vector: list[float], model_name: str,
               model_revision: str, dimensions: int, top_k: int,
               text_format_version: str) -> list[RetrievedVisualInterpretation]:
        """Rank only compatible vectors in one repeatable-read snapshot."""
        if (not 1 <= top_k <= 20 or not 1 <= dimensions <= 16_000
                or len(query_vector) != dimensions
                or any(not math.isfinite(value) for value in query_vector)
                or not math.isfinite(math.hypot(*query_vector))
                or math.hypot(*query_vector) == 0):
            raise ValueError("Invalid visual search vector or limit.")
        query_json = json.dumps(query_vector, allow_nan=False)
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    require_current_schema(connection)
                    spaces = connection.execute(
                        "SELECT DISTINCT model_name, model_revision, dimensions "
                        "FROM visual_interpretation_embeddings WHERE collection_id = %s "
                        "AND text_format_version = %s",
                        (collection_id, text_format_version),
                    ).fetchall()
                    if not spaces:
                        return []
                    matching = [row for row in spaces if row[0] == model_name]
                    if not matching:
                        raise ModelNameMismatch("Visual embeddings use another model.")
                    if not any(row[1] == model_revision for row in matching):
                        raise ModelRevisionMismatch("Visual embeddings use another model revision.")
                    if not any(row[2] == dimensions for row in matching if row[1] == model_revision):
                        raise ModelDimensionMismatch("Visual embeddings use another vector dimension.")
                    rows = connection.execute(
                        "WITH eligible AS MATERIALIZED ("
                        "SELECT e.collection_id, e.document_id, e.element_id, e.interpretation_id, e.embedding "
                        "FROM visual_interpretation_embeddings e "
                        "WHERE e.collection_id = %s AND e.model_name = %s AND e.model_revision = %s "
                        "AND e.dimensions = %s AND e.text_format_version = %s "
                        "AND vector_dims(e.embedding) = %s AND vector_norm(e.embedding) > 0"
                        "), ranked AS MATERIALIZED ("
                        "SELECT collection_id, document_id, element_id, interpretation_id, "
                        "embedding <=> %s::vector AS cosine_distance FROM eligible "
                        "ORDER BY cosine_distance, document_id, element_id, interpretation_id LIMIT %s"
                        ") SELECT r.document_id, d.filename, r.element_id, r.interpretation_id, "
                        "v.page_number, v.coordinate_system, v.x0, v.y0, v.x1, v.y1, v.element_type, "
                        "i.analysis_task, i.prompt_version, i.model_name, i.model_digest, "
                        "i.description, i.observations, i.limitations, i.evidence_kind, "
                        "i.verification_status, r.cosine_distance "
                        "FROM ranked r JOIN visual_interpretations i ON i.collection_id = r.collection_id "
                        "AND i.document_id = r.document_id AND i.element_id = r.element_id "
                        "AND i.interpretation_id = r.interpretation_id "
                        "JOIN visual_elements v ON v.collection_id = r.collection_id "
                        "AND v.document_id = r.document_id AND v.element_id = r.element_id "
                        "JOIN indexed_documents d ON d.collection_id = r.collection_id "
                        "AND d.document_id = r.document_id "
                        "ORDER BY r.cosine_distance, r.document_id, r.element_id, r.interpretation_id",
                        (collection_id, model_name, model_revision, dimensions, text_format_version,
                         dimensions, query_json, top_k),
                    ).fetchall()
                    return [RetrievedVisualInterpretation(
                        document_id=row[0], filename=row[1], element_id=row[2],
                        interpretation_id=row[3], page_number=row[4], coordinate_system=row[5],
                        x0=float(row[6]), y0=float(row[7]), x1=float(row[8]), y1=float(row[9]),
                        element_type=row[10], analysis_task=row[11], prompt_version=row[12],
                        interpretation_model_name=row[13], interpretation_model_digest=row[14],
                        description=row[15], observations=row[16], limitations=row[17],
                        evidence_kind=row[18], verification_status=row[19], cosine_distance=row[20],
                    ) for row in rows]
        except (psycopg.Error, MigrationError) as exc:
            raise StorageError("Visual retrieval is unavailable.") from exc
