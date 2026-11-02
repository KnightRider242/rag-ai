"""Exact text and visual search in one read-only PostgreSQL snapshot."""

import json
import math
from uuid import UUID

import psycopg

from rag_ai.retrieval.models import RetrievedChunk
from rag_ai.retrieval.visual import RetrievedVisualInterpretation, TEXT_FORMAT_VERSION
from rag_ai.storage.migrate import MigrationError, require_current_schema
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch, StorageError,
)


def _check_spaces(spaces: list[tuple[str, str, int]], model_name: str,
                  model_revision: str, dimensions: int) -> None:
    """Reject every incompatible populated space, including mixed collections."""
    if any(name != model_name for name, _, _ in spaces):
        raise ModelNameMismatch("Collection contains incompatible embedding models.")
    if any(revision != model_revision for _, revision, _ in spaces):
        raise ModelRevisionMismatch("Collection contains incompatible model revisions.")
    if any(size != dimensions for _, _, size in spaces):
        raise ModelDimensionMismatch("Collection contains incompatible vector dimensions.")


class PostgresCombinedRetrievalStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def search(self, *, collection_id: UUID, query_vector: list[float], model_name: str,
               model_revision: str, dimensions: int, text_limit: int,
               visual_limit: int) -> tuple[list[RetrievedChunk], list[RetrievedVisualInterpretation]]:
        if (not 1 <= text_limit <= 20 or not 1 <= visual_limit <= 100
                or not 1 <= dimensions <= 16_000 or len(query_vector) != dimensions
                or any(not math.isfinite(value) for value in query_vector)
                or not math.isfinite(math.hypot(*query_vector))
                or math.hypot(*query_vector) == 0):
            raise ValueError("Invalid combined retrieval vector or limit.")
        query_json = json.dumps(query_vector, allow_nan=False)
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    require_current_schema(connection)
                    text_spaces = connection.execute(
                        "SELECT DISTINCT e.model_name, s.model_revision, e.dimensions "
                        "FROM chunk_embeddings e JOIN embedding_spaces s "
                        "ON s.collection_id = e.collection_id AND s.model_name = e.model_name "
                        "AND s.dimensions = e.dimensions WHERE e.collection_id = %s",
                        (collection_id,),
                    ).fetchall()
                    visual_spaces = connection.execute(
                        "SELECT DISTINCT model_name, model_revision, dimensions "
                        "FROM visual_interpretation_embeddings WHERE collection_id = %s",
                        (collection_id,),
                    ).fetchall()
                    _check_spaces(text_spaces, model_name, model_revision, dimensions)
                    _check_spaces(visual_spaces, model_name, model_revision, dimensions)
                    text_rows = connection.execute(
                        "WITH eligible AS MATERIALIZED ("
                        "SELECT e.collection_id, c.document_id, e.chunk_id, c.chunk_index, e.embedding "
                        "FROM chunk_embeddings e JOIN indexed_chunks c "
                        "ON c.collection_id = e.collection_id AND c.chunk_id = e.chunk_id "
                        "JOIN embedding_spaces s ON s.collection_id = e.collection_id "
                        "AND s.model_name = e.model_name AND s.dimensions = e.dimensions "
                        "WHERE e.collection_id = %s AND e.model_name = %s AND e.dimensions = %s "
                        "AND s.model_revision = %s AND vector_dims(e.embedding) = %s "
                        "AND vector_norm(e.embedding) > 0"
                        "), ranked AS MATERIALIZED ("
                        "SELECT collection_id, document_id, chunk_id, chunk_index, "
                        "embedding <=> %s::vector AS cosine_distance FROM eligible "
                        "ORDER BY cosine_distance, document_id, chunk_index, chunk_id LIMIT %s"
                        ") SELECT r.document_id, r.chunk_id, r.chunk_index, d.filename, d.file_type, "
                        "c.page_number, c.content, r.cosine_distance FROM ranked r "
                        "JOIN indexed_chunks c ON c.collection_id = r.collection_id AND c.chunk_id = r.chunk_id "
                        "JOIN indexed_documents d ON d.collection_id = r.collection_id "
                        "AND d.document_id = r.document_id "
                        "ORDER BY r.cosine_distance, r.document_id, r.chunk_index, r.chunk_id",
                        (collection_id, model_name, dimensions, model_revision,
                         dimensions, query_json, text_limit),
                    ).fetchall()
                    visual_rows = connection.execute(
                        "WITH eligible AS MATERIALIZED ("
                        "SELECT e.collection_id, e.document_id, e.element_id, e.interpretation_id, e.embedding "
                        "FROM visual_interpretation_embeddings e "
                        "WHERE e.collection_id = %s AND e.model_name = %s AND e.model_revision = %s "
                        "AND e.dimensions = %s AND e.text_format_version = %s "
                        "AND vector_dims(e.embedding) = %s AND vector_norm(e.embedding) > 0"
                        "), scored AS MATERIALIZED ("
                        "SELECT collection_id, document_id, element_id, interpretation_id, "
                        "embedding <=> %s::vector AS cosine_distance FROM eligible"
                        "), distinct_elements AS MATERIALIZED ("
                        "SELECT DISTINCT ON (collection_id, document_id, element_id) "
                        "collection_id, document_id, element_id, interpretation_id, cosine_distance "
                        "FROM scored ORDER BY collection_id, document_id, element_id, "
                        "cosine_distance, interpretation_id"
                        "), ranked AS MATERIALIZED ("
                        "SELECT * FROM distinct_elements "
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
                        (collection_id, model_name, model_revision, dimensions,
                         TEXT_FORMAT_VERSION, dimensions, query_json, visual_limit),
                    ).fetchall()
                    texts = [RetrievedChunk(
                        document_id=r[0], chunk_id=r[1], chunk_index=r[2], filename=r[3],
                        file_type=r[4], page_number=r[5], text=r[6], cosine_distance=r[7],
                    ) for r in text_rows]
                    visuals = [RetrievedVisualInterpretation(
                        document_id=r[0], filename=r[1], element_id=r[2], interpretation_id=r[3],
                        page_number=r[4], coordinate_system=r[5], x0=float(r[6]), y0=float(r[7]),
                        x1=float(r[8]), y1=float(r[9]), element_type=r[10], analysis_task=r[11],
                        prompt_version=r[12], interpretation_model_name=r[13],
                        interpretation_model_digest=r[14], description=r[15], observations=r[16],
                        limitations=r[17], evidence_kind=r[18], verification_status=r[19],
                        cosine_distance=r[20],
                    ) for r in visual_rows]
                    return texts, visuals
        except (psycopg.Error, MigrationError) as exc:
            raise StorageError("Combined retrieval is unavailable.") from exc
