"""Minimal PostgreSQL/pgvector store for local document indexing."""

import json
from hashlib import sha256
from itertools import chain
import math
from typing import Iterable
from uuid import UUID

import psycopg

from rag_ai.documents.chunker import ChunkedDocument
from rag_ai.documents.parser import ParsedDocument
from rag_ai.embeddings.base import EMBEDDING_BATCH_SIZE, EmbeddingBatch
from rag_ai.retrieval.models import RetrievedChunk
from rag_ai.storage.migrate import MigrationError, require_current_schema


class StorageError(RuntimeError):
    """Storage is unavailable; details are intentionally omitted."""


class ModelDimensionMismatch(ValueError):
    """A model name was already recorded with a different vector size."""


class ModelRevisionMismatch(ValueError):
    """A model name was already recorded with a different revision."""


class ModelNameMismatch(ValueError):
    """The collection has embeddings, but none use the requested model."""


class PostgresIndexStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def save(
        self, *, collection_id: UUID, document: ParsedDocument, original_content: bytes,
        chunked: ChunkedDocument, first_batch: EmbeddingBatch,
        remaining_batches: Iterable[EmbeddingBatch] = (),
    ) -> None:
        """Stage bounded batches, then atomically replace a collection document."""
        if (sha256(original_content).hexdigest() != document.document_id
                or chunked.document_id != document.document_id
                or any(chunk.document_id != document.document_id for chunk in chunked.chunks)):
            raise ValueError("Document identity does not match its indexed content.")
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                require_current_schema(connection)
                connection.execute(
                    "CREATE TEMP TABLE rag_ai_index_stage ("
                    "chunk_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, page_number INTEGER, "
                    "content TEXT NOT NULL, embedding vector NOT NULL) ON COMMIT PRESERVE ROWS"
                )
                connection.commit()
                offset = 0
                for batch in chain((first_batch,), remaining_batches):
                    if (batch.model_name != first_batch.model_name
                            or batch.model_revision != first_batch.model_revision
                            or batch.dimensions != first_batch.dimensions
                            or not 1 <= len(batch.vectors) <= EMBEDDING_BATCH_SIZE
                            or offset + len(batch.vectors) > len(chunked.chunks)
                            or any(len(vector) != batch.dimensions
                                   or any(not math.isfinite(value) for value in vector)
                                   for vector in batch.vectors)):
                        raise ValueError("Embedding batch does not match the document or model.")
                    with connection.transaction():
                        for chunk, vector in zip(
                            chunked.chunks[offset:offset + len(batch.vectors)], batch.vectors, strict=True,
                        ):
                            connection.execute(
                                "INSERT INTO rag_ai_index_stage(chunk_id, chunk_index, page_number, content, embedding) "
                                "VALUES (%s, %s, %s, %s, %s::vector)",
                                (chunk.chunk_id, chunk.chunk_index, chunk.page_number, chunk.text,
                                 json.dumps(vector, allow_nan=False)),
                            )
                    offset += len(batch.vectors)
                if offset != len(chunked.chunks):
                    raise ValueError("Embedding batches do not cover every chunk.")

                with connection.transaction():
                    connection.execute("INSERT INTO collections(id) VALUES (%s) ON CONFLICT DO NOTHING", (collection_id,))
                    connection.execute(
                        "INSERT INTO embedding_spaces(collection_id, model_name, model_revision, dimensions) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (collection_id, first_batch.model_name, first_batch.model_revision, first_batch.dimensions),
                    )
                    row = connection.execute(
                        "SELECT dimensions, model_revision FROM embedding_spaces "
                        "WHERE collection_id = %s AND model_name = %s",
                        (collection_id, first_batch.model_name),
                    ).fetchone()
                    if row is None or row[0] != first_batch.dimensions:
                        raise ModelDimensionMismatch(
                            "Embedding model dimensions differ from this collection's recorded dimensions."
                        )
                    if row[1] != first_batch.model_revision:
                        raise ModelRevisionMismatch(
                            "Embedding model revision differs from this collection's recorded revision."
                        )
                    connection.execute(
                        "INSERT INTO indexed_documents(collection_id, document_id, filename, file_type, "
                        "page_count, original_content) VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (collection_id, document_id) DO UPDATE SET "
                        "filename = EXCLUDED.filename, file_type = EXCLUDED.file_type, "
                        "page_count = EXCLUDED.page_count, original_content = EXCLUDED.original_content",
                        (collection_id, document.document_id, document.filename, document.file_type,
                         document.page_count, original_content),
                    )
                    connection.execute(
                        "DELETE FROM indexed_chunks WHERE collection_id = %s AND document_id = %s",
                        (collection_id, document.document_id),
                    )
                    connection.execute(
                        "INSERT INTO indexed_chunks(collection_id, chunk_id, document_id, chunk_index, "
                        "page_number, content, chunk_size, chunk_overlap) "
                        "SELECT %s, chunk_id, %s, chunk_index, page_number, content, %s, %s "
                        "FROM rag_ai_index_stage",
                        (collection_id, document.document_id, chunked.chunk_size, chunked.chunk_overlap),
                    )
                    connection.execute(
                        "INSERT INTO chunk_embeddings(collection_id, chunk_id, model_name, dimensions, embedding) "
                        "SELECT %s, chunk_id, %s, %s, embedding FROM rag_ai_index_stage",
                        (collection_id, first_batch.model_name, first_batch.dimensions),
                    )
        except (psycopg.Error, MigrationError) as exc:
            raise StorageError("Database indexing failed.") from exc

    def delete_document(self, *, collection_id: UUID, document_id: str) -> None:
        """Delete only the selected collection's document and its derived rows."""
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    require_current_schema(connection)
                    connection.execute(
                        "DELETE FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
                        (collection_id, document_id),
                    )
        except (psycopg.Error, MigrationError) as exc:
            raise StorageError("Database document deletion failed.") from exc

    def search(
        self, *, collection_id: UUID, query_vector: list[float], model_name: str,
        model_revision: str, dimensions: int, top_k: int,
    ) -> list[RetrievedChunk]:
        """Rank only compatible, nonzero vectors in one stable read-only snapshot."""
        if not 1 <= top_k <= 20 or not 1 <= dimensions <= 16_000:
            raise ValueError("Invalid retrieval limit or vector dimensions.")
        if (len(query_vector) != dimensions
                or any(not math.isfinite(value) for value in query_vector)):
            raise ValueError("Invalid query vector.")
        norm = math.hypot(*query_vector)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Invalid query vector.")
        query_json = json.dumps(query_vector, allow_nan=False)
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    require_current_schema(connection)
                    spaces = connection.execute(
                        "SELECT DISTINCT e.model_name, s.model_revision, e.dimensions "
                        "FROM chunk_embeddings e JOIN embedding_spaces s "
                        "ON s.collection_id = e.collection_id AND s.model_name = e.model_name "
                        "AND s.dimensions = e.dimensions "
                        "WHERE e.collection_id = %s",
                        (collection_id,),
                    ).fetchall()
                    if not spaces:
                        return []
                    matching = [row for row in spaces if row[0] == model_name]
                    if not matching:
                        raise ModelNameMismatch("Collection embeddings use another model.")
                    if not any(row[1] == model_revision for row in matching):
                        raise ModelRevisionMismatch("Collection embeddings use another model revision.")
                    if not any(row[2] == dimensions for row in matching if row[1] == model_revision):
                        raise ModelDimensionMismatch("Collection embeddings use another vector dimension.")

                    rows = connection.execute(
                        "WITH eligible AS MATERIALIZED ("
                        "SELECT e.collection_id, c.document_id, e.chunk_id, c.chunk_index, e.embedding "
                        "FROM chunk_embeddings e "
                        "JOIN indexed_chunks c ON c.collection_id = e.collection_id AND c.chunk_id = e.chunk_id "
                        "JOIN embedding_spaces s ON s.collection_id = e.collection_id "
                        "AND s.model_name = e.model_name AND s.dimensions = e.dimensions "
                        "WHERE e.collection_id = %s AND e.model_name = %s "
                        "AND e.dimensions = %s AND s.model_revision = %s "
                        "AND vector_dims(e.embedding) = %s AND vector_norm(e.embedding) > 0"
                        "), ranked AS MATERIALIZED ("
                        "SELECT collection_id, document_id, chunk_id, chunk_index, "
                        "embedding <=> %s::vector AS cosine_distance FROM eligible "
                        "ORDER BY cosine_distance ASC, document_id ASC, chunk_index ASC, chunk_id ASC LIMIT %s"
                        ") SELECT r.document_id, r.chunk_id, r.chunk_index, d.filename, d.file_type, "
                        "c.page_number, c.content, r.cosine_distance "
                        "FROM ranked r JOIN indexed_chunks c "
                        "ON c.collection_id = r.collection_id AND c.chunk_id = r.chunk_id "
                        "JOIN indexed_documents d ON d.collection_id = r.collection_id "
                        "AND d.document_id = r.document_id "
                        "ORDER BY r.cosine_distance ASC, r.document_id ASC, "
                        "r.chunk_index ASC, r.chunk_id ASC",
                        (collection_id, model_name, dimensions, model_revision, dimensions, query_json, top_k),
                    ).fetchall()
                    return [RetrievedChunk(
                        document_id=row[0], chunk_id=row[1], chunk_index=row[2],
                        filename=row[3], file_type=row[4], page_number=row[5],
                        text=row[6], cosine_distance=row[7],
                    ) for row in rows]
        except (psycopg.Error, MigrationError) as exc:
            raise StorageError("Database retrieval failed.") from exc
