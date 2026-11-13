"""Fresh collection-scoped citation checks after local generation."""

from decimal import Decimal
from uuid import UUID

import psycopg

from rag_ai.retrieval.models import CombinedTextResult, CombinedVisualResult
from rag_ai.storage.postgres import StorageError


class PostgresAnswerSourceResolver:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def verify_sources(self, *, collection_id: UUID,
                       sources: list[CombinedTextResult | CombinedVisualResult]) -> bool:
        """Reject deletion or mutation since retrieval in a fresh database snapshot."""
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    for source in sources:
                        if isinstance(source, CombinedTextResult):
                            row = connection.execute(
                                "SELECT c.chunk_index, c.page_number, c.content, d.filename, d.file_type "
                                "FROM indexed_chunks c JOIN indexed_documents d "
                                "ON d.collection_id = c.collection_id AND d.document_id = c.document_id "
                                "WHERE c.collection_id = %s AND c.document_id = %s AND c.chunk_id = %s",
                                (collection_id, source.document_id, source.chunk_id),
                            ).fetchone()
                            expected = (source.chunk_index, source.page_number, source.text,
                                        source.filename, source.file_type)
                        else:
                            row = connection.execute(
                                "SELECT v.page_number, v.coordinate_system, v.x0, v.y0, v.x1, v.y1, "
                                "v.element_type, i.analysis_task, i.prompt_version, i.model_name, "
                                "i.model_digest, i.description, i.observations, i.limitations, "
                                "i.evidence_kind, i.verification_status, d.filename "
                                "FROM visual_interpretations i JOIN visual_elements v "
                                "ON v.collection_id = i.collection_id AND v.document_id = i.document_id "
                                "AND v.element_id = i.element_id JOIN indexed_documents d "
                                "ON d.collection_id = i.collection_id AND d.document_id = i.document_id "
                                "WHERE i.collection_id = %s AND i.document_id = %s "
                                "AND i.element_id = %s AND i.interpretation_id = %s",
                                (collection_id, source.document_id, source.element_id,
                                 source.interpretation_id),
                            ).fetchone()
                            # NUMERIC(12,4) is read as Decimal, while combined retrieval
                            # exposes floats. Their shortest decimal strings round-trip
                            # the stored four-place values without a comparison tolerance.
                            expected = (source.page_number, source.coordinate_system,
                                        *(Decimal(str(value)) for value in
                                          (source.x0, source.y0, source.x1, source.y1)),
                                        source.element_type, source.analysis_task, source.prompt_version,
                                        source.interpretation_model_name, source.interpretation_model_digest,
                                        source.description, source.observations, source.limitations,
                                        source.evidence_kind, source.verification_status, source.filename)
                        if row is None or tuple(row) != expected:
                            return False
                    return True
        except psycopg.Error as exc:
            raise StorageError("Citation sources are unavailable.") from exc
