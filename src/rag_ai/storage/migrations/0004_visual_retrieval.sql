ALTER TABLE embedding_spaces
    ADD CONSTRAINT embedding_spaces_revision_identity_unique
    UNIQUE (collection_id, model_name, model_revision, dimensions);

CREATE TABLE visual_interpretation_embeddings (
    collection_id UUID NOT NULL,
    document_id TEXT NOT NULL,
    element_id TEXT NOT NULL,
    interpretation_id UUID NOT NULL,
    model_name TEXT NOT NULL CHECK (length(model_name) BETWEEN 1 AND 200),
    model_revision TEXT NOT NULL CHECK (model_revision ~ '^[0-9a-f]{64}$'),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 16000),
    text_format_version TEXT NOT NULL CHECK (length(text_format_version) BETWEEN 1 AND 80),
    text_sha256 TEXT NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, document_id, element_id, interpretation_id,
                 model_name, text_format_version),
    FOREIGN KEY (collection_id, document_id, element_id, interpretation_id)
        REFERENCES visual_interpretations(collection_id, document_id, element_id, interpretation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (collection_id, model_name, model_revision, dimensions)
        REFERENCES embedding_spaces(collection_id, model_name, model_revision, dimensions)
        ON DELETE CASCADE,
    CHECK (vector_dims(embedding) = dimensions AND vector_norm(embedding) > 0)
);

CREATE INDEX visual_interpretation_embeddings_collection_model_idx
    ON visual_interpretation_embeddings
    (collection_id, model_name, text_format_version, document_id, element_id, interpretation_id);
