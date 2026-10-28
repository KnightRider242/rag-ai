CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE collections (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE indexed_documents (
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    page_count INTEGER,
    original_content BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, document_id)
);

CREATE TABLE indexed_chunks (
    collection_id UUID NOT NULL,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    PRIMARY KEY (collection_id, chunk_id),
    UNIQUE (collection_id, document_id, chunk_index),
    FOREIGN KEY (collection_id, document_id)
        REFERENCES indexed_documents(collection_id, document_id) ON DELETE CASCADE
);

CREATE TABLE embedding_spaces (
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 16000),
    PRIMARY KEY (collection_id, model_name),
    UNIQUE (collection_id, model_name, dimensions)
);

CREATE TABLE chunk_embeddings (
    collection_id UUID NOT NULL,
    chunk_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (collection_id, chunk_id, model_name),
    FOREIGN KEY (collection_id, chunk_id)
        REFERENCES indexed_chunks(collection_id, chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (collection_id, model_name, dimensions)
        REFERENCES embedding_spaces(collection_id, model_name, dimensions) ON DELETE CASCADE,
    CHECK (vector_dims(embedding) = dimensions)
);

CREATE INDEX indexed_chunks_collection_document_idx
    ON indexed_chunks(collection_id, document_id);
