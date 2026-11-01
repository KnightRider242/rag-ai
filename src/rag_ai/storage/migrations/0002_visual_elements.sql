CREATE TABLE visual_elements (
    collection_id UUID NOT NULL,
    document_id TEXT NOT NULL,
    element_id TEXT NOT NULL CHECK (element_id ~ '^[0-9a-f]{64}$'),
    source_sha256 TEXT NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$' AND source_sha256 = document_id),
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    element_type TEXT NOT NULL CHECK (element_type IN ('raster_occurrence', 'table', 'vector_path')),
    occurrence_index INTEGER NOT NULL CHECK (occurrence_index >= 0),
    extractor_name TEXT NOT NULL CHECK (length(extractor_name) BETWEEN 1 AND 80),
    extractor_version TEXT NOT NULL CHECK (length(extractor_version) BETWEEN 1 AND 80),
    coordinate_system TEXT NOT NULL CHECK (coordinate_system = 'displayed_pdf_points'),
    page_width NUMERIC(12,4) NOT NULL,
    page_height NUMERIC(12,4) NOT NULL,
    page_rotation SMALLINT NOT NULL CHECK (page_rotation IN (0, 90, 180, 270)),
    x0 NUMERIC(12,4) NOT NULL,
    y0 NUMERIC(12,4) NOT NULL,
    x1 NUMERIC(12,4) NOT NULL,
    y1 NUMERIC(12,4) NOT NULL,
    extraction_metadata JSONB NOT NULL CHECK (
        jsonb_typeof(extraction_metadata) = 'object'
        AND octet_length(extraction_metadata::text) <= 16384
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, document_id, element_id),
    UNIQUE (collection_id, document_id, extractor_name, extractor_version,
            page_number, element_type, occurrence_index),
    FOREIGN KEY (collection_id, document_id)
        REFERENCES indexed_documents(collection_id, document_id) ON DELETE CASCADE,
    CHECK (
        page_width <> 'NaN'::numeric AND page_height <> 'NaN'::numeric
        AND x0 <> 'NaN'::numeric AND y0 <> 'NaN'::numeric
        AND x1 <> 'NaN'::numeric AND y1 <> 'NaN'::numeric
        AND page_width > 0 AND page_height > 0
        AND 0 <= x0 AND x0 < x1 AND x1 <= page_width
        AND 0 <= y0 AND y0 < y1 AND y1 <= page_height
    )
);

CREATE INDEX visual_elements_document_page_idx
    ON visual_elements (collection_id, document_id, page_number, element_type, element_id);
