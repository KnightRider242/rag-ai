CREATE FUNCTION valid_visual_interpretation_items(items JSONB, max_items INTEGER)
RETURNS BOOLEAN LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE item JSONB;
BEGIN
    IF jsonb_typeof(items) <> 'array' OR jsonb_array_length(items) > max_items THEN
        RETURN FALSE;
    END IF;
    FOR item IN SELECT value FROM jsonb_array_elements(items) AS value LOOP
        IF jsonb_typeof(item) <> 'string'
            OR length(trim(item #>> '{}')) NOT BETWEEN 1 AND 240 THEN
            RETURN FALSE;
        END IF;
    END LOOP;
    RETURN TRUE;
END;
$$;

CREATE TABLE visual_interpretations (
    collection_id UUID NOT NULL,
    document_id TEXT NOT NULL,
    element_id TEXT NOT NULL,
    interpretation_id UUID NOT NULL,
    request_key UUID,
    analysis_task TEXT NOT NULL CHECK (analysis_task IN (
        'describe_figure', 'describe_diagram', 'summarize_page', 'interpret_table',
        'examine_scanned_page', 'extract_visible_labels')),
    prompt_version TEXT NOT NULL CHECK (length(prompt_version) BETWEEN 1 AND 80),
    model_name TEXT NOT NULL CHECK (length(model_name) BETWEEN 1 AND 200),
    model_digest TEXT NOT NULL CHECK (model_digest ~ '^[0-9a-f]{64}$'),
    render_scale NUMERIC(3,2) NOT NULL CHECK (render_scale BETWEEN 0.5 AND 2),
    rendered_image_sha256 TEXT NOT NULL CHECK (rendered_image_sha256 ~ '^[0-9a-f]{64}$'),
    description TEXT NOT NULL CHECK (length(trim(description)) BETWEEN 1 AND 1200),
    observations JSONB NOT NULL CHECK (
        valid_visual_interpretation_items(observations, 8)
        AND octet_length(observations::text) <= 8192),
    limitations JSONB NOT NULL CHECK (
        valid_visual_interpretation_items(limitations, 5)
        AND octet_length(limitations::text) <= 5120),
    evidence_kind TEXT NOT NULL DEFAULT 'model_generated_interpretation'
        CHECK (evidence_kind = 'model_generated_interpretation'),
    verification_status TEXT NOT NULL DEFAULT 'unverified'
        CHECK (verification_status = 'unverified'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, document_id, element_id, interpretation_id),
    FOREIGN KEY (collection_id, document_id, element_id)
        REFERENCES visual_elements(collection_id, document_id, element_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX visual_interpretations_request_key_idx
    ON visual_interpretations (collection_id, document_id, element_id, request_key)
    WHERE request_key IS NOT NULL;

CREATE INDEX visual_interpretations_element_created_idx
    ON visual_interpretations (collection_id, document_id, element_id, created_at DESC, interpretation_id);

CREATE FUNCTION reject_visual_interpretation_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'visual interpretations are immutable';
END;
$$;

CREATE TRIGGER visual_interpretations_no_update
    BEFORE UPDATE ON visual_interpretations
    FOR EACH ROW EXECUTE FUNCTION reject_visual_interpretation_update();
