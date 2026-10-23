-- Top-K matching resume chunks for a job embedding using pgvector HNSW index (Task P4.6)
-- Query: Orders by cosine distance (<=>) using vector_cosine_ops
-- Returns chunk metadata and computed cosine similarity (1.0 - distance)

SET LOCAL hnsw.ef_search = :ef_search;

SELECT
    rc.id AS chunk_id,
    rc.resume_id,
    r.candidate_name,
    rc.section_type,
    rc.text,
    rc.token_count,
    1.0 - (rc.embedding <=> :query_embedding) AS cosine_similarity
FROM resume_chunks rc
JOIN resumes r ON r.id = rc.resume_id
ORDER BY rc.embedding <=> :query_embedding ASC
LIMIT :limit;
