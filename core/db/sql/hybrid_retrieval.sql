-- Hybrid retrieval in PostgreSQL: pgvector HNSW distance + ts_rank_cd full-text search (Task P4.9)
-- Fuses semantic ranking and lexical ranking using Reciprocal Rank Fusion (RRF, default k=60)

SET LOCAL hnsw.ef_search = :ef_search;

WITH semantic_ranking AS (
    SELECT
        rc.id AS chunk_id,
        rc.resume_id,
        rc.section_type,
        rc.text,
        1.0 - (rc.embedding <=> :query_embedding) AS sem_score,
        ROW_NUMBER() OVER (ORDER BY rc.embedding <=> :query_embedding ASC) AS sem_rank
    FROM resume_chunks rc
    LIMIT :candidate_pool_size
),
lexical_ranking AS (
    SELECT
        rc.id AS chunk_id,
        ts_rank_cd(to_tsvector('english', rc.text), plainto_tsquery('english', :query_text)) AS lex_score,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', rc.text), plainto_tsquery('english', :query_text)) DESC) AS lex_rank
    FROM resume_chunks rc
    WHERE to_tsvector('english', rc.text) @@ plainto_tsquery('english', :query_text)
    LIMIT :candidate_pool_size
)
SELECT
    s.chunk_id,
    s.resume_id,
    r.candidate_name,
    s.section_type,
    s.text,
    s.sem_score,
    COALESCE(l.lex_score, 0.0) AS lex_score,
    (1.0 / (:rrf_k + s.sem_rank)) + (1.0 / (:rrf_k + COALESCE(l.lex_rank, 1000))) AS rrf_score
FROM semantic_ranking s
JOIN resumes r ON r.id = s.resume_id
LEFT JOIN lexical_ranking l ON s.chunk_id = l.chunk_id
ORDER BY rrf_score DESC
LIMIT :limit;
