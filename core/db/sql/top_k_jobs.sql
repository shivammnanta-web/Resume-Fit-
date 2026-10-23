-- Reverse retrieval: Top-K matching jobs for a candidate's resume embedding (Task P4.7)
-- Query: Orders by cosine distance (<=>) using jobs table vector index

SET LOCAL hnsw.ef_search = :ef_search;

SELECT
    j.id AS job_id,
    j.role_title,
    j.min_years_experience,
    j.min_degree,
    j.required_skills,
    j.preferred_skills,
    1.0 - (j.embedding <=> :candidate_embedding) AS cosine_similarity
FROM jobs j
ORDER BY j.embedding <=> :candidate_embedding ASC
LIMIT :limit;
