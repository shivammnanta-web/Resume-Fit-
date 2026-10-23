-- Section-restricted vector retrieval: filter chunks by specific section types (Task P4.8)
-- Prioritizes high-evidence sections (e.g. 'experience', 'projects') before weaker sections

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
WHERE rc.section_type = ANY(:allowed_sections)
ORDER BY rc.embedding <=> :query_embedding ASC
LIMIT :limit;
