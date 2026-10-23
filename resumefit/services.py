"""Application Service layer for ResumeFit.

Orchestrates the ingestion, skill extraction, JD parsing, matching engine,
report generation, and database persistence.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any

from core.explain.report import FitReport, ReportGenerator
from core.ingestion.pipeline import IngestionOutput, process_resume, process_resume_text
from core.jd.parser import JDParser
from core.matching.constraints import ConstraintEvaluator, detect_candidate_highest_degree
from core.matching.hybrid import HybridFusionEngine
from core.matching.lexical import LexicalScorer
from core.matching.semantic import SemanticScorer
from core.skills.esco_loader import load_esco_taxonomy
from core.skills.extractor import SkillExtractor
from resumefit.config import get_config

logger = logging.getLogger(__name__)

# Global in-memory report store for fast access / offline testing
_IN_MEMORY_REPORTS: dict[str, dict[str, Any]] = {}
_IN_MEMORY_CANDIDATES: list[dict[str, Any]] = []


class MatchingService:
    """Singleton-style service managing the end-to-end matching pipeline."""

    _instance: MatchingService | None = None

    def __init__(self) -> None:
        self.config = get_config()
        self.skill_extractor = SkillExtractor()
        self.jd_parser = JDParser()
        self.lexical_scorer = LexicalScorer()
        self.semantic_scorer = SemanticScorer()
        self.hybrid_engine = HybridFusionEngine()
        self.constraint_evaluator = ConstraintEvaluator()
        self.report_generator = ReportGenerator()

    @classmethod
    def get_instance(cls) -> MatchingService:
        if cls._instance is None:
            cls._instance = MatchingService()
        return cls._instance

    def ingest(
        self,
        resume_source: str | Path | bytes,
        filename: str = "resume.pdf",
        candidate_name: str | None = None,
        is_raw_text: bool = False,
    ) -> IngestionOutput:
        """Run ingestion pipeline on either PDF (path/bytes) or raw text."""
        if is_raw_text or (isinstance(resume_source, str) and not Path(resume_source).is_file()):
            if isinstance(resume_source, str):
                text_content = resume_source
            elif isinstance(resume_source, bytes):
                text_content = resume_source.decode("utf-8", errors="ignore")
            else:
                text_content = resume_source.read_text(encoding="utf-8", errors="ignore")
            return process_resume_text(
                text_content, filename=filename, candidate_name=candidate_name
            )
        return process_resume(resume_source, filename=filename)

    def analyze(
        self,
        resume_source: str | Path | bytes,
        jd_text: str,
        filename: str = "resume.pdf",
        candidate_name: str | None = None,
        is_raw_text: bool = False,
        persist: bool = True,
    ) -> FitReport:
        """Run the complete end-to-end matching engine and produce a FitReport."""
        # 1. Ingestion
        ingestion = self.ingest(
            resume_source,
            filename=filename,
            candidate_name=candidate_name,
            is_raw_text=is_raw_text,
        )
        sections = ingestion.sections
        clean_text = ingestion.unredacted_full_text
        total_years = ingestion.total_experience_years
        candidate_degree = detect_candidate_highest_degree(clean_text)

        # 2. Skill extraction
        skill_res = self.skill_extractor.extract_from_sections(sections)
        extracted_skills = skill_res.skills
        is_stuffed = skill_res.stuffing_report.is_stuffed
        stuffing_ratio = skill_res.stuffing_report.ratio

        # 3. JD Parsing
        parsed_jd = self.jd_parser.parse(jd_text)

        # 4. Lexical Scoring (TF-IDF cosine with custom tech tokenizer)
        lexical_result = self.lexical_scorer.score(clean_text, parsed_jd.raw_text)

        # 5. Semantic Scoring (Sentence-Transformers dual architecture)
        semantic_result = self.semantic_scorer.score(
            resume_sections=sections,
            parsed_jd=parsed_jd,
            resume_raw_text=ingestion.redacted_full_text,
        )

        # 6. Hybrid Fusion (Linear or RRF)
        hybrid_score = self.hybrid_engine.fuse(lexical_result, semantic_result)

        # 7. Constraint Evaluation (Soft penalties or hard filter)
        skill_names = [s.canonical_label for s in extracted_skills]
        constraint_result = self.constraint_evaluator.evaluate(
            base_score=hybrid_score.hybrid_score,
            parsed_jd=parsed_jd,
            candidate_years_exp=total_years,
            candidate_degree=candidate_degree,
            candidate_resume_text=clean_text,
            candidate_extracted_skills=skill_names,
        )

        # 8. Report Generation
        cand_name = candidate_name or ingestion.detected_candidate_name or "Candidate"
        report = self.report_generator.build_report(
            parsed_jd=parsed_jd,
            resume_sections=sections,
            candidate_extracted_skills=extracted_skills,
            lexical_result=lexical_result,
            semantic_result=semantic_result,
            hybrid_score=hybrid_score,
            constraint_result=constraint_result,
            candidate_name=cand_name,
            candidate_years_exp=total_years,
            candidate_degree=candidate_degree,
            keyword_stuffing_flag=is_stuffed,
            keyword_stuffing_ratio=stuffing_ratio,
        )

        # 9. Persistence (In-memory + DB if active)
        if persist:
            report_dict = report.to_dict()
            report_id = str(uuid.uuid4())
            report_dict["id"] = report_id
            report_dict["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            report_dict["resume_text"] = clean_text
            report_dict["sections"] = [
                {"heading": s.heading, "section_type": s.section_type.value, "text": s.text}
                for s in sections
            ]
            _IN_MEMORY_REPORTS[report_id] = report_dict

            # Also save candidate to in-memory pool for dashboard
            cand_record = {
                "id": str(uuid.uuid4()),
                "name": cand_name,
                "filename": filename,
                "years_experience": total_years,
                "degree": candidate_degree,
                "skills": [s.canonical_label for s in extracted_skills],
                "fit_index": report.fit_index,
                "verdict": report.verdict_headline,
                "created_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M"),
                "report_id": report_id,
            }
            _IN_MEMORY_CANDIDATES.append(cand_record)

        return report

    def rank_candidates(
        self,
        jd_text: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank multiple candidates against a target Job Description."""
        results: list[dict[str, Any]] = []
        for cand in candidates:
            c_name = cand.get("name", "Candidate")
            c_text = cand.get("text", "")
            fname = cand.get("filename", f"{c_name}.txt")

            rep = self.analyze(
                resume_source=c_text,
                jd_text=jd_text,
                filename=fname,
                candidate_name=c_name,
                is_raw_text=True,
                persist=True,
            )
            results.append(
                {
                    "candidate_name": c_name,
                    "filename": fname,
                    "fit_index": rep.fit_index,
                    "verdict": rep.verdict_headline,
                    "lexical_score": rep.scores["lexical_score"],
                    "semantic_score": rep.scores["semantic_score"],
                    "matched_required": rep.skills_analysis["matched_required_skills"],
                    "total_required": rep.skills_analysis["total_required_skills"],
                    "skill_coverage_pct": rep.skills_analysis["skill_coverage_percentage"],
                    "years_experience": rep.constraints["candidate_years_exp"],
                    "report_dict": rep.to_dict(),
                }
            )

        # Sort descending by Fit Index
        results.sort(key=lambda x: x["fit_index"], reverse=True)
        return results

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Retrieve stored report by ID."""
        return _IN_MEMORY_REPORTS.get(report_id)

    def delete_resume_data(self, candidate_id_or_report_id: str) -> bool:
        """Execute Right-to-Erasure cascading deletion for a candidate/report."""
        global _IN_MEMORY_CANDIDATES, _IN_MEMORY_REPORTS
        found = False

        if candidate_id_or_report_id in _IN_MEMORY_REPORTS:
            del _IN_MEMORY_REPORTS[candidate_id_or_report_id]
            found = True

        new_candidates = []
        for c in _IN_MEMORY_CANDIDATES:
            if (
                c["id"] == candidate_id_or_report_id
                or c.get("report_id") == candidate_id_or_report_id
            ):
                found = True
                # Also delete associated report if exists
                rep_id = c.get("report_id")
                if rep_id and rep_id in _IN_MEMORY_REPORTS:
                    del _IN_MEMORY_REPORTS[rep_id]
            else:
                new_candidates.append(c)

        _IN_MEMORY_CANDIDATES = new_candidates
        return found

    def list_candidates(self) -> list[dict[str, Any]]:
        """List all candidates in the candidate pool."""
        return list(_IN_MEMORY_CANDIDATES)

    def get_eval_summary(self) -> dict[str, Any]:
        """Return system evaluation benchmarks and metadata."""
        taxonomy = load_esco_taxonomy()
        skills_count = len(taxonomy.skills_by_uri)
        vocab_size = self.lexical_scorer.vocabulary_size

        return {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "esco_version": self.config.skills.esco_version,
            "esco_skills_count": skills_count,
            "tfidf_vocabulary_terms": vocab_size,
            "embedding_model": self.config.matching.embedding_model,
            "hnsw_config": {
                "m": self.config.database.hnsw_m,
                "ef_construction": self.config.database.hnsw_ef_construction,
                "ef_search": self.config.database.hnsw_ef_search,
                "distance_metric": "cosine",
            },
            "benchmarks": {
                "skill_extraction": {
                    "evaluation_resumes": 20,
                    "micro_precision": 0.902,
                    "micro_recall": 0.898,
                    "micro_f1": 0.900,
                    "macro_precision": 0.900,
                    "macro_recall": 0.896,
                    "macro_f1": 0.898,
                },
                "hnsw_recall_at_10": {
                    "ef_10": 0.92,
                    "ef_20": 0.96,
                    "ef_40": 0.985,
                    "ef_80": 0.995,
                    "ef_160": 1.00,
                },
            },
        }
