"""REST API route handlers for ResumeFit."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError
from werkzeug.utils import secure_filename

from core.ingestion.types import UnsupportedDocument
from resumefit.api.schemas import (
    AnalyzeTextRequest,
    DeleteResponse,
    ErrorDetail,
    ErrorEnvelope,
    HealthResponse,
    RankRequest,
    SuggestJobsRequest,
)
from resumefit.limiter import limiter
from resumefit.services import MatchingService

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def make_error_response(code: str, message: str, status_code: int, details: Any = None) -> Any:
    """Format and return a standard API ErrorEnvelope JSON response.

    Masks raw exception details on 500 errors unless debug mode is active to prevent
    sensitive information disclosure in production environments.
    """
    safe_message = message
    safe_details = details

    try:
        is_debug = bool(current_app and current_app.config.get("DEBUG", False))
    except Exception:
        is_debug = False

    if status_code >= 500 and not is_debug:
        safe_message = "An unexpected internal server error occurred."
        safe_details = None

    payload = ErrorEnvelope(
        error=ErrorDetail(code=code, message=safe_message, details=safe_details)
    ).model_dump()
    return jsonify(payload), status_code


@api_bp.route("/health", methods=["GET"])
def health_check() -> Any:
    """System health check endpoint."""
    resp = HealthResponse()
    return jsonify(resp.model_dump()), 200


@api_bp.route("/analyze", methods=["POST"])
@limiter.limit("20 per minute")
def analyze_resume() -> Any:
    """Analyze a single resume against a target job description.

    Supports:
    1. multipart/form-data: 'resume' (file upload, PDF) + 'jd_text' (form field)
    2. application/json: 'resume_text' + 'jd_text' + optional 'candidate_name'
    """
    service = MatchingService.get_instance()

    try:
        # 1. Form-data upload (PDF)
        if "resume" in request.files:
            file_obj = request.files["resume"]
            jd_text = request.form.get("jd_text", "").strip()

            if not jd_text:
                return make_error_response(
                    code="MISSING_FIELD",
                    message="Job description ('jd_text') is required.",
                    status_code=422,
                )

            filename = secure_filename(file_obj.filename or "resume.pdf")
            if not filename.lower().endswith(".pdf"):
                return make_error_response(
                    code="INVALID_FILE_TYPE",
                    message="Only PDF documents are supported for upload.",
                    status_code=422,
                )

            file_bytes = file_obj.read()
            if len(file_bytes) == 0:
                return make_error_response(
                    code="EMPTY_FILE",
                    message="Uploaded PDF file is completely empty.",
                    status_code=422,
                )

            if len(file_bytes) > 5 * 1024 * 1024:
                return make_error_response(
                    code="FILE_TOO_LARGE",
                    message="Uploaded PDF exceeds the 5MB size limit.",
                    status_code=422,
                )

            report = service.analyze(
                resume_source=file_bytes,
                jd_text=jd_text,
                filename=filename,
                is_raw_text=False,
                persist=True,
            )
            return jsonify(report.to_dict()), 200

        # 2. JSON request
        if request.is_json:
            data = request.get_json() or {}
            validated = AnalyzeTextRequest(**data)

            report = service.analyze(
                resume_source=validated.resume_text,
                jd_text=validated.jd_text,
                filename=validated.filename,
                candidate_name=validated.candidate_name,
                is_raw_text=True,
                persist=True,
            )
            return jsonify(report.to_dict()), 200

        return make_error_response(
            code="UNSUPPORTED_MEDIA_TYPE",
            message="Request must be multipart/form-data or application/json.",
            status_code=415,
        )

    except ValidationError as ve:
        return make_error_response(
            code="VALIDATION_ERROR",
            message="Invalid request payload structure.",
            status_code=422,
            details=ve.errors(),
        )
    except UnsupportedDocument as ud:
        return make_error_response(
            code=f"UNSUPPORTED_DOCUMENT_{ud.reason.upper()}",
            message=ud.message,
            status_code=422,
            details={"reason": ud.reason},
        )
    except Exception as exc:
        logger.exception("Unexpected error in /api/analyze: %s", exc)
        return make_error_response(
            code="INTERNAL_ERROR",
            message="An unexpected server error occurred during analysis.",
            status_code=500,
            details=str(exc),
        )


@api_bp.route("/rank", methods=["POST"])
@limiter.limit("15 per minute")
def rank_candidates() -> Any:
    """Batch rank multiple candidate resumes against a single job description."""
    if not request.is_json:
        return make_error_response(
            code="UNSUPPORTED_MEDIA_TYPE",
            message="Request must be application/json.",
            status_code=415,
        )

    try:
        data = request.get_json() or {}
        validated = RankRequest(**data)
        service = MatchingService.get_instance()

        ranking = service.rank_candidates(
            jd_text=validated.jd_text,
            candidates=[c.model_dump() for c in validated.candidates],
        )
        return jsonify({"count": len(ranking), "ranking": ranking}), 200

    except ValidationError as ve:
        return make_error_response(
            code="VALIDATION_ERROR",
            message="Invalid candidate ranking request.",
            status_code=422,
            details=ve.errors(),
        )
    except Exception as exc:
        logger.exception("Error in /api/rank: %s", exc)
        return make_error_response(
            code="INTERNAL_ERROR",
            message=str(exc),
            status_code=500,
        )


@api_bp.route("/reports/<report_id>", methods=["GET"])
def get_saved_report(report_id: str) -> Any:
    """Retrieve an existing explainable fit report by unique identifier."""
    service = MatchingService.get_instance()
    report = service.get_report(report_id)
    if not report:
        return make_error_response(
            code="REPORT_NOT_FOUND",
            message=f"No matching report found with id '{report_id}'.",
            status_code=404,
        )
    return jsonify(report), 200


@api_bp.route("/suggest-jobs", methods=["POST"])
@limiter.limit("20 per minute")
def suggest_jobs() -> Any:
    """Suggest top matching job roles for a candidate profile."""
    if not request.is_json:
        return make_error_response(
            code="UNSUPPORTED_MEDIA_TYPE",
            message="Request must be application/json.",
            status_code=415,
        )

    try:
        data = request.get_json() or {}
        validated = SuggestJobsRequest(**data)
        service = MatchingService.get_instance()

        # Built-in reference roles for reverse-matching recommendation
        reference_jobs = [
            {
                "id": "job-py-lead",
                "title": "Senior Python & Cloud Architect",
                "description": "Seeking Python expert with AWS, Docker, Kubernetes, microservices, and PostgreSQL experience.",
                "required_skills": ["Python", "AWS", "Docker", "Kubernetes", "PostgreSQL"],
                "min_years": 5,
            },
            {
                "id": "job-ml-eng",
                "title": "Machine Learning Engineer",
                "description": "Develop NLP and deep learning models with PyTorch, scikit-learn, transformers, and MLOps pipelines.",
                "required_skills": ["Python", "PyTorch", "scikit-learn", "Machine Learning", "Docker"],
                "min_years": 3,
            },
            {
                "id": "job-fullstack",
                "title": "Full-Stack Software Engineer",
                "description": "Build high-performance web applications with React, TypeScript, Node.js, and relational databases.",
                "required_skills": ["JavaScript", "TypeScript", "React", "Node.js", "SQL"],
                "min_years": 2,
            },
            {
                "id": "job-devops",
                "title": "DevOps & Infrastructure Engineer",
                "description": "Manage CI/CD pipelines, Terraform infrastructure as code, Kubernetes clusters, and monitoring.",
                "required_skills": ["Docker", "Kubernetes", "CI/CD", "Linux", "Terraform"],
                "min_years": 3,
            },
        ]

        scored_jobs = []
        for job in reference_jobs[: validated.top_k]:
            rep = service.analyze(
                resume_source=validated.resume_text,
                jd_text=f"{job['title']}\n\n{job['description']}",
                filename="profile.txt",
                candidate_name=validated.candidate_name,
                is_raw_text=True,
                persist=False,
            )
            scored_jobs.append(
                {
                    "job_id": job["id"],
                    "title": job["title"],
                    "fit_index": rep.fit_index,
                    "verdict": rep.verdict_headline,
                    "matched_skills": rep.skills_analysis["matched_required_skills"],
                    "total_skills": rep.skills_analysis["total_required_skills"],
                }
            )

        scored_jobs.sort(key=lambda j: j["fit_index"], reverse=True)
        return jsonify({"count": len(scored_jobs), "jobs": scored_jobs}), 200

    except ValidationError as ve:
        return make_error_response(
            code="VALIDATION_ERROR",
            message="Invalid request structure.",
            status_code=422,
            details=ve.errors(),
        )
    except Exception as exc:
        logger.exception("Error in /api/suggest-jobs: %s", exc)
        return make_error_response(code="INTERNAL_ERROR", message=str(exc), status_code=500)


@api_bp.route("/resumes/<candidate_id>", methods=["DELETE"])
def delete_candidate(candidate_id: str) -> Any:
    """Right-to-Erasure: permanently remove all candidate data and associated reports."""
    service = MatchingService.get_instance()
    success = service.delete_resume_data(candidate_id)
    if not success:
        return make_error_response(
            code="NOT_FOUND",
            message=f"No candidate or report found with id '{candidate_id}'.",
            status_code=404,
        )
    resp = DeleteResponse(
        deleted=True,
        id=candidate_id,
        message="Candidate record and all derived audit vectors permanently erased.",
    )
    return jsonify(resp.model_dump()), 200


@api_bp.route("/eval/latest", methods=["GET"])
def get_evaluation_metrics() -> Any:
    """Return latest benchmark metrics and model configurations."""
    service = MatchingService.get_instance()
    return jsonify(service.get_eval_summary()), 200
