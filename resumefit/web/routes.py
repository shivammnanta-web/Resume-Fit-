"""Web view routes and HTML rendering for ResumeFit."""

from __future__ import annotations

import hmac
import logging
from typing import Any

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from core.ingestion.types import UnsupportedDocument
from resumefit.services import MatchingService

logger = logging.getLogger(__name__)

web_bp = Blueprint("web", __name__)


def validate_csrf() -> bool:
    """Validate CSRF token on state-changing web form POSTs."""
    if current_app.config.get("TESTING", False) or not current_app.config.get("CSRF_ENABLED", True):
        return True
    expected = session.get("_csrf_token")
    provided = request.form.get("csrf_token")
    return bool(expected and provided and hmac.compare_digest(expected, provided))


@web_bp.route("/", methods=["GET"])
def index() -> str:
    """Home page: dual-input matching interface with presets."""
    return render_template("index.html")


@web_bp.route("/analyze", methods=["POST"])
def analyze_view() -> Any:
    """Form submission handler for resume matching."""
    if not validate_csrf():
        flash("Invalid or expired session security token (CSRF). Please resubmit.", "error")
        return redirect(url_for("web.index"))

    service = MatchingService.get_instance()
    jd_text = request.form.get("jd_text", "").strip()
    candidate_name = request.form.get("candidate_name", "").strip() or None

    if not jd_text:
        flash("Please provide a job description.", "error")
        return redirect(url_for("web.index"))

    try:
        # Check PDF file upload
        if "resume_file" in request.files and request.files["resume_file"].filename:
            file_obj = request.files["resume_file"]
            fname = secure_filename(file_obj.filename or "resume.pdf")
            file_bytes = file_obj.read()
            if not file_bytes:
                flash("Uploaded PDF file was empty.", "error")
                return redirect(url_for("web.index"))

            report = service.analyze(
                resume_source=file_bytes,
                jd_text=jd_text,
                filename=fname,
                candidate_name=candidate_name,
                is_raw_text=False,
                persist=True,
            )
        else:
            # Check pasted resume text
            resume_text = request.form.get("resume_text", "").strip()
            if not resume_text:
                flash("Please upload a PDF resume or paste resume text.", "error")
                return redirect(url_for("web.index"))

            report = service.analyze(
                resume_source=resume_text,
                jd_text=jd_text,
                filename="pasted_resume.txt",
                candidate_name=candidate_name,
                is_raw_text=True,
                persist=True,
            )

        report_id = report.id
        if not report_id:
            cand_list = service.list_candidates()
            report_id = cand_list[-1]["report_id"] if cand_list else "demo"

        return redirect(url_for("web.view_report", report_id=report_id))

    except UnsupportedDocument as ud:
        flash(f"Document rejected: {ud.message} (Reason: {ud.reason})", "error")
        return redirect(url_for("web.index"))
    except Exception as exc:
        logger.exception("Error during web analysis: %s", exc)
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("web.index"))


@web_bp.route("/report/<report_id>", methods=["GET"])
def view_report(report_id: str) -> str:
    """Render interactive explainable fit report with evidence span highlighting."""
    service = MatchingService.get_instance()
    report_data = service.get_report(report_id)
    if not report_data:
        abort(404)
    return render_template("report.html", report=report_data)


@web_bp.route("/compare", methods=["GET", "POST"])
def compare_candidates() -> Any:
    """Side-by-side comparison of 2-3 candidate resumes against a job description."""
    service = MatchingService.get_instance()

    if request.method == "POST":
        if not validate_csrf():
            flash("Invalid or expired session security token (CSRF). Please resubmit.", "error")
            return render_template("compare.html", ranking=None)

        jd_text = request.form.get("jd_text", "").strip()

        # Helper: resolve candidate input from either PDF upload or pasted text
        def _resolve_candidate(idx: int) -> tuple[str, str | None, bytes | None]:
            """Return (name, text_or_none, pdf_bytes_or_none) for candidate slot idx (1-3)."""
            name = request.form.get(f"cand{idx}_name", f"Candidate {idx}").strip()
            text = request.form.get(f"cand{idx}_text", "").strip() or None
            pdf_bytes: bytes | None = None
            file_key = f"cand{idx}_file"
            if file_key in request.files and request.files[file_key].filename:
                file_obj = request.files[file_key]
                pdf_bytes = file_obj.read()
                if not pdf_bytes:
                    pdf_bytes = None
            return name, text, pdf_bytes

        candidates: list[dict[str, Any]] = []
        cand_names: list[str] = []
        cand_texts: list[str] = []

        for idx in range(1, 4):
            name, text, pdf_bytes = _resolve_candidate(idx)
            cand_names.append(name)
            cand_texts.append(text or "")
            if pdf_bytes:
                # PDF upload path: run full ingestion on bytes
                fname = secure_filename(
                    request.files[f"cand{idx}_file"].filename or f"{name}.pdf"
                )
                try:
                    report = service.analyze(
                        resume_source=pdf_bytes,
                        jd_text=jd_text,
                        filename=fname,
                        candidate_name=name,
                        is_raw_text=False,
                        persist=True,
                    )
                    candidates.append({
                        "name": name,
                        "report": report,
                    })
                except UnsupportedDocument as ud:
                    flash(f"{name}: Document rejected — {ud.message}", "error")
                except Exception as exc:
                    logger.exception("Error processing PDF for %s: %s", name, exc)
                    flash(f"{name}: PDF processing failed — {exc}", "error")
            elif text:
                candidates.append({
                    "name": name,
                    "text": text,
                    "filename": f"{name}.txt",
                })

        if len(candidates) < 2:
            flash("Please provide at least two candidates (via PDF upload or pasted text).", "error")
            return render_template(
                "compare.html",
                ranking=None,
                jd_text=jd_text,
                cand1_name=cand_names[0] if len(cand_names) > 0 else "",
                cand1_text=cand_texts[0] if len(cand_texts) > 0 else "",
                cand2_name=cand_names[1] if len(cand_names) > 1 else "",
                cand2_text=cand_texts[1] if len(cand_texts) > 1 else "",
            )

        # Build ranking: some candidates may already have reports (PDF path)
        pre_analyzed = [c for c in candidates if "report" in c]
        text_candidates = [c for c in candidates if "report" not in c]

        # Rank text-only candidates through the standard path
        text_ranking = service.rank_candidates(jd_text, text_candidates) if text_candidates else []

        # Merge pre-analyzed PDF reports into ranking format
        ranking: list[dict[str, Any]] = list(text_ranking)
        for c in pre_analyzed:
            rep = c["report"]
            ranking.append({
                "candidate_name": c["name"],
                "filename": rep.id or "",
                "report_id": rep.id,
                "fit_index": rep.fit_index,
                "verdict": rep.verdict_headline,
                "lexical_score": rep.scores["lexical_score"],
                "semantic_score": rep.scores["semantic_score"],
                "matched_required": rep.skills_analysis["matched_required_skills"],
                "total_required": rep.skills_analysis["total_required_skills"],
                "skill_coverage_pct": rep.skills_analysis["skill_coverage_percentage"],
                "years_experience": rep.constraints["candidate_years_exp"],
                "report_dict": rep.to_dict(),
            })

        # Sort descending by Fit Index
        ranking.sort(key=lambda x: x["fit_index"], reverse=True)

        return render_template(
            "compare.html",
            ranking=ranking,
            jd_text=jd_text,
            cand1_name=cand_names[0] if len(cand_names) > 0 else "",
            cand1_text=cand_texts[0] if len(cand_texts) > 0 else "",
            cand2_name=cand_names[1] if len(cand_names) > 1 else "",
            cand2_text=cand_texts[1] if len(cand_texts) > 1 else "",
            cand3_name=cand_names[2] if len(cand_names) > 2 else "",
            cand3_text=cand_texts[2] if len(cand_texts) > 2 else "",
        )

    return render_template("compare.html", ranking=None)


@web_bp.route("/dashboard", methods=["GET"])
def dashboard() -> str:
    """Candidate pool dashboard for search, filtering, and right-to-erasure management."""
    service = MatchingService.get_instance()
    candidates = service.list_candidates()
    return render_template("dashboard.html", candidates=candidates)



