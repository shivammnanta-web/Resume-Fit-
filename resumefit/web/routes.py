"""Web view routes and HTML rendering for ResumeFit."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from core.ingestion.types import UnsupportedDocument
from resumefit.services import MatchingService

logger = logging.getLogger(__name__)

web_bp = Blueprint("web", __name__)


@web_bp.route("/", methods=["GET"])
def index() -> str:
    """Home page: dual-input matching interface with presets."""
    return render_template("index.html")


@web_bp.route("/analyze", methods=["POST"])
def analyze_view() -> Any:
    """Form submission handler for resume matching."""
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

            service.analyze(
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

            service.analyze(
                resume_source=resume_text,
                jd_text=jd_text,
                filename="pasted_resume.txt",
                candidate_name=candidate_name,
                is_raw_text=True,
                persist=True,
            )

        # Find latest report ID from service
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
        jd_text = request.form.get("jd_text", "").strip()
        cand1_name = request.form.get("cand1_name", "Candidate A").strip()
        cand1_text = request.form.get("cand1_text", "").strip()
        cand2_name = request.form.get("cand2_name", "Candidate B").strip()
        cand2_text = request.form.get("cand2_text", "").strip()
        cand3_name = request.form.get("cand3_name", "").strip()
        cand3_text = request.form.get("cand3_text", "").strip()

        candidates = []
        if cand1_text:
            candidates.append({"name": cand1_name, "text": cand1_text, "filename": f"{cand1_name}.txt"})
        if cand2_text:
            candidates.append({"name": cand2_name, "text": cand2_text, "filename": f"{cand2_name}.txt"})
        if cand3_text:
            candidates.append({"name": cand3_name, "text": cand3_text, "filename": f"{cand3_name}.txt"})

        if len(candidates) < 2:
            flash("Please enter at least two candidates to compare.", "error")
            return render_template("compare.html", ranking=None)

        ranking = service.rank_candidates(jd_text, candidates)
        return render_template("compare.html", ranking=ranking, jd_text=jd_text)

    return render_template("compare.html", ranking=None)


@web_bp.route("/dashboard", methods=["GET"])
def dashboard() -> str:
    """Candidate pool dashboard for search, filtering, and right-to-erasure management."""
    service = MatchingService.get_instance()
    candidates = service.list_candidates()
    return render_template("dashboard.html", candidates=candidates)


@web_bp.route("/eval", methods=["GET"])
def evaluation_dashboard() -> str:
    """Academic benchmark and metrics evaluation dashboard."""
    service = MatchingService.get_instance()
    summary = service.get_eval_summary()
    return render_template("eval.html", eval_data=summary)
