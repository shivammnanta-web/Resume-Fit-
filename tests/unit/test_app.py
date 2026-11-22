"""Unit tests for Flask application factory, REST API, error envelopes, and web views."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from resumefit.app import create_app


@pytest.fixture
def client() -> FlaskClient:
    """Create test client with testing mode enabled."""
    app = create_app({"TESTING": True, "RATELIMIT_ENABLED": False})
    return app.test_client()


def test_health_check_endpoint(client: FlaskClient) -> None:
    """Verify /api/health returns 200 OK with expected JSON keys."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["app"] == "ResumeFit"
    assert data["database_connected"] is True


def test_analyze_json_request(client: FlaskClient) -> None:
    """Verify /api/analyze accepts JSON and returns an auditable FitReport."""
    payload = {
        "candidate_name": "Test Engineer",
        "resume_text": "Experienced Python developer with 4 years in FastAPI, PostgreSQL, and Docker.",
        "jd_text": "Looking for Python Engineer with experience in Docker and PostgreSQL.",
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()

    assert "fit_index" in data
    assert 0.0 <= data["fit_index"] <= 100.0
    assert "scores" in data
    assert "lexical_score" in data["scores"]
    assert "semantic_score" in data["scores"]
    assert "skills_analysis" in data
    assert "audit_metadata" in data


def test_analyze_multipart_pdf_upload(client: FlaskClient) -> None:
    """Verify /api/analyze processes valid PDF uploads via multipart/form-data."""
    pdf_path = Path("tests/fixtures/pdfs/single_col_1.pdf")
    if not pdf_path.is_file():
        pytest.skip("Test PDF fixture not found")

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    data = {
        "resume": (io.BytesIO(file_bytes), "sample_resume.pdf", "application/pdf"),
        "jd_text": "Seeking Python Software Developer with SQL and Git skills.",
    }
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    report = resp.get_json()
    assert "fit_index" in report
    assert report["candidate_name"] is not None


def test_error_envelope_on_validation_failure(client: FlaskClient) -> None:
    """Verify invalid payloads produce a standardized ErrorEnvelope with 422 status."""
    # Text too short for schema
    payload = {"resume_text": "short", "jd_text": "short"}
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 422
    data = resp.get_json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_rank_candidates_endpoint(client: FlaskClient) -> None:
    """Verify /api/rank endpoint compares and orders multiple candidate resumes."""
    payload = {
        "jd_text": "Senior Machine Learning Engineer with Python, PyTorch, and scikit-learn.",
        "candidates": [
            {
                "name": "ML Specialist",
                "text": "Experienced ML Engineer with deep knowledge of Python, PyTorch, transformers, and scikit-learn.",
            },
            {
                "name": "Generalist Dev",
                "text": "Frontend developer with JavaScript, CSS, HTML, and basic Python.",
            },
        ],
    }
    resp = client.post("/api/rank", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 2
    assert len(data["ranking"]) == 2

    # First candidate should have higher Fit Index than generalist
    top_cand = data["ranking"][0]
    second_cand = data["ranking"][1]
    assert top_cand["candidate_name"] == "ML Specialist"
    assert top_cand["fit_index"] >= second_cand["fit_index"]


def test_right_to_erasure_api(client: FlaskClient) -> None:
    """Verify DELETE /api/resumes/<id> cascaded erasure."""
    # First create an analysis to store candidate & report
    payload = {
        "candidate_name": "Erase Me",
        "resume_text": "DevOps engineer with 5 years experience in Kubernetes, Docker, AWS, and CI/CD.",
        "jd_text": "Seeking DevOps Specialist with Docker and Kubernetes experience.",
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    report = resp.get_json()
    report_id = report.get("id")

    assert report_id is not None
    # Delete by ID
    del_resp = client.delete(f"/api/resumes/{report_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.get_json()
    assert del_data["deleted"] is True

    # Second deletion should return 404
    second_del = client.delete(f"/api/resumes/{report_id}")
    assert second_del.status_code == 404


def test_web_routes_render_html(client: FlaskClient) -> None:
    """Verify all web templates render properly with 200 status code."""
    # 1. Home / Matcher view
    resp_home = client.get("/")
    assert resp_home.status_code == 200
    assert b"Explainable, Auditable Resume Matching" in resp_home.data

    # 2. Compare view
    resp_compare = client.get("/compare")
    assert resp_compare.status_code == 200
    assert b"Side-by-Side Candidate Comparison" in resp_compare.data

    # 3. Dashboard view
    resp_dash = client.get("/dashboard")
    assert resp_dash.status_code == 200
    assert b"Candidate Pool" in resp_dash.data

    # 4. Verify removed /eval page returns 404
    resp_eval = client.get("/eval")
    assert resp_eval.status_code == 404


def test_web_form_analyze_and_report_view(client: FlaskClient) -> None:
    """Verify HTML form submission on /analyze redirects and renders match report view."""
    form_data = {
        "candidate_name": "Web Form Tester",
        "resume_text": "Full stack engineer with Python, Flask, HTML, CSS, and PostgreSQL skills.",
        "jd_text": "Seeking Python and Flask Web Developer with PostgreSQL experience.",
    }
    resp = client.post("/analyze", data=form_data, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Web Form Tester" in resp.data
    assert b"FIT INDEX" in resp.data or b"Fit Index" in resp.data


def test_web_compare_form_submission(client: FlaskClient) -> None:
    """Verify HTML form submission on /compare renders side-by-side comparison."""
    form_data = {
        "jd_text": "Backend Engineer with Python and SQL experience.",
        "cand1_name": "Alice",
        "cand1_text": "Python and SQL developer with 5 years backend experience.",
        "cand2_name": "Bob",
        "cand2_text": "Frontend designer with CSS and HTML experience.",
    }
    resp = client.post("/compare", data=form_data)
    assert resp.status_code == 200
    assert b"Ranked Evaluation Matrix" in resp.data
    assert b"Alice" in resp.data


def test_web_compare_with_pdf_upload(client: FlaskClient) -> None:
    """Verify HTML form submission on /compare handles PDF resume upload alongside text."""
    pdf_path = Path("tests/fixtures/pdfs/single_col_1.pdf")
    if not pdf_path.is_file():
        pytest.skip("Test PDF fixture not found")

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    form_data = {
        "jd_text": "Python Software Developer with SQL and database experience.",
        "cand1_name": "PDF Candidate",
        "cand1_file": (io.BytesIO(file_bytes), "sample.pdf", "application/pdf"),
        "cand2_name": "Text Candidate",
        "cand2_text": "Frontend designer with CSS and HTML experience.",
    }
    resp = client.post("/compare", data=form_data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Ranked Evaluation Matrix" in resp.data
    assert b"PDF Candidate" in resp.data


def test_api_eval_removed(client: FlaskClient) -> None:
    """Verify /api/eval/latest is removed and returns 404."""
    resp = client.get("/api/eval/latest")
    assert resp.status_code == 404


def test_api_reports_not_found(client: FlaskClient) -> None:
    """Verify GET /api/reports/<nonexistent> returns 404 envelope."""
    resp = client.get("/api/reports/nonexistent-uuid-123")
    assert resp.status_code == 404
    data = resp.get_json()
    assert "error" in data


def test_api_suggest_jobs_endpoint(client: FlaskClient) -> None:
    """Verify /api/suggest-jobs returns top matched job descriptions."""
    payload = {
        "resume_text": "Data scientist with Python, pandas, machine learning, and statistical analysis experience.",
        "top_k": 3,
    }
    resp = client.post("/api/suggest-jobs", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "jobs" in data


def test_custom_404_error_page(client: FlaskClient) -> None:
    """Verify custom 404 error page renders cleanly."""
    resp = client.get("/this-route-does-not-exist-at-all")
    assert resp.status_code == 404
    assert b"404" in resp.data


def test_date_parsing_formats_and_pre_1990_experience() -> None:
    """Verify YYYY/MM, MM/YYYY, and pre-1990 experience calculation."""
    from core.ingestion.sections import (
        DateRange,
        compute_total_experience_years,
        parse_single_date,
    )

    # 1. YYYY/MM vs MM/YYYY
    y1, m1, _ = parse_single_date("2023/05")
    assert y1 == 2023 and m1 == 5

    y2, m2, _ = parse_single_date("05/2023")
    assert y2 == 2023 and m2 == 5

    # 2. Pre-1990 career timeline
    dr1 = DateRange(
        start_str="1984",
        end_str="1989",
        start_year=1984,
        start_month=1,
        end_year=1989,
        end_month=12,
        duration_months=72.0,
    )
    total_exp = compute_total_experience_years([dr1])
    assert total_exp == 6.0


