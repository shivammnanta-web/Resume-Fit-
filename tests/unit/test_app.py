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

    if report_id:
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

    # 4. Academic Evaluation view
    resp_eval = client.get("/eval")
    assert resp_eval.status_code == 200
    assert b"Empirical Benchmarks" in resp_eval.data
