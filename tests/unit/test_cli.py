"""Unit tests for ResumeFit CLI (core/cli.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from core.cli import (
    MatchPipeline,
    format_comparison_table,
    format_terminal_report,
    main,
)
from core.ingestion.types import UnsupportedDocument, UnsupportedReason


@pytest.fixture
def sample_pdf_path() -> Path:
    p = Path("tests/fixtures/pdfs/single_col_1.pdf")
    if not p.is_file():
        pytest.skip("Test PDF fixture not found")
    return p


def test_match_pipeline_run(sample_pdf_path: Path):
    pipeline = MatchPipeline()
    jd = (
        "Role: Software Engineer\n"
        "Requirements:\n"
        "- 3+ years experience with Python and SQL\n"
        "- Experience with Docker and REST APIs\n"
        "Degree: Bachelor of Science in Computer Science\n"
    )
    report = pipeline.run(sample_pdf_path, jd)
    assert report is not None
    assert report.scores["hybrid_raw"] >= 0.0
    assert report.fit_index >= 0.0

    terminal_out = format_terminal_report(report)
    assert "RESUMEFIT EXPLAINABLE MATCH REPORT" in terminal_out
    assert "FIT INDEX:" in terminal_out

    comp_table = format_comparison_table(report)
    assert "ARCHITECTURE COMPARISON" in comp_table
    assert "Lexical" in comp_table
    assert "Semantic" in comp_table


def test_cli_main_analyze(sample_pdf_path: Path, tmp_path: Path, monkeypatch, capsys):
    out_json = tmp_path / "report.json"
    jd = "Role: Python Developer\nRequirements: Python, SQL"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "analyze",
            "--resume",
            str(sample_pdf_path),
            "--jd",
            jd,
            "--output",
            str(out_json),
        ],
    )
    main()
    captured = capsys.readouterr()
    assert "RESUMEFIT EXPLAINABLE MATCH REPORT" in captured.out
    assert out_json.is_file()


def test_cli_main_compare(sample_pdf_path: Path, monkeypatch, capsys):
    jd = "Role: Python Developer\nRequirements: Python, SQL"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "compare",
            "--resume",
            str(sample_pdf_path),
            "--jd",
            jd,
        ],
    )
    main()
    captured = capsys.readouterr()
    assert "ARCHITECTURE COMPARISON" in captured.out


def test_cli_main_unsupported_document(monkeypatch, capsys):
    with patch.object(
        MatchPipeline,
        "run",
        side_effect=UnsupportedDocument(
            reason=UnsupportedReason.SCANNED_OR_IMAGE_ONLY,
            message="No selectable text layer",
        ),
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cli.py",
                "analyze",
                "--resume",
                "fake.pdf",
                "--jd",
                "Python Developer",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "[UNSUPPORTED DOCUMENT]" in captured.err
