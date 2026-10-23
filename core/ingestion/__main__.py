"""CLI entry point for testing the ingestion pipeline directly.

Usage:
    python -m core.ingestion path/to/resume.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from core.ingestion.pipeline import process_resume
from core.ingestion.types import UnsupportedDocument
from resumefit.logging import setup_logging


def main() -> None:
    setup_logging(level="WARNING", log_format="text")

    if len(sys.argv) < 2:
        print("Usage: python -m core.ingestion <path_to_resume.pdf>", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(json.dumps({"error": "file_not_found", "path": str(pdf_path)}), file=sys.stderr)
        sys.exit(1)

    try:
        result = process_resume(pdf_path)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    except UnsupportedDocument as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(json.dumps({"error": "unexpected_error", "message": str(exc)}), file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
