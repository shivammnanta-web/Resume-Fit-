"""Structured, PII-free logging configuration for ResumeFit.

Enforces Non-Negotiable Principle 5:
"Resumes are processed in memory; persistence is opt-in; PII (names, emails, phones,
addresses) is redacted before embedding and never written to logs."
"""

from __future__ import annotations

import logging
import re
import sys

try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:
    from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[attr-defined]

# Regex patterns for scrubbing sensitive PII from log messages
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")


class PIIScrubFilter(logging.Filter):
    """Filter that intercepts and scrubs PII from log message records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            msg = EMAIL_REGEX.sub("[REDACTED_EMAIL]", msg)
            msg = PHONE_REGEX.sub("[REDACTED_PHONE]", msg)
            msg = SSN_REGEX.sub("[REDACTED_SSN]", msg)
            msg = CARD_REGEX.sub("[REDACTED_CARD]", msg)
            record.msg = msg

        # If args were passed, scrub any string args as well
        if record.args:
            scrubbed_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    arg = EMAIL_REGEX.sub("[REDACTED_EMAIL]", arg)
                    arg = PHONE_REGEX.sub("[REDACTED_PHONE]", arg)
                    arg = SSN_REGEX.sub("[REDACTED_SSN]", arg)
                    arg = CARD_REGEX.sub("[REDACTED_CARD]", arg)
                scrubbed_args.append(arg)
            record.args = tuple(scrubbed_args)

        return True


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    service_name: str = "resumefit",
) -> None:
    """Configure the root logger with PII scrubbing and structured formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to prevent duplicate lines
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.addFilter(PIIScrubFilter())

    formatter: logging.Formatter
    if log_format.lower() == "json":
        formatter = JsonFormatter(
            fmt=f"%(asctime)s %(levelname)s %(name)s %(module)s %(message)s [service={service_name}]",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter(
            fmt=f"[{service_name}] [%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Silence excessively verbose external loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
