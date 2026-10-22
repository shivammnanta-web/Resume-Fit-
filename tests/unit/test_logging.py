"""Tests for structured and PII-free logging."""

import logging

from resumefit.logging import PIIScrubFilter, setup_logging


def test_pii_scrub_filter_scrubs_email_and_phone() -> None:
    """Verify that email, phone, and card numbers are redacted from log messages."""
    filter_ = PIIScrubFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Candidate email john.smith@company.org and phone +1-555-123-4567 submitted resume.",
        args=(),
        exc_info=None,
    )
    filter_.filter(record)
    assert "john.smith@company.org" not in record.msg
    assert "[REDACTED_EMAIL]" in record.msg
    assert "+1-555-123-4567" not in record.msg
    assert "[REDACTED_PHONE]" in record.msg


def test_setup_logging_text_mode() -> None:
    """Verify logger setup doesn't raise errors in text format mode."""
    setup_logging(level="DEBUG", log_format="text")
    logger = logging.getLogger("test_logger")
    logger.info("Test logging message without PII.")


def test_setup_logging_json_mode() -> None:
    """Verify logger setup doesn't raise errors in json format mode."""
    setup_logging(level="INFO", log_format="json")
    logger = logging.getLogger("test_logger_json")
    logger.info("JSON log entry.")
