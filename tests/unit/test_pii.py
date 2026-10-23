"""Unit tests for PII detection and redaction."""

from core.ingestion.pii import redact_pii


def test_redact_email_phone_url() -> None:
    text = (
        "Candidate contact info:\n"
        "Email: rahul.sharma@example.org\n"
        "Phone: +91 9876543210\n"
        "Website: https://rahulsharma.dev\n"
        "GitHub: github.com/rahulsharma\n"
        "Summary: Passionate software engineer with experience in Python and Flask."
    )
    result = redact_pii(text)

    # Verify PII is scrubbed from redacted_text
    assert "rahul.sharma@example.org" not in result.redacted_text
    assert "[EMAIL]" in result.redacted_text
    assert "+91 9876543210" not in result.redacted_text
    assert "[PHONE]" in result.redacted_text
    assert "https://rahulsharma.dev" not in result.redacted_text
    assert "[URL]" in result.redacted_text

    # Verify original text is preserved
    assert "rahul.sharma@example.org" in result.original_text

    # Verify technical terms were NOT redacted
    assert "Python" in result.redacted_text
    assert "Flask" in result.redacted_text
