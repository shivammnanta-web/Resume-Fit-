"""Unit tests for text preprocessing functions."""

from core.ingestion.preprocess import (
    dehyphenate,
    normalize_bullets,
    rejoin_broken_lines,
    repair_unicode,
)


def test_repair_unicode_ligatures_and_quotes() -> None:
    text = "We built \ufb01rst-class soft\ufb02oats with “smart quotes” and ‘single’ quotes."
    repaired = repair_unicode(text)
    assert "first-class" in repaired
    assert "softfloats" in repaired
    assert '"smart quotes"' in repaired
    assert "'single'" in repaired


def test_dehyphenate() -> None:
    text = "We have imple-\nmented high-throughput sys-\ntems."
    cleaned = dehyphenate(text)
    assert "implemented" in cleaned
    assert "systems" in cleaned


def test_normalize_bullets() -> None:
    dirty = "• First item\n* Second item\n- Third item\n\u25cf Fourth item"
    norm = normalize_bullets(dirty)
    for line in norm.splitlines():
        assert line.startswith("• ")


def test_rejoin_broken_lines() -> None:
    fragmented = (
        "Architected and deployed distributed\n"
        "microservices using Python and PostgreSQL.\n"
        "• Scaled database throughput by 40%\n"
        "across all regions."
    )
    rejoined = rejoin_broken_lines(fragmented)
    assert "distributed microservices using Python and PostgreSQL." in rejoined
    assert "Scaled database throughput by 40% across all regions." in rejoined
