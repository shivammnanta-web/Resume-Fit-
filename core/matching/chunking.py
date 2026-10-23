"""Token-aware section-based document chunking module (Task P3.2).

Splits structured resume sections into bounded semantic chunks (<= 200 tokens,
with overlap) while strictly preserving section boundaries. Section semantics
(Experience, Projects, Education, Skills) are never mixed within a single chunk.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.ingestion.types import Section, SectionType
from resumefit.config import get_config

logger = logging.getLogger(__name__)

# Default fast word/subword approximation pattern if heavy model tokenizer is not passed
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def simple_tokenize(text: str) -> list[str]:
    """Fast regex-based tokenizer approximating word/subword token counts."""
    return TOKEN_PATTERN.findall(text)


@dataclass
class Chunk:
    """A bounded semantic text chunk extracted from a single section."""

    chunk_id: str
    section_type: SectionType
    text: str
    token_count: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_type": self.section_type.value,
            "text": self.text,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class SectionChunker:
    """Token-aware chunker respecting section boundaries and token limits."""

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        tokenize_fn: Callable[[str], list[Any]] | None = None,
    ) -> None:
        cfg = get_config().matching
        self.max_tokens = max_tokens if max_tokens is not None else cfg.chunk_size_tokens
        self.overlap_tokens = (
            overlap_tokens if overlap_tokens is not None else cfg.chunk_overlap_tokens
        )
        self.tokenize_fn = tokenize_fn or simple_tokenize

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured tokenizer."""
        return len(self.tokenize_fn(text))

    def split_into_units(self, text: str) -> list[str]:
        """Split section text into natural atomic units (bullets, lines, or sentences)."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        units: list[str] = []
        for line in lines:
            # If line is very long, split on sentence terminators (.!?)
            if self.count_tokens(line) > self.max_tokens:
                sentences = re.split(r"(?<=[.!?])\s+", line)
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean:
                        units.append(s_clean)
            else:
                units.append(line)
        return units

    def chunk_section(
        self,
        section: Section,
        section_idx: int = 0,
    ) -> list[Chunk]:
        """Chunk a single section into windows <= max_tokens with overlap."""
        text = section.text.strip()
        if not text:
            return []

        total_tokens = self.count_tokens(text)
        if total_tokens <= self.max_tokens:
            return [
                Chunk(
                    chunk_id=f"sec_{section_idx}_{section.section_type.value}_0",
                    section_type=section.section_type,
                    text=text,
                    token_count=total_tokens,
                    start_char=section.start_char,
                    end_char=section.end_char,
                    metadata={"heading": section.heading},
                )
            ]

        # Break section into units (bullets / sentences)
        units = self.split_into_units(text)
        if not units:
            return []

        chunks: list[Chunk] = []
        current_unit_idx = 0
        chunk_counter = 0

        while current_unit_idx < len(units):
            window_units: list[str] = []
            window_tokens = 0
            start_unit_idx = current_unit_idx

            while current_unit_idx < len(units):
                unit = units[current_unit_idx]
                unit_tokens = self.count_tokens(unit)

                # Case: single unit exceeds max_tokens on its own -> hard word split
                if unit_tokens > self.max_tokens:
                    if window_units:
                        # Flush existing window first
                        break

                    words = unit.split()
                    sub_start = 0
                    while sub_start < len(words):
                        sub_words: list[str] = []
                        sub_toks = 0
                        while sub_start < len(words):
                            w = words[sub_start]
                            w_tok = self.count_tokens(w)
                            if sub_toks + w_tok > self.max_tokens and sub_words:
                                break
                            sub_words.append(w)
                            sub_toks += w_tok
                            sub_start += 1

                        sub_text = " ".join(sub_words)
                        chunks.append(
                            Chunk(
                                chunk_id=f"sec_{section_idx}_{section.section_type.value}_{chunk_counter}",
                                section_type=section.section_type,
                                text=sub_text,
                                token_count=self.count_tokens(sub_text),
                                start_char=section.start_char,
                                end_char=section.end_char,
                                metadata={"heading": section.heading, "sub_split": True},
                            )
                        )
                        chunk_counter += 1

                    current_unit_idx += 1
                    break

                if window_tokens + unit_tokens > self.max_tokens:
                    break

                window_units.append(unit)
                window_tokens += unit_tokens
                current_unit_idx += 1

            if window_units:
                chunk_text = "\n".join(window_units)
                chunks.append(
                    Chunk(
                        chunk_id=f"sec_{section_idx}_{section.section_type.value}_{chunk_counter}",
                        section_type=section.section_type,
                        text=chunk_text,
                        token_count=self.count_tokens(chunk_text),
                        start_char=section.start_char,
                        end_char=section.end_char,
                        metadata={"heading": section.heading},
                    )
                )
                chunk_counter += 1

                # If we've reached the end of units, stop
                if current_unit_idx >= len(units):
                    break

                # Apply overlap: rewind current_unit_idx back to cover overlap_tokens
                if self.overlap_tokens > 0 and current_unit_idx > start_unit_idx + 1:
                    overlap_acc = 0
                    rewind_idx = current_unit_idx - 1
                    while rewind_idx > start_unit_idx:
                        u_tok = self.count_tokens(units[rewind_idx])
                        if overlap_acc + u_tok > self.overlap_tokens:
                            break
                        overlap_acc += u_tok
                        rewind_idx -= 1
                    current_unit_idx = max(rewind_idx + 1, start_unit_idx + 1)

        return chunks

    def chunk_all_sections(self, sections: list[Section]) -> list[Chunk]:
        """Chunk all sections sequentially, maintaining strict section boundaries."""
        all_chunks: list[Chunk] = []
        for idx, sec in enumerate(sections):
            sec_chunks = self.chunk_section(sec, section_idx=idx)
            all_chunks.extend(sec_chunks)
        return all_chunks
