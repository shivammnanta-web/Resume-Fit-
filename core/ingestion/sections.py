"""Section detection, character offset indexing, and experience date parsing.

Maps sections to normalized SectionType enums (Education, Experience, Projects, Skills,
Certifications, Summary, Personal Details) and parses experience duration with overlap merging.
"""

from __future__ import annotations

import re
from datetime import datetime

from core.ingestion.types import DateRange, ExperienceEntry, Section, SectionType

# Heading variant dictionary mapped to canonical SectionType
HEADING_PATTERNS = [
    (
        SectionType.SUMMARY,
        re.compile(
            r"^\s*(?:professional\s+summary|career\s+summary|career\s+objective|summary|profile|about\s+me|executive\s+summary)\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        SectionType.EXPERIENCE,
        re.compile(
            r"^\s*(?:work\s+experience|professional\s+experience|employment\s+history|experience|career\s+history|work\s+history|relevant\s+experience)\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        SectionType.PROJECTS,
        re.compile(
            r"^\s*(?:academic\s+projects|personal\s+projects|key\s+projects|selected\s+projects|projects)\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        SectionType.SKILLS,
        re.compile(
            r"^\s*(?:technical\s+skills|core\s+competencies|technologies|tools\s*(?:&|and)\s*technologies|skills\s*(?:&|and)\s*abilities|skills|areas\s+of\s+expertise|skill\s+set)\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        SectionType.EDUCATION,
        re.compile(
            r"^\s*(?:education(?:\s*(?:&|and)\s*qualifications)?|academic\s+background|degrees|qualifications|academic\s+qualifications|education)\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        SectionType.CERTIFICATIONS,
        re.compile(
            r"^\s*(?:certifications(?:\s*(?:&|and)\s*licenses)?|licenses(?:\s*(?:&|and)\s*certifications)?|certificates|professional\s+certifications|accreditations)\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        SectionType.PERSONAL_DETAILS,
        re.compile(
            r"^\s*(?:personal\s+details|contact\s+information|contact\s+info|contact|personal\s+info)\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    ),
]

# Month name mappings for date parsing
MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Regex to detect date intervals in experience entries
DATE_RANGE_REGEX = re.compile(
    r"\b((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?(?:19|20)\d{2}|\d{1,2}/(?:19|20)\d{2})\s*(?:–|—|-|to)\s*((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?(?:19|20)\d{2}|\d{1,2}/(?:19|20)\d{2}|present|current|ongoing|now)\b",
    re.IGNORECASE,
)


def match_heading_type(line: str) -> SectionType | None:
    """Check if a line matches any known resume section heading."""
    stripped = line.strip().strip(":").strip("-").strip()
    if not stripped or len(stripped) > 50:
        return None

    for stype, pattern in HEADING_PATTERNS:
        if pattern.match(stripped):
            return stype

    return None


def detect_sections(cleaned_text: str) -> list[Section]:
    """Segment cleaned resume text into typed sections with precise character spans."""
    lines = cleaned_text.splitlines(keepends=True)
    if not lines:
        return []

    # Identify all heading positions
    heading_matches: list[
        tuple[int, SectionType, str]
    ] = []  # (char_offset, SectionType, raw_heading)
    current_offset = 0

    for line in lines:
        stype = match_heading_type(line)
        if stype is not None:
            heading_matches.append((current_offset, stype, line.strip()))
        current_offset += len(line)

    sections: list[Section] = []

    # If text before first heading exists, categorize as PERSONAL_DETAILS or SUMMARY
    if heading_matches:
        first_offset = heading_matches[0][0]
        if first_offset > 0:
            preamble_text = cleaned_text[:first_offset].strip()
            if preamble_text:
                sections.append(
                    Section(
                        section_type=SectionType.PERSONAL_DETAILS,
                        heading="Header / Contact",
                        text=preamble_text,
                        start_char=0,
                        end_char=first_offset,
                    )
                )

        # Slice between consecutive headings
        for i in range(len(heading_matches)):
            start_off, sec_type, heading = heading_matches[i]
            # End offset is start of next heading or end of text
            if i + 1 < len(heading_matches):
                end_off = heading_matches[i + 1][0]
            else:
                end_off = len(cleaned_text)

            section_body = cleaned_text[start_off:end_off].strip()
            # Strip the heading line from the section body
            lines_in_body = section_body.splitlines(keepends=True)
            body_content = "".join(lines_in_body[1:]).strip() if lines_in_body else ""

            sections.append(
                Section(
                    section_type=sec_type,
                    heading=heading,
                    text=body_content,
                    start_char=start_off,
                    end_char=end_off,
                )
            )
    else:
        # Fallback: No recognized headings; treat entire document as OTHER
        sections.append(
            Section(
                section_type=SectionType.OTHER,
                heading="General Content",
                text=cleaned_text.strip(),
                start_char=0,
                end_char=len(cleaned_text),
            )
        )

    return sections


def parse_single_date(date_token: str) -> tuple[int | None, int | None, bool]:
    """Parse month and year from a single date string."""
    token = date_token.strip().lower()
    if token in {"present", "current", "ongoing", "now"}:
        now = datetime.now()
        return now.year, now.month, True

    # Pattern: MM/YYYY
    if "/" in token:
        parts = token.split("/")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[1]), int(parts[0]), False

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", token)
    year = int(year_match.group(1)) if year_match else None

    month = None
    for mname, mval in MONTHS.items():
        if re.search(rf"\b{mname}\b", token):
            month = mval
            break

    return year, month, False


def parse_date_range(start_str: str, end_str: str) -> DateRange:
    """Construct a DateRange object and compute its duration in months."""
    s_year, s_month, _ = parse_single_date(start_str)
    e_year, e_month, is_curr = parse_single_date(end_str)

    # Defaults if month is omitted
    start_m = s_month or 1
    end_m = e_month or 12

    duration = 0.0
    if s_year and e_year:
        months_diff = (e_year - s_year) * 12 + (end_m - start_m) + 1
        duration = max(0.0, float(months_diff))

    return DateRange(
        start_str=start_str,
        end_str=end_str,
        start_year=s_year,
        start_month=s_month,
        end_year=e_year,
        end_month=e_month,
        is_current=is_curr,
        duration_months=duration,
    )


def merge_overlapping_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping (start_month_index, end_month_index) intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[int, int]] = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end + 1:  # Continuous or overlapping
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    return merged


def compute_total_experience_years(ranges: list[DateRange]) -> float:
    """Calculate total non-overlapping years of experience across all parsed date ranges."""
    intervals: list[tuple[int, int]] = []
    base_year = 1990

    for r in ranges:
        if r.start_year and r.end_year:
            s_idx = (r.start_year - base_year) * 12 + (r.start_month or 1)
            e_idx = (r.end_year - base_year) * 12 + (r.end_month or 12)
            if e_idx >= s_idx:
                intervals.append((s_idx, e_idx))

    merged = merge_overlapping_intervals(intervals)
    total_months = sum((e - s + 1) for s, e in merged)
    return round(total_months / 12.0, 1)


def _extract_title_company(line: str) -> tuple[str | None, str | None]:
    """Helper to parse role title and organization from a header line."""
    cleaned = DATE_RANGE_REGEX.sub("", line).strip()
    if not cleaned:
        return None, None

    if "|" in cleaned:
        parts = [p.strip() for p in cleaned.split("|") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], None
    elif " — " in cleaned or " - " in cleaned:
        sep = " — " if " — " in cleaned else " - "
        parts = [p.strip() for p in cleaned.split(sep) if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], None
    elif " at " in cleaned.lower():
        idx = cleaned.lower().find(" at ")
        return cleaned[:idx].strip(), cleaned[idx + 4 :].strip()

    return cleaned, None


def parse_experience_entries(experience_text: str) -> list[ExperienceEntry]:
    """Segment the Work Experience section into distinct ExperienceEntry items."""
    lines = experience_text.splitlines()
    entries: list[ExperienceEntry] = []

    current_entry_lines: list[str] = []
    current_date_range: DateRange | None = None
    current_title: str | None = None
    current_company: str | None = None

    for line in lines:
        stripped = line.strip()
        match = DATE_RANGE_REGEX.search(stripped)

        if match:
            start_s, end_s = match.group(1), match.group(2)
            new_date_range = parse_date_range(start_s, end_s)

            if current_date_range is not None:
                # We have a previous active entry. Peel off candidate header lines for the new entry.
                new_entry_header: list[str] = []
                while current_entry_lines:
                    last_line = current_entry_lines[-1].strip()
                    if not last_line:
                        current_entry_lines.pop()
                        continue
                    is_bullet = last_line.startswith(("•", "-", "*"))
                    is_long_desc = len(last_line) > 80 or last_line[-1] in {".", ";", "!"}
                    if is_bullet or is_long_desc:
                        break
                    new_entry_header.insert(0, current_entry_lines.pop())

                if current_entry_lines:
                    text_block = "\n".join(current_entry_lines).strip()
                    years = (
                        (current_date_range.duration_months / 12.0) if current_date_range else 0.0
                    )
                    entries.append(
                        ExperienceEntry(
                            title=current_title,
                            company=current_company,
                            date_range=current_date_range,
                            text=text_block,
                            years=round(years, 1),
                        )
                    )

                current_entry_lines = new_entry_header + [stripped]
                current_date_range = new_date_range

                # Determine title & company for the new entry
                current_title, current_company = None, None
                for hline in new_entry_header:
                    t, c = _extract_title_company(hline)
                    if t or c:
                        current_title, current_company = t, c
                        break
                if not current_title and not current_company:
                    current_title, current_company = _extract_title_company(stripped)
            else:
                # First date range found in section
                current_date_range = new_date_range
                for hline in current_entry_lines:
                    t, c = _extract_title_company(hline)
                    if t or c:
                        current_title, current_company = t, c
                        break
                if not current_title and not current_company:
                    current_title, current_company = _extract_title_company(stripped)
                current_entry_lines.append(stripped)
        else:
            current_entry_lines.append(stripped)

    if current_entry_lines:
        text_block = "\n".join(current_entry_lines).strip()
        years = (current_date_range.duration_months / 12.0) if current_date_range else 0.0
        entries.append(
            ExperienceEntry(
                title=current_title,
                company=current_company,
                date_range=current_date_range,
                text=text_block,
                years=round(years, 1),
            )
        )

    return entries
