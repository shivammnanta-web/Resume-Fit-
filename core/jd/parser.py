"""Job Description (JD) parsing and requirement extraction module (Task P3.1).

Extracts:
1. Role title
2. Minimum years of experience requirement
3. Minimum education degree requirement (Bachelor, Master, PhD)
4. Segregated required vs. preferred/nice-to-have skill sets (using SkillExtractor)
5. Individual atomic requirement statement blocks for requirement-to-chunk matching
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.ingestion.types import Section, SectionType
from core.skills.extractor import ExtractedSkill, SkillExtractor

logger = logging.getLogger(__name__)

# Regex patterns for minimum years of experience
YEARS_EXP_PATTERNS = [
    re.compile(
        r"\b(\d{1,2})\+?\s*(?:to\s*\d{1,2})?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+[\w-]{1,20}){0,4}\s+experience\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bminimum\s+(?:of\s+)?(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bat\s+least\s+(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexperience:\s*(\d{1,2})\+?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})\+?\s*(?:years?|yrs?)\s+in\s+[\w\s-]{1,30}\b",
        re.IGNORECASE,
    ),
]

# Regex patterns for degree requirements
DEGREE_PATTERNS = [
    (re.compile(r"\b(?:ph\.?d\.?|doctorate|doctoral)\b", re.IGNORECASE), "phd"),
    (
        re.compile(
            r"\b(?:master'?s?|m\.?s\.?|m\.?tech|m\.?sc|postgraduate)\b",
            re.IGNORECASE,
        ),
        "master",
    ),
    (
        re.compile(
            r"\b(?:bachelor'?s?|b\.?s\.?|b\.?tech|b\.?e\.?|b\.?sc|undergraduate)\b",
            re.IGNORECASE,
        ),
        "bachelor",
    ),
]

# Section headers distinguishing Required from Preferred
REQUIRED_HEADER_PATTERN = re.compile(
    r"^(?:requirements|qualifications|must\s+have|required\s+skills?|minimum\s+qualifications?|what\s+you'?ll?\s+need|what\s+we\s+look\s+for)",
    re.IGNORECASE,
)
PREFERRED_HEADER_PATTERN = re.compile(
    r"^(?:preferred(?:\s+skills?|\s+qualifications?)?|nice\s+to\s+have|bonus(?:\s+points?)?|desirable|plus|good\s+to\s+have)",
    re.IGNORECASE,
)
RESPONSIBILITIES_HEADER_PATTERN = re.compile(
    r"^(?:responsibilities|duties|what\s+you'?ll?\s+do|role\s+overview|the\s+role)",
    re.IGNORECASE,
)


@dataclass
class ParsedJD:
    """Structured representation of a parsed job description."""

    raw_text: str
    role_title: str | None = None
    min_years_experience: float = 0.0
    min_degree: str | None = None  # "bachelor", "master", "phd", or None
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    all_skills: list[str] = field(default_factory=list)
    extracted_skill_details: list[ExtractedSkill] = field(default_factory=list)
    requirement_statements: list[str] = field(default_factory=list)
    responsibility_statements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_title": self.role_title,
            "min_years_experience": self.min_years_experience,
            "min_degree": self.min_degree,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "all_skills": self.all_skills,
            "requirement_statements": self.requirement_statements,
            "responsibility_statements": self.responsibility_statements,
        }


def extract_role_title(text: str) -> str | None:
    """Heuristic extraction of target job title from first lines of JD."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # Check for explicit label "Job Title: ..." or "Role: ..."
    for line in lines[:5]:
        m = re.match(r"^(?:job\s+title|role|position|title)\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Otherwise first line if it looks like a title (< 60 chars, not starting with about/company)
    first = lines[0]
    if len(first) <= 60 and not first.lower().startswith(("about", "we are", "welcome", "company")):
        return first.strip().strip(":").strip("-").strip()

    return None


def extract_min_years(text: str) -> float:
    """Extract minimum years of required professional experience."""
    max_years = 0.0
    for pattern in YEARS_EXP_PATTERNS:
        for match in pattern.finditer(text):
            val = float(match.group(1))
            # Sanity cap (e.g. 0 to 25 years)
            if 0.0 <= val <= 25.0 and val > max_years:
                max_years = val

    return max_years


def extract_min_degree(text: str) -> str | None:
    """Extract minimum educational degree requirement (hierarchical)."""
    # Check for PhD, Master, Bachelor in text
    for pattern, deg_name in DEGREE_PATTERNS:
        if pattern.search(text):
            return deg_name

    return None


def extract_statements_from_block(text: str) -> list[str]:
    """Break a section text block into distinct bullet points or non-empty sentences."""
    statements: list[str] = []
    for line in text.splitlines():
        s = line.strip().lstrip("•-*· \t").strip()
        if len(s) > 10:
            statements.append(s)
    return statements


def parse_job_description(
    text: str,
    extractor: SkillExtractor | None = None,
) -> ParsedJD:
    """Parse raw job description text into structured requirements and skills.

    Splits text into required and preferred blocks, runs skill extraction cascade,
    and calculates constraints (years of experience, degree level).
    """
    skill_extractor = extractor or SkillExtractor()

    cleaned_text = text.strip()
    role_title = extract_role_title(cleaned_text)
    min_years = extract_min_years(cleaned_text)
    min_degree = extract_min_degree(cleaned_text)

    # Segment JD into sections: General, Required, Preferred, Responsibilities
    lines = cleaned_text.splitlines()
    req_lines: list[str] = []
    pref_lines: list[str] = []
    resp_lines: list[str] = []
    current_mode = "required"  # default to required

    for line in lines:
        sline = line.strip().strip(":").strip("-").strip()
        if not sline:
            continue

        if PREFERRED_HEADER_PATTERN.match(sline):
            current_mode = "preferred"
            continue
        elif REQUIRED_HEADER_PATTERN.match(sline):
            current_mode = "required"
            continue
        elif RESPONSIBILITIES_HEADER_PATTERN.match(sline):
            current_mode = "responsibilities"
            continue

        if current_mode == "preferred":
            pref_lines.append(line)
        elif current_mode == "responsibilities":
            resp_lines.append(line)
        else:
            req_lines.append(line)

    req_text = "\n".join(req_lines)
    pref_text = "\n".join(pref_lines)
    resp_text = "\n".join(resp_lines)

    req_statements = extract_statements_from_block(req_text)
    resp_statements = extract_statements_from_block(resp_text)

    # Extract skills from required and preferred blocks
    req_section = Section(
        section_type=SectionType.SKILLS,
        heading="Requirements",
        text=req_text + "\n" + resp_text,
        start_char=0,
        end_char=len(req_text) + len(resp_text),
    )
    pref_section = Section(
        section_type=SectionType.SKILLS,
        heading="Preferred",
        text=pref_text,
        start_char=0,
        end_char=len(pref_text),
    )

    req_res = skill_extractor.extract_from_sections([req_section])
    pref_res = skill_extractor.extract_from_sections([pref_section])

    req_skills = req_res.canonical_names
    pref_skills = [s for s in pref_res.canonical_names if s not in req_skills]

    all_skills_dict: dict[str, ExtractedSkill] = {}
    for s in req_res.skills:
        all_skills_dict[s.canonical_label] = s
    for s in pref_res.skills:
        if s.canonical_label not in all_skills_dict:
            all_skills_dict[s.canonical_label] = s

    all_skills = list(all_skills_dict.keys())
    all_details = list(all_skills_dict.values())

    return ParsedJD(
        raw_text=cleaned_text,
        role_title=role_title,
        min_years_experience=min_years,
        min_degree=min_degree,
        required_skills=req_skills,
        preferred_skills=pref_skills,
        all_skills=all_skills,
        extracted_skill_details=all_details,
        requirement_statements=req_statements,
        responsibility_statements=resp_statements,
    )


class JDParser:
    """Class wrapper for job description parsing."""

    def __init__(self, skill_extractor: SkillExtractor | None = None) -> None:
        self.skill_extractor = skill_extractor

    def parse(self, text: str) -> ParsedJD:
        return parse_job_description(text, self.skill_extractor)

