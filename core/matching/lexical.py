"""Lexical matching module using pre-fitted TF-IDF reference corpus (Task P3.4).

Enforces:
- Non-Negotiable Principle 3: TF-IDF Reference Corpus. Uses a globally pre-fitted
  vectorizer to avoid unstable IDF estimation from 2-document ad-hoc fitting.
- Custom tokenizer preserving tech tokens (C++, C#, .NET, Node.js, CI/CD, etc.).
- Transparent attribution: returns exact matched terms and missing JD keywords.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path("data/artifacts/tfidf_reference.joblib")

# Technical terms and compound tokens to preserve with punctuation
TECH_TOKEN_RE = re.compile(
    r"""
    \b[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+     # node.js, vue.js, next.js, v1.2
    |\b\.net\b                             # .net
    |\b[a-zA-Z]+(?:\+\+|\#)                # c++, c#
    |\b[a-zA-Z0-9]+(?:[-/][a-zA-Z0-9]+)+  # ci/cd, tcp/ip, scikit-learn, rest-api
    |\b[a-zA-Z0-9_]{2,}\b                 # standard words (length >= 2)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Standard minimal stop words that do not collide with tech tokens
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


def tech_tokenize(text: str) -> list[str]:
    """Tokenize text while preserving technical keywords and acronyms."""
    tokens: list[str] = []
    for match in TECH_TOKEN_RE.finditer(text):
        tok = match.group(0).lower().strip()
        if tok and tok not in STOP_WORDS:
            tokens.append(tok)
    return tokens


@dataclass
class MatchedTerm:
    """A matched term between resume and JD with TF-IDF weight and attribution."""

    term: str
    resume_weight: float
    jd_weight: float
    contribution: float


@dataclass
class LexicalMatchResult:
    """Result of lexical TF-IDF cosine comparison."""

    score: float  # [0.0, 1.0]
    matched_terms: list[MatchedTerm] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "matched_terms": [
                {
                    "term": m.term,
                    "resume_weight": round(m.resume_weight, 4),
                    "jd_weight": round(m.jd_weight, 4),
                    "contribution": round(m.contribution, 4),
                }
                for m in self.matched_terms
            ],
            "missing_terms": self.missing_terms[:15],
        }


class LexicalScorer:
    """Computes TF-IDF cosine similarity against pre-fitted reference vocabulary."""

    def __init__(self, artifact_path: Path | str | None = None) -> None:
        self.artifact_path = Path(artifact_path or ARTIFACT_PATH)
        self.vectorizer: TfidfVectorizer = self._load_or_build_vectorizer()

    def _load_or_build_vectorizer(self) -> TfidfVectorizer:
        if self.artifact_path.exists():
            try:
                vec = joblib.load(self.artifact_path)
                logger.info("Loaded reference TF-IDF vectorizer from %s", self.artifact_path)
                return vec
            except Exception as e:
                logger.warning("Failed to load %s: %s. Building default fallback.", self.artifact_path, e)

        # Fallback: create fresh vectorizer and fit on seed reference corpus
        from scripts.build_tfidf_reference import build_reference_corpus

        corpus = build_reference_corpus()
        tfidf_kwargs: dict[str, Any] = {
            "tokenizer": tech_tokenize,
            "token_pattern": None,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
        }
        vec = TfidfVectorizer(**tfidf_kwargs)
        vec.fit(corpus)
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vec, self.artifact_path)
        logger.info("Fitted and saved reference TF-IDF vectorizer to %s", self.artifact_path)
        return vec

    @property
    def vocabulary_size(self) -> int:
        """Return the number of terms in the fitted reference vocabulary."""
        return len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") else 0

    def score(self, resume_text: str, jd_text: str, top_k_terms: int = 10) -> LexicalMatchResult:
        """Compute cosine similarity and term attribution between resume and JD."""
        if not resume_text.strip() or not jd_text.strip():
            return LexicalMatchResult(score=0.0)

        # Transform using pre-fitted vocabulary (IDF weights are frozen)
        tfidf_matrix = self.vectorizer.transform([resume_text, jd_text])
        v_resume = tfidf_matrix[0].toarray()[0]
        v_jd = tfidf_matrix[1].toarray()[0]

        norm_resume = float(np.linalg.norm(v_resume))
        norm_jd = float(np.linalg.norm(v_jd))

        if norm_resume == 0.0 or norm_jd == 0.0:
            return LexicalMatchResult(score=0.0)

        # Cosine similarity
        cosine_sim = float(np.dot(v_resume, v_jd) / (norm_resume * norm_jd))
        cosine_sim = max(0.0, min(1.0, cosine_sim))

        # Attribution: per-term contribution to normalized dot product
        elementwise_contrib = (v_resume * v_jd) / (norm_resume * norm_jd)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Top matched terms
        nz_indices = np.where(elementwise_contrib > 0.0)[0]
        matched_terms: list[MatchedTerm] = []
        if len(nz_indices) > 0:
            sorted_nz = nz_indices[np.argsort(-elementwise_contrib[nz_indices])]
            for idx in sorted_nz[:top_k_terms]:
                matched_terms.append(
                    MatchedTerm(
                        term=str(feature_names[idx]),
                        resume_weight=float(v_resume[idx]),
                        jd_weight=float(v_jd[idx]),
                        contribution=float(elementwise_contrib[idx]),
                    )
                )

        # Missing terms: in JD with high weight, but 0 in resume
        missing_indices = np.where((v_jd > 0.0) & (v_resume == 0.0))[0]
        missing_terms: list[str] = []
        if len(missing_indices) > 0:
            sorted_missing = missing_indices[np.argsort(-v_jd[missing_indices])]
            missing_terms = [str(feature_names[idx]) for idx in sorted_missing[:15]]

        return LexicalMatchResult(
            score=cosine_sim,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
        )
