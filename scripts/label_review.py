"""CLI review and audit utility for the 500 gold labeled Resume-JD pairs (Task P6.4)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

GOLD_LABELS_PATH = Path("data/gold/labels.jsonl")
SPLITS_PATH = Path("data/gold/splits.json")


def review_labels(limit: int = 10, filter_grade: int | None = None) -> None:
    """Print labeled pairs and summary distribution."""
    if not GOLD_LABELS_PATH.is_file():
        print(f"Error: {GOLD_LABELS_PATH} not found.")
        return

    pairs = [json.loads(line) for line in GOLD_LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(p["relevance"] for p in pairs)

    print("=" * 65)
    print("ResumeFit Gold Label Benchmark Dataset Audit")
    print("=" * 65)
    print(f"Total Labeled Pairs: {len(pairs)}")
    print(f"  Relevance 2 (Strong Match):   {counts[2]} ({counts[2]/len(pairs)*100:.1f}%)")
    print(f"  Relevance 1 (Partial Match):  {counts[1]} ({counts[1]/len(pairs)*100:.1f}%)")
    print(f"  Relevance 0 (Irrelevant):     {counts[0]} ({counts[0]/len(pairs)*100:.1f}%)")

    if SPLITS_PATH.is_file():
        splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
        print("\nDev/Test Partition:")
        print(f"  Dev Set:  {splits['dev_jd_count']} JDs ({splits['dev_jd_count']*10} pairs)")
        print(f"  Test Set: {splits['test_jd_count']} JDs ({splits['test_jd_count']*10} pairs)")

    print("-" * 65)
    print(f"Displaying sample pairs (limit={limit}):\n")

    shown = 0
    for p in pairs:
        if filter_grade is not None and p["relevance"] != filter_grade:
            continue
        grade_label = {2: "[STRONG 2]", 1: "[PARTIAL 1]", 0: "[IRRELEVANT 0]"}.get(p["relevance"], "[?]")
        print(f"Pair ID: {p['pair_id']}  Grade: {grade_label}")
        print(f"Candidate: {p.get('candidate_name', 'Unknown')} ({p.get('candidate_persona', 'N/A')})")
        print(f"Rationale: {p['rationale']}\n")
        shown += 1
        if shown >= limit:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review gold labeled Resume-JD evaluation pairs")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to display")
    parser.add_argument("--grade", type=int, choices=[0, 1, 2], default=None, help="Filter by relevance grade")
    args = parser.parse_args()

    review_labels(limit=args.limit, filter_grade=args.grade)
