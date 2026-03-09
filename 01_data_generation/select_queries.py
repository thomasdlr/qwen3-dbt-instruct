#!/usr/bin/env python3
"""
Select and categorize queries from b-mc2/sql-create-context for dbt fine-tuning.

Analyzes SQL complexity using structural heuristics, assigns a complexity label,
and selects a balanced ~1000-row sample biased toward complex queries that
produce interesting dbt patterns.

Usage:
    uv run select_queries.py
    uv run select_queries.py --target 1000 --output data/selected_queries.jsonl
    uv run select_queries.py --target 1000 --output data/selected_queries.jsonl --stats
"""

import argparse
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Complexity scoring ──────────────────────────────────────────────────────

# Patterns and their point contributions to the complexity score.
# Each tuple: (compiled regex, points, feature name)
_FEATURES: list[tuple[re.Pattern, int, str]] = [
    (re.compile(r"\bJOIN\b", re.I), 2, "join"),
    (re.compile(r"\bLEFT\s+(?:OUTER\s+)?JOIN\b", re.I), 1, "left_join"),
    (re.compile(r"\bCROSS\s+JOIN\b", re.I), 2, "cross_join"),
    (re.compile(r"\bGROUP\s+BY\b", re.I), 2, "group_by"),
    (re.compile(r"\bHAVING\b", re.I), 2, "having"),
    (re.compile(r"\bORDER\s+BY\b", re.I), 1, "order_by"),
    (re.compile(r"\bUNION(?:\s+ALL)?\b", re.I), 3, "union"),
    (re.compile(r"\bEXCEPT\b", re.I), 3, "except"),
    (re.compile(r"\bINTERSECT\b", re.I), 3, "intersect"),
    (re.compile(r"\bCASE\b", re.I), 2, "case_when"),
    (re.compile(r"\bCOALESCE\b", re.I), 1, "coalesce"),
    (re.compile(r"\b(?:ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|NTILE)\s*\(", re.I), 3, "window_fn"),
    (re.compile(r"\bOVER\s*\(", re.I), 3, "over_clause"),
    (re.compile(r"\bWITH\b", re.I), 2, "cte"),
    (re.compile(r"\bEXISTS\s*\(", re.I), 2, "exists"),
    (re.compile(r"\bIN\s*\(\s*SELECT\b", re.I), 2, "subquery_in"),
    (re.compile(r"\bNOT\s+IN\s*\(\s*SELECT\b", re.I), 3, "not_in_subquery"),
    (re.compile(r"\(\s*SELECT\b", re.I), 2, "scalar_subquery"),
    (re.compile(r"\bDISTINCT\b", re.I), 1, "distinct"),
    (re.compile(r"\bLIMIT\b", re.I), 1, "limit"),
    (re.compile(r"\bBETWEEN\b", re.I), 1, "between"),
    (re.compile(r"\bLIKE\b", re.I), 1, "like"),
    (re.compile(r"\b(?:SUM|AVG|COUNT|MIN|MAX)\s*\(", re.I), 1, "aggregation"),
    (re.compile(r"\bCAST\s*\(", re.I), 1, "cast"),
    (re.compile(r"\b(?:DATE|DATETIME|TIMESTAMP|INTERVAL)\b", re.I), 1, "temporal"),
]

# Bonus: number of source tables
_CREATE_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+\w+", re.I)

# Complexity category thresholds
COMPLEXITY_LEVELS = {
    "simple": (0, 3),       # Basic SELECT, maybe one WHERE/ORDER
    "moderate": (4, 6),     # JOIN + GROUP BY or equivalent
    "complex": (7, 10),     # Multi-join, subqueries, window functions
    "advanced": (11, 999),  # Everything: CTEs, UNION, HAVING, window fns, etc.
}


@dataclass
class ScoredQuery:
    """A query with complexity metadata."""

    question: str
    context: str
    answer: str
    score: int
    complexity: str
    features: list[str]
    table_count: int

    def to_jsonl_record(self) -> dict:
        return {
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "complexity": self.complexity,
            "complexity_score": self.score,
            "features": self.features,
            "table_count": self.table_count,
        }


def score_query(question: str, context: str, sql: str) -> ScoredQuery:
    """Compute a complexity score and category for a SQL query."""
    full_text = f"{sql} {question}"
    score = 0
    features: list[str] = []

    for pattern, points, name in _FEATURES:
        matches = pattern.findall(full_text)
        if matches:
            # Count each occurrence but cap contribution per feature
            count = min(len(matches), 3)
            score += points * count
            features.append(f"{name}(x{count})" if count > 1 else name)

    # Bonus for number of source tables (more tables → more interesting DAG)
    table_count = len(_CREATE_TABLE_RE.findall(context))
    if table_count >= 3:
        score += (table_count - 2) * 3  # Strong boost for multi-table schemas
        features.append(f"tables({table_count})")
    elif table_count >= 2:
        score += 1
        features.append(f"tables({table_count})")

    # Determine complexity category
    complexity = "simple"
    for level, (lo, hi) in COMPLEXITY_LEVELS.items():
        if lo <= score <= hi:
            complexity = level
            break

    return ScoredQuery(
        question=question,
        context=context,
        answer=sql,
        score=score,
        complexity=complexity,
        features=features,
        table_count=table_count,
    )


# ─── Selection strategy ──────────────────────────────────────────────────────


def select_balanced_sample(
    scored: list[ScoredQuery],
    target: int,
    seed: int = 42,
) -> list[ScoredQuery]:
    """
    Select a balanced sample biased toward complex queries.

    Target distribution (approximate):
        simple:   10%
        moderate: 25%
        complex:  40%
        advanced: 25%
    """
    rng = random.Random(seed)

    by_level: dict[str, list[ScoredQuery]] = {k: [] for k in COMPLEXITY_LEVELS}
    for q in scored:
        by_level[q.complexity].append(q)

    # Shuffle each bucket
    for items in by_level.values():
        rng.shuffle(items)

    # Desired proportions — heavy on complex
    proportions = {
        "simple": 0.10,
        "moderate": 0.25,
        "complex": 0.40,
        "advanced": 0.25,
    }

    selected: list[ScoredQuery] = []
    remaining_budget = target

    # First pass: take up to the proportion from each bucket
    for level, prop in proportions.items():
        want = int(target * prop)
        available = by_level[level]
        take = min(want, len(available))
        selected.extend(available[:take])
        by_level[level] = available[take:]
        remaining_budget -= take

    # Second pass: fill remaining budget from any bucket with leftovers,
    # prioritizing complex > advanced > moderate > simple
    priority_order = ["complex", "advanced", "moderate", "simple"]
    for level in priority_order:
        if remaining_budget <= 0:
            break
        available = by_level[level]
        take = min(remaining_budget, len(available))
        selected.extend(available[:take])
        remaining_budget -= take

    # Final shuffle so the output isn't grouped by complexity
    rng.shuffle(selected)

    return selected


# ─── Deduplication ────────────────────────────────────────────────────────────


def _has_duplicate_tables(context: str) -> bool:
    """Check if context has duplicate CREATE TABLE statements."""
    tables = _CREATE_TABLE_RE.findall(context)
    unique_tables = set(t.lower() for t in tables)
    return len(tables) != len(unique_tables)


def _has_duplicate_columns(context: str) -> bool:
    """Check if any table has duplicate column names (heuristic)."""
    # Look for table definitions with duplicate identifiers
    table_pattern = r"CREATE\s+TABLE\s+\w+\s*\(([^)]+)\)"
    for match in re.finditer(table_pattern, context, re.I):
        cols = match.group(1)
        # Extract column names (simple heuristic: word after some identifier)
        col_names = re.findall(r"\b(\w+)\s+(?:VARCHAR|INT|DECIMAL|TIMESTAMP|DATE)", cols, re.I)
        if len(col_names) != len(set(c.lower() for c in col_names)):
            return True
    return False


def deduplicate(scored: list[ScoredQuery]) -> list[ScoredQuery]:
    """Remove near-duplicate queries based on normalized SQL, and filter malformed contexts."""
    seen: set[str] = set()
    unique: list[ScoredQuery] = []
    for q in scored:
        # Skip malformed contexts
        if _has_duplicate_tables(q.context) or _has_duplicate_columns(q.context):
            continue
        
        # Normalize: lowercase, collapse whitespace, strip quotes
        key = re.sub(r"\s+", " ", q.answer.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and categorize queries from b-mc2/sql-create-context."
    )
    parser.add_argument(
        "--target",
        "-n",
        type=int,
        default=1000,
        help="Target number of queries to select (default: 1000)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/selected_queries.jsonl"),
        help="Output JSONL path (default: data/selected_queries.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics after selection",
    )
    parser.add_argument(
        "--min-tables",
        type=int,
        default=2,
        help="Minimum number of tables in schema (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading b-mc2/sql-create-context dataset from Hugging Face...")
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    log.info(f"Full dataset: {len(dataset)} examples")

    # Score every query
    log.info("Scoring query complexity...")
    scored: list[ScoredQuery] = []
    for row in dataset:
        q = score_query(row["question"], row["context"], row["answer"])
        if q.table_count >= args.min_tables:
            scored.append(q)
    log.info(f"Scored {len(scored)} queries (min {args.min_tables} table(s))")

    # Deduplicate
    before = len(scored)
    scored = deduplicate(scored)
    log.info(f"After dedup & quality filtering: {len(scored)} (removed {before - len(scored)} duplicates/malformed)")

    # Distribution before selection
    dist = Counter(q.complexity for q in scored)
    log.info("Full dataset distribution:")
    for level in COMPLEXITY_LEVELS:
        log.info(f"  {level:>10s}: {dist.get(level, 0):>6d}")

    # Select balanced sample
    selected = select_balanced_sample(scored, target=args.target, seed=args.seed)
    log.info(f"Selected {len(selected)} queries")

    # Distribution after selection
    sel_dist = Counter(q.complexity for q in selected)
    log.info("Selected distribution:")
    for level in COMPLEXITY_LEVELS:
        count = sel_dist.get(level, 0)
        pct = 100 * count / len(selected) if selected else 0
        log.info(f"  {level:>10s}: {count:>4d} ({pct:.0f}%)")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q in selected:
            f.write(json.dumps(q.to_jsonl_record(), ensure_ascii=False) + "\n")
    log.info(f"Saved → {args.output}")

    # Detailed stats
    if args.stats:
        feature_counts = Counter()
        for q in selected:
            for feat in q.features:
                # Strip count suffix for aggregation
                name = feat.split("(")[0]
                feature_counts[name] += 1
        log.info("Feature frequency in selection:")
        for feat, cnt in feature_counts.most_common():
            log.info(f"  {feat:>20s}: {cnt:>4d}")

        scores = [q.score for q in selected]
        log.info(f"Score range: {min(scores)} – {max(scores)}, "
                 f"median: {sorted(scores)[len(scores)//2]}, "
                 f"mean: {sum(scores)/len(scores):.1f}")

        table_counts = Counter(q.table_count for q in selected)
        log.info("Table count distribution:")
        for n in sorted(table_counts):
            log.info(f"  {n} table(s): {table_counts[n]}")


if __name__ == "__main__":
    main()
