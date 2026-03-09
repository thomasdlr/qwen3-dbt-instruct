#!/usr/bin/env python3
"""
Test the dbt transformation on a small sample of selected queries.

Usage:
    uv run test_transform.py --complexity advanced --limit 5
    uv run test_transform.py --complexity complex --limit 10 --output test_results.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError
from transform_sql_to_dbt import (
    TRANSFORMATION_SYSTEM_PROMPT,
    TrainingExample,
    _clean_xml_output,
    _validate_dbt_dag,
)

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

LOCAL_LM_BASE_URL = os.getenv("LOCAL_LM_BASE_URL", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

client = AsyncOpenAI(
    base_url=LOCAL_LM_BASE_URL,
    api_key=LM_API_KEY,
)


def _build_user_message(original_sql: str, context: str, question: str) -> str:
    return (
        f"Business question: {question}\n\n"
        f"SQL schema:\n{context}\n\n"
        f"SQL query:\n{original_sql}\n\n"
        f"Decompose into a dbt DAG. Output ONLY <file> XML tags:"
    )


async def transform_query(question: str, context: str, sql: str) -> tuple[bool, str, str | None]:
    """
    Transform a single query. Returns (success, output_or_error, answer_if_valid).
    """
    try:
        response = await client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": TRANSFORMATION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(sql, context, question)},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        )
    except BadRequestError as e:
        return False, f"BadRequestError: {e}", None

    raw = response.choices[0].message.content.strip()
    dbt_dag = _clean_xml_output(raw)

    if not _validate_dbt_dag(dbt_dag):
        return False, "DAG validation failed", None

    return True, "✓ Valid", dbt_dag


async def main():
    parser = argparse.ArgumentParser(description="Test dbt transformation on selected queries")
    parser.add_argument(
        "--complexity",
        choices=["simple", "moderate", "complex", "advanced"],
        default="advanced",
        help="Filter by complexity level (default: advanced)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of queries to transform (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSONL file (default: print to stdout)",
    )
    args = parser.parse_args()

    # Load selected queries
    queries_file = Path("data/selected_queries.jsonl")
    if not queries_file.exists():
        log.error(f"{queries_file} not found")
        return

    log.info(f"Loading selected queries from {queries_file}")
    queries = [json.loads(l) for l in open(queries_file)]

    # Filter by complexity
    filtered = [q for q in queries if q["complexity"] == args.complexity]
    log.info(f"Found {len(filtered)} queries with complexity={args.complexity}")

    if not filtered:
        log.error(f"No queries found with complexity={args.complexity}")
        return

    # Take limit
    sample = filtered[: args.limit]
    log.info(f"Testing {len(sample)} queries")

    results: list[dict] = []
    successes = 0

    for idx, query in enumerate(sample, 1):
        label = f"[{idx}/{len(sample)}]"
        log.info(f"{label} {query['question'][:60]}...")

        success, msg, answer = await transform_query(
            query["question"],
            query["context"],
            query["answer"],
        )

        if success:
            log.info(f"{label} ✓ {msg}")
            successes += 1
            results.append({
                "question": query["question"],
                "context": query["context"],
                "original_sql": query["answer"],
                "complexity": query["complexity"],
                "complexity_score": query["complexity_score"],
                "features": query["features"],
                "table_count": query["table_count"],
                "dbt_dag": answer,
                "status": "success",
            })
        else:
            log.warning(f"{label} ✗ {msg}")
            results.append({
                "question": query["question"],
                "context": query["context"],
                "original_sql": query["answer"],
                "complexity": query["complexity"],
                "complexity_score": query["complexity_score"],
                "features": query["features"],
                "table_count": query["table_count"],
                "error": msg,
                "status": "failed",
            })

    log.info(f"\nResults: {successes}/{len(sample)} successful ({100*successes//len(sample)}%)")

    # Output
    if args.output:
        args.output.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
            encoding="utf-8",
        )
        log.info(f"Saved results → {args.output}")
    else:
        for r in results:
            if r["status"] == "success":
                print(f"\n{'='*70}")
                print(f"Q: {r['question']}")
                print(f"Complexity: {r['complexity']} (score={r['complexity_score']}, tables={r['table_count']})")
                print(f"Features: {', '.join(r['features'])}")
                print(f"\nGenerated dbt DAG:")
                print(f"{'-'*70}")
                print(r["dbt_dag"])


if __name__ == "__main__":
    asyncio.run(main())
