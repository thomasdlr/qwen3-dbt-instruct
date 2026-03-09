#!/usr/bin/env python3
"""
Transform selected queries into multi-file dbt DAGs.

Reads from data/selected_queries.jsonl (produced by select_queries.py),
calls the local LLM to decompose each SQL query into staging/intermediate/marts
dbt models, and writes results incrementally so the run can be interrupted and
resumed at any time.

Usage:
    uv run transform_sql_to_dbt.py
    uv run transform_sql_to_dbt.py --sample-size 1000 --concurrency 4 --output data/dbt_dag_dataset.jsonl
    uv run transform_sql_to_dbt.py --input data/selected_queries.jsonl --output data/dbt_dag_dataset.jsonl

Checkpointing:
    If the output file already exists, rows whose 'question' field is already
    present with status='success' are skipped automatically.  Just re-run the
    same command to resume after a stop.
"""

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ─── Configuration (loaded from .env) ────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

LOCAL_LM_BASE_URL   = os.getenv("LOCAL_LM_BASE_URL", "http://localhost:1234/v1")
LM_API_KEY          = os.getenv("LM_API_KEY", "lm-studio")
DEFAULT_INPUT       = Path(os.getenv("INPUT_FILE",  "data/selected_queries.jsonl"))
DEFAULT_OUTPUT      = Path(os.getenv("OUTPUT_FILE", "data/dbt_dag_dataset.jsonl"))

DEFAULT_CONCURRENCY = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
DEFAULT_SAMPLE_SIZE = int(os.getenv("DEFAULT_SAMPLE_SIZE", "1000"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "3"))
MAX_TOKENS          = int(os.getenv("MAX_TOKENS", "12000"))  # YAML + multi-file output needs more tokens

TEMPERATURE         = float(os.getenv("TEMPERATURE", "0.3"))  # Lower temp for more deterministic transforms
TOP_P               = float(os.getenv("TOP_P", "0.95"))
TOP_K               = int(os.getenv("TOP_K", "20"))
MIN_P               = float(os.getenv("MIN_P", "0.0"))
PRESENCE_PENALTY    = float(os.getenv("PRESENCE_PENALTY", "0.0"))
REPETITION_PENALTY  = float(os.getenv("REPETITION_PENALTY", "1.0"))
ENABLE_THINKING     = os.getenv("ENABLE_THINKING", "false").lower() == "true"  # Disable by default for transformations

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── LM Studio client ─────────────────────────────────────────────────────────

client = AsyncOpenAI(
    base_url=LOCAL_LM_BASE_URL,
    api_key=LM_API_KEY,
)

# ─── Prompts ──────────────────────────────────────────────────────────────────

TRANSFORMATION_SYSTEM_PROMPT = """\
You are a Staff Analytics Engineer expert in dbt (data build tool).

I will give you a business question and the raw SQL query that answers it.
Your mission: decompose this query into a production-ready dbt DAG (directed
acyclic graph) following dbt Labs best practices EXACTLY.

═══════════════════════════════════════════════════════════════════
 PROJECT STRUCTURE
═══════════════════════════════════════════════════════════════════

models/
  staging/         ← clean raw data (1:1 with sources)
  intermediate/    ← business transformations (joins, reshaping)
  marts/
    core/          ← final analytics tables (fct_ and dim_)

═══════════════════════════════════════════════════════════════════
 LAYER RULES
═══════════════════════════════════════════════════════════════════

1. STAGING (models/staging/stg_*.sql)
   - One model per source table (1:1 mapping).
   - Reference raw data with {{ source('raw', 'Table_Name') }}.
   - Include {{ config(materialized='view') }} at the top.
   - Rename ALL columns to snake_case with meaningful aliases.
   - Minimal logic: column selection, renaming, casting only.
   - NEVER join, aggregate, or filter in staging.

2. INTERMEDIATE (models/intermediate/int_*.sql) — ONLY IF NEEDED
   - Create ONLY when the query involves: subqueries/CTEs, window
     functions, HAVING clauses, or joins on 3+ tables.
   - Reference staging models with {{ ref('stg_...') }}.
   - Include {{ config(materialized='view') }}.
   - Use CTEs with descriptive names (e.g., active_users, monthly_totals).
   - Prepare joins, reshape data, or change the grain.
   - No heavy aggregations (save those for marts).

3. MARTS (models/marts/core/fct_*.sql or models/marts/core/dim_*.sql)
   - This is ALWAYS the final model answering the business question.
   - Reference {{ ref('stg_...') }} or {{ ref('int_...') }} — NEVER {{ source() }}.
   - Include {{ config(materialized='table') }}.
   - fct_ for fact tables: events, transactions, metrics, aggregated data.
     Must define a clear grain (e.g., "1 row per participant").
   - dim_ for dimension tables: descriptive attributes, entity lookups.
   - Use CTEs to organize logic: import CTE → transform CTE → final SELECT.

═══════════════════════════════════════════════════════════════════
 SQL CONVENTIONS (MANDATORY)
═══════════════════════════════════════════════════════════════════

- snake_case for ALL column names (participant_id, NOT ParticipantID).
- NEVER use SELECT * — always list columns explicitly.
- Explicit GROUP BY with column names (NOT GROUP BY 1, 2).
- Use CTEs (WITH ... AS) for logical steps, not subqueries.
- Add SQL comments for non-obvious business logic.
- Consistent formatting: uppercase SQL keywords, lowercase identifiers.

═══════════════════════════════════════════════════════════════════
 YAML FILES (REQUIRED — one file per directory)
═══════════════════════════════════════════════════════════════════

Always generate these YAML files alongside the SQL files:

4. models/staging/_staging_sources.yml
   - Declares ALL source tables used by stg_* models.
   - One sources: block with name: raw.
   - Add not_null + unique tests on primary-key columns.

5. models/staging/_staging_models.yml
   - Declares ALL stg_* models with their columns and tests.
   - Add not_null tests on primary-key columns.

6. models/intermediate/_intermediate_models.yml  — ONLY if intermediate exists
   - Declares ALL int_* models.

7. models/marts/core/_core_models.yml
   - Declares ALL fct_* and dim_* models with columns and tests.
   - Add not_null + unique tests on grain/primary-key columns.

═══════════════════════════════════════════════════════════════════
 OUTPUT FORMAT — CRITICAL
═══════════════════════════════════════════════════════════════════

- Wrap EACH file (SQL and YAML) in XML tags:
    <file path="models/layer/name.sql">...</file>
    <file path="models/staging/_staging_sources.yml">...</file>
- Marts SQL goes in models/marts/core/ (not models/marts/).
- Provide NO explanation, NO markdown fences, NO commentary.
- ONLY output the <file> XML tags.

═══════════════════════════════════════════════════════════════════
 NAMING CONVENTIONS
═══════════════════════════════════════════════════════════════════

- staging SQL:      stg_{source_table_snake_case}.sql
- intermediate SQL: int_{descriptive_name}.sql
- fact SQL:         fct_{metric_name}.sql
- dimension SQL:    dim_{entity_name}.sql
- staging YAML:     models/staging/_staging_sources.yml
                    models/staging/_staging_models.yml
- intermediate YAML: models/intermediate/_intermediate_models.yml
- marts YAML:       models/marts/core/_core_models.yml

═══════════════════════════════════════════════════════════════════
 EXAMPLE
═══════════════════════════════════════════════════════════════════

Input:
SQL schema: CREATE TABLE customers (id INT, name VARCHAR); CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL)
SQL query: SELECT c.name, SUM(o.amount) AS total_revenue FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name

Output:
<file path="models/staging/stg_customers.sql">
{{ config(materialized='view') }}

SELECT
    id AS customer_id,
    name
FROM {{ source('raw', 'customers') }}
</file>

<file path="models/staging/stg_orders.sql">
{{ config(materialized='view') }}

SELECT
    id AS order_id,
    customer_id,
    amount
FROM {{ source('raw', 'orders') }}
</file>

<file path="models/staging/_staging_sources.yml">
version: 2

sources:
  - name: raw
    tables:
      - name: customers
      - name: orders
</file>

<file path="models/staging/_staging_models.yml">
version: 2

models:
  - name: stg_customers
    columns:
      - name: customer_id
        tests:
          - not_null
          - unique
      - name: name
        tests:
          - not_null

  - name: stg_orders
    columns:
      - name: order_id
        tests:
          - not_null
          - unique
      - name: customer_id
        tests:
          - not_null
</file>

<file path="models/marts/core/fct_revenue_by_customer.sql">
{{ config(materialized='table') }}

WITH customers AS (

    SELECT *
    FROM {{ ref('stg_customers') }}

),

orders AS (

    SELECT *
    FROM {{ ref('stg_orders') }}

)

SELECT
    customers.name,
    SUM(orders.amount) AS total_revenue
FROM customers
INNER JOIN orders
    ON customers.customer_id = orders.customer_id
GROUP BY
    customers.name
</file>

<file path="models/marts/core/_core_models.yml">
version: 2

models:
  - name: fct_revenue_by_customer
    description: "Total revenue aggregated by customer name."
    columns:
      - name: name
        tests:
          - not_null
          - unique
      - name: total_revenue
        tests:
          - not_null
</file>

═══════════════════════════════════════════════════════════════════
 COMMON MISTAKES TO AVOID
═══════════════════════════════════════════════════════════════════

- Referencing raw tables without source() in staging.
- Missing {{ config(materialized=...) }} block.
- Using SELECT * anywhere.
- Mixing staging logic (source) with business logic (joins/aggs).
- Using GROUP BY 1, 2 instead of explicit column names.
- Putting marts directly in models/marts/ instead of models/marts/core/.
- Not defining a clear grain for fact tables.
- Missing YAML files — they are REQUIRED in every output.
- Combining sources and models in one YAML file.
"""


def _build_user_message(original_sql: str, context: str, question: str) -> str:
    return (
        f"Business question: {question}\n\n"
        f"SQL schema:\n{context}\n\n"
        f"SQL query:\n{original_sql}\n\n"
        f"Decompose into a dbt DAG with ALL required SQL and YAML files. Output ONLY <file> XML tags:"
    )


# ─── Data model ───────────────────────────────────────────────────────────────


@dataclass
class TrainingExample:
    question: str
    context: str
    answer: str  # dbt-transformed SQL

    def to_jsonl_record(self) -> dict:
        """SQL-create-context format: question, context, answer."""
        return {
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
        }


# ─── Dataset filtering ────────────────────────────────────────────────────────

_JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)


def _is_complex_query(sql: str) -> bool:
    """Return True if the SQL contains at least one JOIN and a GROUP BY."""
    return bool(_JOIN_RE.search(sql)) and bool(_GROUP_BY_RE.search(sql))


def _extract_table_names(context: str) -> list[str]:
    """Extract table names from CREATE TABLE statements in context."""
    return re.findall(r"CREATE\s+TABLE\s+(\w+)", context, re.IGNORECASE)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _strip_reasoning(text: str) -> str:
    """Remove reasoning/thinking preambles emitted by reasoning models."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _clean_xml_output(text: str) -> str:
    """Extract the <file> XML blocks from model output, stripping noise."""
    text = _strip_reasoning(text)

    # Remove markdown fences if the model wrapped them
    text = re.sub(r"```(?:xml|sql)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # Extract all <file ...>...</file> blocks and rejoin
    blocks = re.findall(
        r"(<file\s+path=\"[^\"]+\">.*?</file>)",
        text,
        re.DOTALL,
    )
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)

    # Fallback: return cleaned text as-is (will likely fail validation)
    return text.strip()


def _validate_dbt_dag(output: str) -> bool:
    """Validate the multi-file dbt DAG output."""
    files = re.findall(r'<file\s+path="([^"]+)">(.*?)</file>', output, re.DOTALL)
    if not files:
        return False

    has_staging_sql = False
    has_marts_sql = False
    has_source = False
    has_ref = False
    has_config = False
    has_sources_yml = False
    has_staging_models_yml = False
    has_core_models_yml = False

    for path, content in files:
        if path.endswith(".sql"):
            if "/staging/" in path:
                has_staging_sql = True
            if "/marts/" in path:
                has_marts_sql = True
            if "source(" in content:
                has_source = True
            if "ref(" in content:
                has_ref = True
            if "config(" in content:
                has_config = True
            if "SELECT" not in content.upper():
                return False
        elif path.endswith((".yml", ".yaml")):
            if "_staging_sources" in path:
                has_sources_yml = True
            if "_staging_models" in path:
                has_staging_models_yml = True
            if "_core_models" in path:
                has_core_models_yml = True

    return (
        has_staging_sql
        and has_marts_sql
        and has_source
        and has_ref
        and has_config
        and has_sources_yml
        and has_staging_models_yml
        and has_core_models_yml
    )


# ─── Async transformation ─────────────────────────────────────────────────────


def _make_retry_decorator(semaphore_ref: asyncio.Semaphore):
    """Build the @retry decorator so it can reference the runtime semaphore."""

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError)),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    async def _transform(example: dict, label: str) -> Optional[TrainingExample]:
        async with semaphore_ref:
            log.info(f"{label} – transforming…")
            try:
                response = await client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": TRANSFORMATION_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_message(
                            example["answer"], example["context"], example["question"]
                        )},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    presence_penalty=PRESENCE_PENALTY,
                    max_tokens=MAX_TOKENS,
                    extra_body={
                        "top_k": TOP_K,
                        "min_p": MIN_P,
                        "repetition_penalty": REPETITION_PENALTY,
                        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
                    },
                )
            except BadRequestError as e:
                log.error(f"{label} – model returned 400 (crash?), skipping: {e}")
                return None
            
            raw = response.choices[0].message.content.strip()
            dbt_dag = _clean_xml_output(raw)

        if not _validate_dbt_dag(dbt_dag):
            log.warning(
                f"{label} – DAG validation failed (need staging+marts with source+ref).\n"
                f"First 300 chars: {dbt_dag[:300]!r}"
            )
            return None

        log.info(f"{label} ✓")
        # Carry over all input fields (question, context, complexity, features, etc.)
        # rename 'answer' → 'original_sql', add 'dbt_dag' and 'status'
        result = {k: v for k, v in example.items() if k != "answer"}
        result["original_sql"] = example["answer"]
        result["dbt_dag"] = dbt_dag
        result["status"] = "success"
        return result

    return _transform


async def transform_dataset(
    examples: list[dict],
    concurrency: int,
    output_path: Path,
    n_done: int,
) -> tuple[int, int]:
    """Transform examples, writing each result immediately. Returns (success, failed)."""
    semaphore = asyncio.Semaphore(concurrency)
    _transform = _make_retry_decorator(semaphore)

    total = len(examples)
    log.info(
        f"Transforming {total} rows "
        f"({concurrency} concurrent, {MAX_RETRIES} retries each)…"
    )

    # Create all tasks upfront, keeping a reverse map task→example for error rows
    task_to_example: dict[asyncio.Task, dict] = {}
    for i, ex in enumerate(examples):
        t = asyncio.create_task(
            _transform(ex, label=f"[{n_done + i + 1}/{n_done + total}]"),
            name=f"row_{i}",
        )
        task_to_example[t] = ex

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    failed_count  = 0

    with output_path.open("a", encoding="utf-8") as f:
        try:
            for fut in asyncio.as_completed(list(task_to_example)):
                try:
                    result = await fut
                except Exception as exc:
                    log.error(f"Task failed permanently: {exc}")
                    failed_count += 1
                    continue

                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    success_count += 1
                else:
                    failed_count += 1

                done = success_count + failed_count
                pct  = 100 * success_count / done if done else 0
                log.info(
                    f"Progress {n_done + done}/{n_done + total} "
                    f"| ✓ {success_count}  ✗ {failed_count}  ({pct:.0f}% success)"
                )

        except (KeyboardInterrupt, asyncio.CancelledError):
            log.warning("\nInterrupted — cancelling pending tasks…")
            for t in task_to_example:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*task_to_example, return_exceptions=True)
            log.info(
                f"Stopped after {success_count + failed_count}/{total} rows. "
                f"{success_count} saved to {output_path}. "
                "Re-run the same command to resume."
            )
            return success_count, failed_count

    return success_count, failed_count


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform selected queries into dbt DAGs with checkpoint support."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL of scored queries (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Max rows to process (default: {DEFAULT_SAMPLE_SIZE}; use 0 for all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM requests (default: {DEFAULT_CONCURRENCY})",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # ── Load input ───────────────────────────────────────────────────────────
    if not args.input.exists():
        log.error(
            "Input file not found: %s\n"
            "Run 'uv run select_queries.py' first to generate it.",
            args.input,
        )
        return

    all_examples: list[dict] = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log.info("Loaded %d examples from %s", len(all_examples), args.input)

    # ── Checkpoint: skip already-successful rows ──────────────────────────────
    done_questions: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("status") == "success":
                    done_questions.add(row["question"])
            except json.JSONDecodeError:
                pass
        if done_questions:
            log.info(
                "Checkpoint: %d rows already done — resuming…", len(done_questions)
            )

    # ── Select pending rows ───────────────────────────────────────────────────
    pending = [e for e in all_examples if e["question"] not in done_questions]
    if args.sample_size > 0:
        pending = pending[: args.sample_size]

    if not pending:
        log.info("All rows already processed. Nothing to do.")
        return

    log.info(
        "Transforming %d rows (%d already done in checkpoint)",
        len(pending),
        len(done_questions),
    )

    # ── Transform SQL → dbt DAG ───────────────────────────────────────────────
    try:
        success, failed = await transform_dataset(
            pending,
            concurrency=args.concurrency,
            output_path=args.output,
            n_done=len(done_questions),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        return

    total_attempted = success + failed
    log.info(
        "Done. %d/%d transformed successfully (%.0f%%).",
        success,
        total_attempted,
        100 * success / total_attempted if total_attempted else 0,
    )


if __name__ == "__main__":
    asyncio.run(main())
