#!/usr/bin/env python3
"""
Transform selected queries into multi-file dbt DAGs using Claude Sonnet 4.6.

Reads from data/selected_queries.jsonl, calls Claude Sonnet 4.6 to generate
production-ready dbt DAGs, and writes results to data/dbt_dag_dataset_sonnet.jsonl.

Checkpointing: rows whose 'question' is already in the output with status='success'
are skipped automatically.
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CONCURRENCY = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
MAX_TOKENS = 4096

INPUT_FILE = Path("data/selected_queries.jsonl")
OUTPUT_FILE = Path("data/dbt_dag_dataset_sonnet.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
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

3. MARTS (models/marts/core/{prefix}_*.sql)
   - This is ALWAYS the final model answering the business question.
   - Reference {{ ref('stg_...') }} or {{ ref('int_...') }} — NEVER {{ source() }}.
   - Include {{ config(materialized='table') }}.
   - Use CTEs to organize logic: import CTE → transform CTE → final SELECT.

   PREFIX DECISION TREE — ask these questions in order:

   1. Does the model represent a core business entity?
      (customer, product, user, artist, driver, school…)
      → dim_   Entity table. May include attribute enrichment and even
               filtered sub-populations (dim_active_users is still a dim).
              Grain: one row per entity instance.

   2. Does the model represent events or individual measurable facts?
      (orders, clicks, payments, visits, race results…)
      → fct_   Row-level event table. Does NOT require GROUP BY — the grain
               is the event itself.
              Grain: one row per event/occurrence.

   3. Does the model produce derived metrics or aggregated measures?
      (revenue by customer, count of races per driver, avg score per team…)
      → fct_   Aggregated fact. Requires GROUP BY + aggregate functions.
              Grain: one row per entity/period being measured.

   4. None of the above — the query answers a specific ad-hoc question:
      filtered subsets, top-N, set operations (INTERSECT/EXCEPT), rankings.
      → rpt_   Report model. Not a reusable entity or metric table.
              Grain: one row per result of the specific question.

   QUICK EXAMPLES:
      dim_artist              — all artists with attributes
      dim_active_users        — filtered entity, still a dim
      fct_orders              — one row per order (no GROUP BY needed)
      fct_revenue_by_customer — aggregated metric (GROUP BY)
      rpt_top_chart_artists   — filtered/ranked answer to a question
      rpt_visitors_both_eras  — INTERSECT query, not a reusable entity

   MODEL TYPE — always set meta.model_type in the YAML:
      dim_  → meta: {model_type: dimension}
      fct_  → meta: {model_type: fact}
      rpt_  → meta: {model_type: report}

   GRAIN — derive it from the SQL structure, then set meta.grain:

      The grain = the set of columns that UNIQUELY IDENTIFIES each output row.
      Reason from the SQL, not from the business question wording:

      • GROUP BY col            → "1 row per {col}"
      • JOIN without DISTINCT   → "1 row per {left_pk}-{right_pk} pair"
        (e.g. artist JOIN volume → "1 row per artist-volume pair")
      • SELECT DISTINCT col     → "1 row per distinct {col}"
      • INTERSECT / EXCEPT      → "1 row per distinct {col} in result set"
      • ORDER BY … LIMIT 1      → "1 row (top result)"

      meta:
        model_type: fact
        grain: "1 row per customer"   ← always derived from the SQL above
      description: "Total revenue aggregated by customer name."

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
   - Declares ALL fct_*, dim_*, and rpt_* models with columns and tests.
   - Add not_null + unique tests on grain/primary-key columns.
   - Every model MUST have:
       description: plain English description of what the model contains.
       meta:
         model_type: dimension   # or: fact | report
         grain: "1 row per <entity>"

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
- fact SQL:         fct_{event_or_metric_name}.sql      ← events OR aggregated metrics
- dimension SQL:    dim_{entity_name}.sql               ← core business entity
- report SQL:       rpt_{question_name}.sql             ← filtered/ranked/set ops
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
    meta:
      model_type: fact
      grain: "1 row per customer"
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
- Missing meta.model_type in the _core_models.yml (must be: dimension, fact, or report).
- Missing meta.grain in the _core_models.yml (e.g. "1 row per customer").
- Embedding grain in the description instead of in meta.grain.
- Using dim_ only for unfiltered entities — filtered entity subsets are still dim_.
- Using fct_ only when there is a GROUP BY — event tables without aggregation are also fct_.
- Using dim_ for ad-hoc filtered query results — use rpt_ instead.
- Missing YAML files — they are REQUIRED in every output.
- Combining sources and models in one YAML file.
"""


def _build_user_message(sql: str, context: str, question: str) -> str:
    return (
        f"Business question: {question}\n\n"
        f"SQL schema:\n{context}\n\n"
        f"SQL query:\n{sql}\n\n"
        f"Decompose into a dbt DAG with ALL required SQL and YAML files. Output ONLY <file> XML tags:"
    )


def _clean_xml_output(text: str) -> str:
    text = re.sub(r"```(?:xml|sql)?\s*", "", text)
    text = re.sub(r"```", "", text)
    blocks = re.findall(r"(<file\s+path=\"[^\"]+\">.*?</file>)", text, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return text.strip()


def _validate_dbt_dag(output: str) -> bool:
    files = re.findall(r'<file\s+path="([^"]+)">(.*?)</file>', output, re.DOTALL)
    if not files:
        return False

    has_staging_sql = has_marts_sql = has_source = has_ref = has_config = False
    has_sources_yml = has_staging_models_yml = has_core_models_yml = False

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


async def _transform_one(
    client: anthropic.AsyncAnthropic,
    example: dict,
    semaphore: asyncio.Semaphore,
    label: str,
) -> dict | None:
    async with semaphore:
        log.info(f"{label} – transforming…")
        for attempt in range(3):
            try:
                msg = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": _build_user_message(
                                example["answer"], example["context"], example["question"]
                            ),
                        }
                    ],
                )
                break
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                if attempt == 2:
                    log.error(f"{label} – failed after 3 attempts: {e}")
                    return None
                wait = 4 * (2 ** attempt)
                log.warning(f"{label} – attempt {attempt+1} failed, retrying in {wait}s: {e}")
                await asyncio.sleep(wait)
            except anthropic.RateLimitError as e:
                wait = 30
                log.warning(f"{label} – rate limit, waiting {wait}s: {e}")
                await asyncio.sleep(wait)
                # retry same attempt
                attempt -= 1
            except Exception as e:
                log.error(f"{label} – unexpected error: {e}")
                return None

        raw = msg.content[0].text
        dbt_dag = _clean_xml_output(raw)

        if not _validate_dbt_dag(dbt_dag):
            log.warning(f"{label} – DAG validation failed. First 200 chars: {dbt_dag[:200]!r}")
            return None

        log.info(f"{label} ✓")
        result = {k: v for k, v in example.items() if k != "answer"}
        result["original_sql"] = example["answer"]
        result["dbt_dag"] = dbt_dag
        result["status"] = "success"
        return result


async def main() -> None:
    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    # Load input
    all_examples = [
        json.loads(line)
        for line in INPUT_FILE.read_text().splitlines()
        if line.strip()
    ]
    log.info("Loaded %d examples from %s", len(all_examples), INPUT_FILE)

    # Checkpoint: skip already-successful rows
    done_questions: set[str] = set()
    if OUTPUT_FILE.exists():
        for line in OUTPUT_FILE.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("status") == "success":
                    done_questions.add(row["question"])
            except json.JSONDecodeError:
                pass
        if done_questions:
            log.info("Checkpoint: %d rows already done — resuming…", len(done_questions))

    pending = [e for e in all_examples if e["question"] not in done_questions]
    if not pending:
        log.info("All rows already processed.")
        return

    log.info("Transforming %d rows (%d already done)", len(pending), len(done_questions))

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = len(pending)
    n_done = len(done_questions)

    task_map: dict[asyncio.Task, dict] = {}
    for i, ex in enumerate(pending):
        t = asyncio.create_task(
            _transform_one(client, ex, semaphore, f"[{n_done+i+1}/{n_done+total}]"),
            name=f"row_{i}",
        )
        task_map[t] = ex

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    success_count = failed_count = 0

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        try:
            for fut in asyncio.as_completed(list(task_map)):
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
                pct = 100 * success_count / done if done else 0
                log.info(
                    "Progress %d/%d | ✓ %d  ✗ %d  (%.0f%% success)",
                    n_done + done, n_done + total, success_count, failed_count, pct,
                )

        except (KeyboardInterrupt, asyncio.CancelledError):
            log.warning("Interrupted — cancelling pending tasks…")
            for t in task_map:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*task_map, return_exceptions=True)
            log.info(
                "Stopped after %d/%d rows. %d saved. Re-run to resume.",
                success_count + failed_count, total, success_count,
            )
            return

    log.info(
        "Done. %d/%d transformed successfully (%.0f%%).",
        success_count, success_count + failed_count,
        100 * success_count / (success_count + failed_count) if (success_count + failed_count) else 0,
    )


if __name__ == "__main__":
    asyncio.run(main())
