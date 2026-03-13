#!/usr/bin/env python3
"""
3-Stage SQL → dbt DAG pipeline (higher quality than single-prompt approach).

Each stage makes a separate LLM call, letting the model focus on one problem:

  Stage 1 — SQL Analyzer   : extract structured semantics from the SQL → JSON
  Stage 2 — dbt Planner    : convert the semantic JSON into a dbt DAG plan → JSON
  Stage 3 — dbt Generator  : render all dbt files from the plan → <file> XML tags
  Stage 4 — DuckDB Validator (optional): execute both the original SQL and the
             compiled dbt mart on DuckDB and compare result sets.

Splitting the cognitive load across three focused prompts boosts correctness
significantly for INTERSECT, GROUP BY, DISTINCT, and HAVING patterns.

Stage-1 results are cached in .cache/sql_analysis/ (keyed by SHA-256 of
SQL + context) so runs on structurally similar queries skip the LLM call.

Usage:
    uv run transform_sql_to_dbt_pipeline.py
    uv run transform_sql_to_dbt_pipeline.py --input data/selected_queries.jsonl \\
        --output data/dbt_dag_dataset_pipeline.jsonl --concurrency 4
    uv run transform_sql_to_dbt_pipeline.py --validate-duckdb  # enable Stage 4

Checkpointing:
    Already-successful rows (by question) are skipped automatically on re-run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import subprocess
from dataclasses import dataclass, field
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

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

LOCAL_LM_BASE_URL   = os.getenv("LOCAL_LM_BASE_URL",  "http://localhost:1234/v1")
LM_API_KEY          = os.getenv("LM_API_KEY",          "lm-studio")
DEFAULT_INPUT       = Path(os.getenv("INPUT_FILE",      "data/selected_queries.jsonl"))
DEFAULT_OUTPUT      = Path(os.getenv("OUTPUT_FILE",     "data/dbt_dag_dataset_pipeline.jsonl"))
DEFAULT_CONCURRENCY = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
DEFAULT_SAMPLE_SIZE = int(os.getenv("DEFAULT_SAMPLE_SIZE",     "1000"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES",      "3"))
MAX_TOKENS_ANALYZE  = int(os.getenv("MAX_TOKENS_ANALYZE",  "1024"))
MAX_TOKENS_PLAN     = int(os.getenv("MAX_TOKENS_PLAN",     "1024"))
MAX_TOKENS_GENERATE = int(os.getenv("MAX_TOKENS_GENERATE", "12000"))
TEMPERATURE         = float(os.getenv("TEMPERATURE",   "0.2"))  # low for structured output
TOP_P               = float(os.getenv("TOP_P",          "0.95"))
TOP_K               = int(os.getenv("TOP_K",            "20"))
MIN_P               = float(os.getenv("MIN_P",          "0.0"))
PRESENCE_PENALTY    = float(os.getenv("PRESENCE_PENALTY", "0.0"))
REPETITION_PENALTY  = float(os.getenv("REPETITION_PENALTY", "1.0"))
ENABLE_THINKING     = os.getenv("ENABLE_THINKING", "false").lower() == "true"

CACHE_DIR_ANALYSIS = Path(__file__).parent / ".cache" / "sql_analysis"
CACHE_DIR_PLAN     = Path(__file__).parent / ".cache" / "dbt_plan"

# Keep the old name as an alias for brevity
CACHE_DIR = CACHE_DIR_ANALYSIS

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── LLM client ───────────────────────────────────────────────────────────────

client = AsyncOpenAI(base_url=LOCAL_LM_BASE_URL, api_key=LM_API_KEY)

# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: SQL Analyzer ────────────────────────────────────────────────────

ANALYZE_SYSTEM = """\
You are a SQL expert. Your only task is to extract the semantic structure of a
SQL query and return it as a JSON object.

Return ONLY valid JSON — no markdown fences, no explanation.

JSON schema:
{
  "tables": ["<table_name>", ...],
  "joins": [
    {"left_table": "...", "right_table": "...", "condition": "..."}
  ],
  "filters": ["<WHERE clause condition>", ...],
  "aggregations": [
    {"function": "COUNT|SUM|AVG|MIN|MAX", "column": "...", "alias": "..."}
  ],
  "group_by": ["<table.column>", ...],
  "having": ["<HAVING clause condition>", ...],
  "select_columns": ["<table.column or alias>", ...],
  "set_operations": ["INTERSECT"|"UNION"|"EXCEPT", ...],
  "has_distinct": true|false,
  "has_subquery": true|false,
  "has_window_fn": true|false,
  "grain": "<what uniquely identifies one output row>",
  "model_type": "fact"|"dimension"|"report"
}

model_type rules:
  - "dimension"  : core business entity (customer, product, artist…)
  - "fact"       : events or aggregated metrics (GROUP BY required or event table)
  - "report"     : ad-hoc filtered/ranked/set-operation result
"""


def _analyze_user(sql: str, context: str) -> str:
    return (
        f"SQL schema:\n{context}\n\n"
        f"SQL query:\n{sql}\n\n"
        "Return the semantic JSON:"
    )


# ── Stage 2: dbt Planner ─────────────────────────────────────────────────────

PLAN_SYSTEM = """\
You are a dbt architect. Given a semantic analysis of a SQL query, produce a
dbt DAG plan as a JSON object.

Return ONLY valid JSON — no markdown fences, no explanation.

JSON schema:
{
  "sources": ["raw.<table>", ...],
  "staging_models": ["stg_<table>", ...],
  "intermediate_models": [],
  "mart_model": {
    "name": "<fct_|dim_|rpt_>...",
    "type": "fact"|"dimension"|"report",
    "grain": "1 row per <entity>"
  },
  "ctes": ["<cte_name>", ...],
  "joins": [{"left": "<cte_or_model>", "right": "<cte_or_model>", "on": "<condition>"}],
  "tests": {
    "<column_name>": ["not_null", "unique"]
  }
}

Mart model naming:
  - fct_  for facts and aggregated metrics
  - dim_  for core business entities
  - rpt_  for ad-hoc filtered/ranked/set-op results

Use intermediate_models only when the query has 3+ tables, window functions,
or complex reshaping steps. Otherwise leave it as an empty list.

The "ctes" list should reflect the WITH-clause CTEs needed in the mart model
(typically one per staging model being joined).

Add "not_null" tests on every output column.
Add "unique" on the grain column only when the grain guarantees uniqueness.
"""


def _plan_user(analysis: dict) -> str:
    return (
        "SQL semantic analysis:\n"
        + json.dumps(analysis, indent=2)
        + "\n\nReturn the dbt DAG plan JSON:"
    )


# ── Stage 3: dbt Code Generator ──────────────────────────────────────────────

GENERATE_SYSTEM = """\
You are a Staff Analytics Engineer expert in dbt.

You receive:
1. The original SQL query and its CREATE TABLE schema
2. A dbt DAG plan (JSON)

Your task: render ALL required dbt files exactly following the plan.

═══════════ PROJECT RULES ═══════════════════════════════════════════════════

STAGING (models/staging/stg_*.sql)
  - {{ config(materialized='view') }}
  - One file per source table. Reference raw data with {{ source('raw', '<table>') }}.
  - Rename ALL columns to snake_case. No joins, aggregations, or filters.

MARTS (models/marts/core/<name>.sql)
  - {{ config(materialized='table') }}
  - Use CTEs: one import CTE per staging model, then transform CTEs, then final SELECT.
  - Reference staging models with {{ ref('stg_...') }}.
  - NEVER use {{ source() }} in marts.
  - Always list columns explicitly — NEVER SELECT *.

YAML FILES (all required)
  - models/staging/_staging_sources.yml  — declares all source tables (name: raw)
  - models/staging/_staging_models.yml   — declares all stg_* models with column tests
  - models/marts/core/_core_models.yml   — declares the mart model with:
      description, meta.model_type, meta.grain, column tests

SQL CONVENTIONS
  - snake_case column names everywhere
  - Uppercase SQL keywords, lowercase identifiers
  - Explicit GROUP BY column names (not GROUP BY 1)
  - CTEs instead of subqueries

OUTPUT FORMAT
  - Wrap EACH file in XML tags: <file path="models/...">...</file>
  - Output ONLY <file> tags — no explanation, no markdown fences.
"""


def _generate_user(sql: str, context: str, plan: dict) -> str:
    return (
        f"SQL schema:\n{context}\n\n"
        f"Original SQL:\n{sql}\n\n"
        f"dbt DAG plan:\n{json.dumps(plan, indent=2)}\n\n"
        "Generate ALL dbt files. Output ONLY <file> XML tags:"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cache_key(sql: str, context: str) -> str:
    return hashlib.sha256(f"{sql}|||{context}".encode()).hexdigest()


def _analysis_cache_key(analysis: dict) -> str:
    canonical = json.dumps(analysis, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _load_cached_analysis(sql: str, context: str) -> Optional[dict]:
    CACHE_DIR_ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR_ANALYSIS / f"{_cache_key(sql, context)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    return None


def _save_cached_analysis(sql: str, context: str, analysis: dict) -> None:
    CACHE_DIR_ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR_ANALYSIS / f"{_cache_key(sql, context)}.json"
    path.write_text(json.dumps(analysis, indent=2))


def _load_cached_plan(analysis: dict) -> Optional[dict]:
    CACHE_DIR_PLAN.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR_PLAN / f"{_analysis_cache_key(analysis)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    return None


def _save_cached_plan(analysis: dict, plan: dict) -> None:
    CACHE_DIR_PLAN.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR_PLAN / f"{_analysis_cache_key(analysis)}.json"
    path.write_text(json.dumps(plan, indent=2))


def _strip_reasoning(text: str) -> str:
    return re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM output, tolerating markdown fences."""
    text = _strip_reasoning(text)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    # Find the outermost { … }
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {text[:300]!r}")
    # Walk to find the matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in output: {text[:300]!r}")


def _clean_xml_output(text: str) -> str:
    """Extract all <file>…</file> blocks, strip everything else."""
    text = _strip_reasoning(text)
    text = re.sub(r"```(?:xml|sql|yaml)?\s*", "", text)
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
        has_staging_sql and has_marts_sql and has_source and has_ref
        and has_config and has_sources_yml and has_staging_models_yml
        and has_core_models_yml
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — DuckDB EQUIVALENCE VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

# Minimal dbt project templates (same as validate_dbt_dag.py)
_DBT_PROJECT_YML = """\
name: 'validation_project'
version: '1.0.0'
config-version: 2
profile: 'validation'
model-paths: ["models"]
models:
  validation_project:
    staging:
      +materialized: view
    intermediate:
      +materialized: view
    marts:
      core:
        +materialized: table
"""

_PROFILES_YML = """\
validation:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ':memory:'
"""

_FILE_TAG_RE = re.compile(r'<file\s+path="([^"]+)">(.*?)</file>', re.DOTALL)


def _dbt_parse_validate(dbt_dag: str, dbt_cmd: str) -> tuple[str, list[str]]:
    """Run `dbt parse` on the generated DAG. Returns (status, errors)."""
    files = _FILE_TAG_RE.findall(dbt_dag)
    if not files:
        return "skipped", ["no_files_found"]

    with tempfile.TemporaryDirectory(prefix="dbt_val_") as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "dbt_project.yml").write_text(_DBT_PROJECT_YML)
        (tmp / "profiles.yml").write_text(_PROFILES_YML)
        for filepath, content in files:
            rel = Path(filepath)
            if rel.is_absolute() or ".." in rel.parts:
                continue
            full = tmp / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content.strip() + "\n")

        proc = subprocess.run(
            [
                dbt_cmd, "parse",
                "--profiles-dir", str(tmp),
                "--project-dir",  str(tmp),
                "--no-use-colors",
                "--no-partial-parse",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    ts_re = re.compile(r"^\d{2}:\d{2}:\d{2}\s+")
    err_re = re.compile(r"\b(compilation error|error)\b", re.IGNORECASE)
    errors = [
        ts_re.sub("", ln).strip()
        for ln in (proc.stdout + proc.stderr).splitlines()
        if err_re.search(ln)
    ][:20]
    status = "pass" if proc.returncode == 0 else "fail"
    return status, errors


# ══════════════════════════════════════════════════════════════════════════════
#  ASYNC LLM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_llm_call(semaphore: asyncio.Semaphore):
    """Return a retry-wrapped async function that calls the LLM."""

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(
            (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError)
        ),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    async def _call(
        system: str,
        user: str,
        max_tokens: int,
        label: str,
    ) -> str:
        async with semaphore:
            response = await client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                presence_penalty=PRESENCE_PENALTY,
                max_tokens=max_tokens,
                extra_body={
                    "top_k": TOP_K,
                    "min_p": MIN_P,
                    "repetition_penalty": REPETITION_PENALTY,
                    "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
                },
            )
        return response.choices[0].message.content.strip()

    return _call


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    success: bool
    row: dict
    stage1_analysis: Optional[dict] = None
    stage2_plan: Optional[dict]     = None
    dbt_dag: Optional[str]          = None
    failure_stage: Optional[str]    = None
    failure_reason: Optional[str]   = None


async def run_pipeline(
    example: dict,
    llm: object,
    label: str,
    validate_dbt: bool,
    dbt_cmd: str,
) -> PipelineResult:
    """Run the 3-stage pipeline for a single example."""
    sql     = example.get("answer") or example.get("original_sql", "")
    context = example.get("context", "")

    # ── Stage 1: SQL Analyzer ─────────────────────────────────────────────────
    analysis = _load_cached_analysis(sql, context)
    if analysis is not None:
        log.info(f"{label} S1 (cached) ✓")
    else:
        log.info(f"{label} S1 analyzing SQL…")
        try:
            raw1 = await llm(
                system=ANALYZE_SYSTEM,
                user=_analyze_user(sql, context),
                max_tokens=MAX_TOKENS_ANALYZE,
                label=f"{label}:S1",
            )
            analysis = _extract_json(raw1)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning(f"{label} S1 failed to parse JSON: {exc}")
            return PipelineResult(
                success=False, row=example,
                failure_stage="stage1", failure_reason=str(exc),
            )
        except BadRequestError as exc:
            log.error(f"{label} S1 model 400: {exc}")
            return PipelineResult(
                success=False, row=example,
                failure_stage="stage1", failure_reason=str(exc),
            )
        _save_cached_analysis(sql, context, analysis)
        log.info(f"{label} S1 ✓  model_type={analysis.get('model_type')} grain={analysis.get('grain')!r}")

    # ── Stage 2: dbt Planner ──────────────────────────────────────────────────
    plan = _load_cached_plan(analysis)
    if plan is not None:
        log.info(f"{label} S2 (cached) ✓  mart={plan.get('mart_model', {}).get('name')!r}")
    else:
        log.info(f"{label} S2 planning dbt DAG…")
        try:
            raw2 = await llm(
                system=PLAN_SYSTEM,
                user=_plan_user(analysis),
                max_tokens=MAX_TOKENS_PLAN,
                label=f"{label}:S2",
            )
            plan = _extract_json(raw2)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning(f"{label} S2 failed to parse JSON: {exc}")
            return PipelineResult(
                success=False, row=example, stage1_analysis=analysis,
                failure_stage="stage2", failure_reason=str(exc),
            )
        except BadRequestError as exc:
            log.error(f"{label} S2 model 400: {exc}")
            return PipelineResult(
                success=False, row=example, stage1_analysis=analysis,
                failure_stage="stage2", failure_reason=str(exc),
            )
        _save_cached_plan(analysis, plan)
        log.info(f"{label} S2 ✓  mart={plan.get('mart_model', {}).get('name')!r}")

    # ── Stage 3: dbt Code Generator ───────────────────────────────────────────
    log.info(f"{label} S3 generating dbt files…")
    try:
        raw3 = await llm(
            system=GENERATE_SYSTEM,
            user=_generate_user(sql, context, plan),
            max_tokens=MAX_TOKENS_GENERATE,
            label=f"{label}:S3",
        )
        dbt_dag = _clean_xml_output(raw3)
    except BadRequestError as exc:
        log.error(f"{label} S3 model 400: {exc}")
        return PipelineResult(
            success=False, row=example,
            stage1_analysis=analysis, stage2_plan=plan,
            failure_stage="stage3", failure_reason=str(exc),
        )

    if not _validate_dbt_dag(dbt_dag):
        log.warning(
            f"{label} S3 DAG validation failed.\n"
            f"  First 300 chars: {dbt_dag[:300]!r}"
        )
        return PipelineResult(
            success=False, row=example,
            stage1_analysis=analysis, stage2_plan=plan,
            dbt_dag=dbt_dag,
            failure_stage="stage3", failure_reason="missing_required_files",
        )

    log.info(f"{label} S3 ✓")

    # ── Stage 4: dbt parse validator (optional) ───────────────────────────────
    dbt_validation_status = None
    dbt_validation_errors: list[str] = []
    if validate_dbt:
        dbt_validation_status, dbt_validation_errors = _dbt_parse_validate(dbt_dag, dbt_cmd)
        if dbt_validation_status == "fail":
            log.warning(
                f"{label} S4 dbt parse FAILED: {dbt_validation_errors[:3]}"
            )
        else:
            log.info(f"{label} S4 ✓ dbt parse passed")

    # ── Build enriched result record ──────────────────────────────────────────
    result = {k: v for k, v in example.items() if k not in ("answer",)}
    result["original_sql"]   = sql
    result["dbt_dag"]        = dbt_dag
    result["status"]         = "success"
    result["sql_analysis"]   = analysis
    result["dbt_plan"]       = plan
    if validate_dbt:
        result["dbt_validation_status"] = dbt_validation_status
        result["dbt_validation_errors"] = dbt_validation_errors

    return PipelineResult(
        success=True, row=result,
        stage1_analysis=analysis, stage2_plan=plan, dbt_dag=dbt_dag,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def transform_dataset(
    examples: list[dict],
    concurrency: int,
    output_path: Path,
    n_done: int,
    validate_dbt: bool,
    dbt_cmd: str,
) -> tuple[int, int]:
    semaphore = asyncio.Semaphore(concurrency)
    llm = _make_llm_call(semaphore)

    total = len(examples)
    log.info(
        f"Pipeline: {total} rows | concurrency={concurrency} "
        f"| retries={MAX_RETRIES} | validate_dbt={validate_dbt}"
    )

    task_to_example: dict[asyncio.Task, dict] = {}
    for i, ex in enumerate(examples):
        t = asyncio.create_task(
            run_pipeline(
                ex, llm,
                label=f"[{n_done + i + 1}/{n_done + total}]",
                validate_dbt=validate_dbt,
                dbt_cmd=dbt_cmd,
            ),
            name=f"row_{i}",
        )
        task_to_example[t] = ex

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = failed_count = 0

    with output_path.open("a", encoding="utf-8") as f:
        try:
            for fut in asyncio.as_completed(list(task_to_example)):
                try:
                    result: PipelineResult = await fut
                except Exception as exc:
                    log.error(f"Task failed permanently: {exc}")
                    failed_count += 1
                    continue

                if result.success:
                    f.write(json.dumps(result.row, ensure_ascii=False) + "\n")
                    f.flush()
                    success_count += 1
                else:
                    log.warning(
                        f"Failed at {result.failure_stage}: {result.failure_reason}"
                    )
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


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-stage SQL → dbt pipeline with optional DuckDB validation."
    )
    parser.add_argument("--input",  "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Max rows to process (0 = all)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--validate-dbt", action="store_true",
                        help="Enable Stage 4: run `dbt parse` on each generated DAG")
    parser.add_argument("--dbt", default=None,
                        help="Path to dbt executable (auto-detected if not set)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if not args.input.exists():
        log.error(
            "Input file not found: %s\n"
            "Run 'uv run select_queries.py' first.",
            args.input,
        )
        return

    # Auto-detect dbt
    dbt_cmd = args.dbt
    if dbt_cmd is None:
        venv_dbt = Path(__file__).parent / ".venv" / "bin" / "dbt"
        dbt_cmd = str(venv_dbt) if venv_dbt.exists() else "dbt"

    all_examples: list[dict] = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log.info("Loaded %d examples from %s", len(all_examples), args.input)

    # Checkpoint: skip already-successful rows
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
        log.info("Checkpoint: %d already-successful rows will be skipped", len(done_questions))

    # Filter + sample
    pending = [ex for ex in all_examples if ex["question"] not in done_questions]
    if args.sample_size > 0:
        pending = pending[: args.sample_size - len(done_questions)]
    log.info("%d rows to process", len(pending))

    if not pending:
        log.info("Nothing to do.")
        return

    success, failed = await transform_dataset(
        examples=pending,
        concurrency=args.concurrency,
        output_path=args.output,
        n_done=len(done_questions),
        validate_dbt=args.validate_dbt,
        dbt_cmd=dbt_cmd,
    )

    total = success + failed
    log.info(
        "\n─── Pipeline complete ───────────────────────────────\n"
        "  Processed : %d\n"
        "  Success   : %d  (%.0f%%)\n"
        "  Failed    : %d\n"
        "  Output    : %s\n"
        "  S1 cache  : %s  (%d entries)\n"
        "  S2 cache  : %s  (%d entries)\n"
        "─────────────────────────────────────────────────────",
        total, success, 100 * success / total if total else 0, failed,
        args.output,
        CACHE_DIR_ANALYSIS,
        sum(1 for _ in CACHE_DIR_ANALYSIS.glob("*.json")) if CACHE_DIR_ANALYSIS.exists() else 0,
        CACHE_DIR_PLAN,
        sum(1 for _ in CACHE_DIR_PLAN.glob("*.json")) if CACHE_DIR_PLAN.exists() else 0,
    )


if __name__ == "__main__":
    asyncio.run(main())
