# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Fine-tuning dataset pipeline: raw SQL queries → multi-file dbt DAGs (SQL + YAML).
Source dataset: `b-mc2/sql-create-context` (HuggingFace). Target: a training dataset of ~1 000 examples where each input is a business question + SQL schema, and each output is a production-ready dbt DAG.

## Setup

```bash
uv sync                  # install all dependencies (Python 3.11+)
cp .env.example .env     # configure LOCAL_LM_BASE_URL and LM_API_KEY
```

All scripts use `uv run <script>.py`. A local LLM server (LM Studio) must be running at `LOCAL_LM_BASE_URL` before calling any transform step.

## Pipeline — run in order

```bash
# Step 1: Select + score queries from HuggingFace dataset
uv run select_queries.py --target 1000 --output data/selected_queries.jsonl --stats

# Step 2 (optional): Smoke-test the LLM on a few examples
uv run test_transform.py --complexity advanced --limit 5 --output data/results_smoke.jsonl

# Step 3a: Single-prompt transformer (simpler)
uv run transform_sql_to_dbt.py --sample-size 1000 --concurrency 4 --output data/dbt_dag_dataset.jsonl

# Step 3b: 3-stage pipeline (higher quality for complex SQL)
uv run transform_sql_to_dbt_pipeline.py --input data/selected_queries.jsonl --output data/dbt_dag_dataset_pipeline.jsonl --concurrency 4

# Step 4: Validate with dbt parse (no database required)
uv run validate_dbt_dag.py data/dbt_dag_dataset.jsonl --workers 8 --output data/validated_dataset.jsonl

# Step 5: Human review TUI
uv run review_dataset.py data/validated_dataset.jsonl
```

All transform scripts support **checkpointing**: rows with `status=success` already in the output file are automatically skipped. Interrupt with Ctrl+C and re-run the same command to resume.

## Architecture

### Scripts

| Script | Role |
|---|---|
| `select_queries.py` | Downloads HuggingFace dataset, scores complexity via 24 regex features (join, group_by, window functions, etc.), outputs balanced JSONL sample |
| `transform_sql_to_dbt.py` | Single-prompt LLM transformer (SQL → dbt DAG). Async with semaphore-based concurrency. Retries on network errors via `tenacity`. |
| `transform_sql_to_dbt_pipeline.py` | 3-stage pipeline: (1) SQL Analyzer → JSON semantics, (2) dbt Planner → DAG plan JSON, (3) dbt Generator → `<file>` XML tags. Stage 1 results cached in `.cache/sql_analysis/` by SHA-256. Optional Stage 4: DuckDB validation comparing original SQL output vs compiled dbt mart. |
| `validate_dbt_dag.py` | Scaffolds each DAG into a temp dbt project, runs `dbt parse` (no DB). Uses `ThreadPoolExecutor`. Exit code 1 if any row fails (CI-friendly). |
| `review_dataset.py` | Textual-based TUI. Annotations saved to `<file>.annotations.json`. |
| `test_transform.py` | Smoke test: runs the transformer on a small slice of selected queries. |

### Data flow

```
HuggingFace (b-mc2/sql-create-context)
  → select_queries.py
  → data/selected_queries.jsonl         (question, context, answer, complexity, features)
  → transform_sql_to_dbt*.py
  → data/dbt_dag_dataset.jsonl          (+ original_sql, dbt_dag XML, status)
  → validate_dbt_dag.py
  → data/validated_dataset.jsonl        (+ dbt_validation_status, dbt_validation_errors)
  → review_dataset.py
  → data/*.annotations.json
```

### dbt DAG output format

LLM outputs are XML-tagged multi-file bundles:

```
<file path="models/staging/stg_orders.sql">...</file>
<file path="models/staging/_staging_sources.yml">...</file>
<file path="models/staging/_staging_models.yml">...</file>
<file path="models/intermediate/int_*.sql">...</file>         ← optional
<file path="models/marts/core/fct_*.sql">...</file>           ← always fct_/dim_/rpt_
<file path="models/marts/core/_core_models.yml">...</file>
```

Validation (`_validate_dbt_dag`) requires: staging SQL, marts SQL, `source()` call, `ref()` call, `config()` block, plus all three YAML files.

## Configuration

All runtime parameters are read from `.env` (loaded via `python-dotenv`):

| Variable | Default | Description |
|---|---|---|
| `LOCAL_LM_BASE_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible endpoint |
| `LM_API_KEY` | `lm-studio` | API key (arbitrary for local) |
| `MAX_CONCURRENT_REQUESTS` | `4` | Semaphore size (2 for M1 Pro, 4–8 for M-Max/Ultra) |
| `MAX_TOKENS` | `12000` | Tokens per LLM response |
| `TEMPERATURE` | `0.3` | Lower = more deterministic dbt output |
| `ENABLE_THINKING` | `false` | Enable reasoning mode (set `true` for supported models) |

## Key implementation details

- All LLM calls use `AsyncOpenAI` with `base_url` pointed at LM Studio. The model name is always `"local-model"` (LM Studio ignores it).
- `tenacity` retries on `httpx.ReadTimeout`, `httpx.ConnectError`, `httpx.RemoteProtocolError` with exponential backoff.
- `_clean_xml_output()` strips reasoning tags (`<think>...</think>`) and markdown fences before extracting `<file>` blocks.
- The pipeline variant caches Stage 1 (SQL analysis) JSON results in `.cache/sql_analysis/` to avoid redundant LLM calls on reruns.
