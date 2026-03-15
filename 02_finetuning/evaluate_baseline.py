#!/usr/bin/env python3
"""
Evaluate a model's dbt DAG generation capability BEFORE fine-tuning.

Sends eval.jsonl prompts to a model running in LM Studio (or any OpenAI-compatible
endpoint) and measures how well it already knows dbt — so you can compare
the same metrics after fine-tuning.

Metrics:
  - dbt_parse_pass_rate  : % rows where the output passes `dbt parse`
  - has_staging          : has at least one stg_*.sql file
  - has_marts            : has at least one fct_/dim_/rpt_*.sql file
  - has_sources_yml      : includes _staging_sources.yml with source() calls
  - has_ref_calls        : intermediate/mart models use {{ ref() }}
  - correct_prefix       : mart model prefix matches expected (fct/dim/rpt)
  - avg_files_generated  : average number of <file> blocks in output

Usage:
    # Start LM Studio with your model, then:
    python evaluate_baseline.py --model "ministral-8b" --limit 50
    python evaluate_baseline.py --model "mistral-7b-instruct" --endpoint http://localhost:1234/v1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError:
    print("Missing dependency: uv sync")
    sys.exit(1)

load_dotenv(Path(__file__).parent / ".env")

_DEFAULT_ENDPOINT = os.getenv("LM_BASE_URL", "http://localhost:1234/v1")
_DEFAULT_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
_DEFAULT_MODEL = os.getenv("LM_MODEL", "local-model")
_DEFAULT_TEMP = float(os.getenv("TEMPERATURE", "0.1"))
_DEFAULT_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))


SYSTEM_PROMPT = """\
You are a Staff Analytics Engineer expert in dbt (data build tool).

Given a business question and the raw SQL schemas (CREATE TABLE statements), \
decompose the query into a production-ready multi-file dbt DAG following dbt \
Labs best practices.

Output ONLY the dbt files as XML-tagged blocks, nothing else:

<file path="models/staging/stg_<table>.sql">...</file>
<file path="models/staging/_staging_sources.yml">...</file>
<file path="models/staging/_staging_models.yml">...</file>
<file path="models/intermediate/int_<name>.sql">...</file>   ← only if needed
<file path="models/marts/core/<prefix>_<name>.sql">...</file>
<file path="models/marts/core/_core_models.yml">...</file>

Rules:
- Staging: {{ source('raw', 'Table') }}, materialized='view', snake_case columns
- Intermediate: only for joins on 3+ tables, window functions, or complex CTEs
- Marts prefix: fct_ (facts/metrics), dim_ (entities), rpt_ (ad-hoc reports)
- Marts: {{ ref('stg_...') }} or {{ ref('int_...') }}, materialized='table'
- Never use {{ source() }} in marts or intermediate models\
"""

# ── dbt validation (reused from 01_data_generation) ──────────────────────────

DBT_PROJECT_YML = """\
name: eval_project
version: '1.0.0'
config-version: 2
profile: duckdb_profile
model-paths: ["models"]
"""

PROFILES_YML = """\
duckdb_profile:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ':memory:'
"""

_FILE_TAG = re.compile(r'<file\s+path="([^"]+)">(.*?)</file>', re.DOTALL)


def _dbt_parse(dbt_dag: str, dbt_cmd: str) -> bool:
    files = _FILE_TAG.findall(dbt_dag)
    if not files:
        return False
    with tempfile.TemporaryDirectory(prefix="dbt_eval_") as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "dbt_project.yml").write_text(DBT_PROJECT_YML)
        (tmp / "profiles.yml").write_text(PROFILES_YML)
        for filepath, content in files:
            rel = Path(filepath)
            if rel.is_absolute() or ".." in rel.parts:
                continue
            full = tmp / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content.strip() + "\n")
        proc = subprocess.run(
            [
                dbt_cmd,
                "parse",
                "--profiles-dir",
                str(tmp),
                "--project-dir",
                str(tmp),
                "--no-use-colors",
                "--no-partial-parse",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    return proc.returncode == 0


# ── Metric helpers ────────────────────────────────────────────────────────────


def _count_files(dag: str) -> int:
    return len(_FILE_TAG.findall(dag))


def _has_staging(dag: str) -> bool:
    return bool(re.search(r'<file path="models/staging/stg_[^"]+\.sql">', dag))


def _has_marts(dag: str) -> bool:
    return bool(
        re.search(r'<file path="models/marts/core/(?:fct|dim|rpt)_[^"]+\.sql">', dag)
    )


def _has_sources_yml(dag: str) -> bool:
    return bool(re.search(r"source\s*\(", dag))


def _has_ref_calls(dag: str) -> bool:
    # ref() should appear in intermediate or mart sections (after staging)
    parts = _FILE_TAG.findall(dag)
    for path, content in parts:
        if "staging" not in path and re.search(r"\{\{\s*ref\s*\(", content):
            return True
    return False


def _correct_prefix(dag: str, expected_model_type: str) -> bool:
    """Check that the mart model uses the expected prefix (fct/dim/rpt)."""
    prefix_map = {"fct": "fct", "dim": "dim", "rpt": "rpt", "int_needed": "fct"}
    expected = prefix_map.get(expected_model_type, "")
    if not expected:
        return True  # can't check
    return bool(
        re.search(
            rf'<file path="models/marts/core/{expected}_[^"]+\.sql">',
            dag,
        )
    )


# ── Main evaluation loop ──────────────────────────────────────────────────────


def evaluate(args: argparse.Namespace) -> None:
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    # Auto-detect dbt
    venv_dbt = (
        Path(__file__).parent.parent / "01_data_generation" / ".venv" / "bin" / "dbt"
    )
    dbt_cmd = str(venv_dbt) if venv_dbt.exists() else "dbt"

    eval_rows = [json.loads(l) for l in open(args.input)]
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    print(f"Evaluating {len(eval_rows)} rows against model: {args.model}")
    print(f"Endpoint: {args.endpoint}\n")

    results = []
    for i, row in enumerate(eval_rows):
        msgs = row.get("messages") or [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Business question: {row['question']}\n\nSQL schemas:\n{row['context']}",
            },
        ]
        # Strip assistant turn if present (we're generating it)
        msgs = [m for m in msgs if m["role"] != "assistant"]

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=msgs,
                max_tokens=_DEFAULT_TOKENS,
                temperature=_DEFAULT_TEMP,
            )
            output = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i+1}] API error: {e}")
            output = ""

        # Metrics
        parsed = _dbt_parse(output, dbt_cmd) if output else False
        model_type = row.get("model_type", "")
        r = {
            "question": row.get("question", ""),
            "model_type": model_type,
            "dbt_parse_pass": parsed,
            "has_staging": _has_staging(output),
            "has_marts": _has_marts(output),
            "has_sources_yml": _has_sources_yml(output),
            "has_ref_calls": _has_ref_calls(output),
            "correct_prefix": _correct_prefix(output, model_type),
            "files_generated": _count_files(output),
            "output_len": len(output),
        }
        results.append(r)
        status = "PASS" if parsed else "fail"
        print(
            f"  [{i+1:3}/{len(eval_rows)}] {status}  files={r['files_generated']}  {row.get('question','')[:55]}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(results)
    print(f"\n{'='*60}")
    print(f"Model : {args.model}")
    print(f"Rows  : {n}")
    print(f"{'='*60}")
    metrics = [
        ("dbt_parse_pass_rate", mean(r["dbt_parse_pass"] for r in results)),
        ("has_staging_rate", mean(r["has_staging"] for r in results)),
        ("has_marts_rate", mean(r["has_marts"] for r in results)),
        ("has_sources_yml_rate", mean(r["has_sources_yml"] for r in results)),
        ("has_ref_calls_rate", mean(r["has_ref_calls"] for r in results)),
        ("correct_prefix_rate", mean(r["correct_prefix"] for r in results)),
        ("avg_files_generated", mean(r["files_generated"] for r in results)),
    ]
    for name, value in metrics:
        if name == "avg_files_generated":
            print(f"  {name:<28} {value:5.2f} files")
        else:
            bar = "█" * int(value * 20)
            print(f"  {name:<28} {value:5.1%}  {bar}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "n_rows": n,
                "metrics": dict(metrics),
                "rows": results,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline dbt DAG generation evaluation"
    )
    parser.add_argument(
        "--input",
        default="data/eval.jsonl",
        help="Eval JSONL file (default: data/eval.jsonl)",
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL, help="Model name as shown in LM Studio"
    )
    parser.add_argument(
        "--endpoint", default=_DEFAULT_ENDPOINT, help="OpenAI-compatible endpoint"
    )
    parser.add_argument("--api-key", default=_DEFAULT_API_KEY)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate first N rows (useful for quick tests)",
    )
    parser.add_argument(
        "--output",
        default="results/baseline.json",
        help="Where to save detailed results",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
