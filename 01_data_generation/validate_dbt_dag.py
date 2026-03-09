#!/usr/bin/env python3
"""
Validate generated dbt DAGs using `dbt parse`.

For each row in a JSONL file that has a `dbt_dag` field, scaffolds a minimal
dbt project in a temp directory, runs `dbt parse` to catch compilation errors,
ref/source resolution failures, YAML syntax issues, and circular dependencies —
all without needing a running database.

Usage:
    uv run validate_dbt_dag.py data/results_codestral.jsonl
    uv run validate_dbt_dag.py data/dbt_dag_dataset.jsonl --output data/validated.jsonl
    uv run validate_dbt_dag.py data/dbt_dag_dataset.jsonl --workers 8 --output data/validated.jsonl
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─── Minimal dbt project templates ───────────────────────────────────────────

DBT_PROJECT_YML = """\
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

PROFILES_YML = """\
validation:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ':memory:'
"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

_FILE_TAG = re.compile(r'<file\s+path="([^"]+)">(.*?)</file>', re.DOTALL)

# dbt error lines look like: "HH:MM:SS  Compilation Error in model ..."
# or "HH:MM:SS  Error"
_ERROR_RE = re.compile(r"\b(compilation error|error)\b", re.IGNORECASE)
_WARN_RE  = re.compile(r"\bwarning\b", re.IGNORECASE)
# Strip leading timestamp like "12:34:56  "
_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2}\s+")


def _strip_ts(line: str) -> str:
    return _TIMESTAMP.sub("", line).strip()


def parse_file_tags(text: str) -> list[tuple[str, str]]:
    return _FILE_TAG.findall(text)


def _extract_errors_warnings(output: str) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for raw in output.splitlines():
        line = _strip_ts(raw)
        if not line:
            continue
        if _ERROR_RE.search(line):
            errors.append(line)
        elif _WARN_RE.search(line):
            warnings.append(line)
    return errors[:20], warnings[:10]


# ─── Per-row validation ───────────────────────────────────────────────────────

def validate_row(row: dict, dbt_cmd: str, idx: int) -> dict:
    dbt_dag = row.get("dbt_dag", "")

    # Skip rows that never successfully generated a DAG
    if not dbt_dag or row.get("status") != "success":
        return {
            **row,
            "dbt_validation_status": "skipped",
            "dbt_validation_errors": [],
            "dbt_validation_warnings": [],
        }

    files = parse_file_tags(dbt_dag)
    if not files:
        return {
            **row,
            "dbt_validation_status": "skipped",
            "dbt_validation_errors": ["no_files_found_in_dbt_dag"],
            "dbt_validation_warnings": [],
        }

    with tempfile.TemporaryDirectory(prefix=f"dbt_val_{idx}_") as tmpdir:
        tmp = Path(tmpdir)

        # Scaffold minimal dbt project
        (tmp / "dbt_project.yml").write_text(DBT_PROJECT_YML)
        (tmp / "profiles.yml").write_text(PROFILES_YML)

        # Write every model file extracted from the dbt_dag XML tags
        for filepath, content in files:
            # Prevent directory traversal attacks from generated content
            rel = Path(filepath)
            if rel.is_absolute() or ".." in rel.parts:
                continue
            full_path = tmp / rel
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content.strip() + "\n")

        # Run dbt parse — validates refs, sources, YAML, circular deps.
        # Does NOT connect to the database.
        proc = subprocess.run(
            [
                dbt_cmd, "parse",
                "--profiles-dir", str(tmp),
                "--project-dir",  str(tmp),
                "--no-use-colors",
                "--no-partial-parse",   # avoid stale manifest cache across rows
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    combined = proc.stdout + "\n" + proc.stderr
    errors, warnings = _extract_errors_warnings(combined)

    status = "pass" if proc.returncode == 0 else "fail"

    return {
        **row,
        "dbt_validation_status": status,
        "dbt_validation_errors": errors,
        "dbt_validation_warnings": warnings,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated dbt DAGs using dbt parse (no database required)."
    )
    parser.add_argument("input",  type=Path, help="JSONL file to validate")
    parser.add_argument("--output",  "-o", type=Path, help="Write annotated JSONL here")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--dbt", default=None,
                        help="Path to dbt executable (auto-detected if not set)")
    parser.add_argument("--only-failures", action="store_true",
                        help="Only print details for failed rows")
    args = parser.parse_args()

    # Auto-detect dbt from the venv next to this script
    if args.dbt is None:
        venv_dbt = Path(__file__).parent / ".venv" / "bin" / "dbt"
        args.dbt = str(venv_dbt) if venv_dbt.exists() else "dbt"

    # Load rows
    rows = [json.loads(ln) for ln in args.input.read_text().splitlines() if ln.strip()]
    total = len(rows)
    print(f"Validating {total} rows with `dbt parse` ({args.workers} workers) …\n")

    results: list[dict | None] = [None] * total

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(validate_row, row, args.dbt, i): i
            for i, row in enumerate(rows)
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                results[i] = {
                    **rows[i],
                    "dbt_validation_status": "error",
                    "dbt_validation_errors": [str(exc)],
                    "dbt_validation_warnings": [],
                }
            done += 1
            print(f"  {done}/{total}", end="\r", flush=True)

    print()  # newline after progress

    # ── Summary ──────────────────────────────────────────────────────────────
    counts: dict[str, int] = {"pass": 0, "fail": 0, "skipped": 0, "error": 0}
    for r in results:
        counts[r["dbt_validation_status"]] += 1  # type: ignore[index]

    attempted = counts["pass"] + counts["fail"]
    pass_rate = (counts["pass"] / attempted * 100) if attempted else 0.0

    print("─" * 60)
    print(f"  pass     {counts['pass']:>5}")
    print(f"  fail     {counts['fail']:>5}")
    print(f"  skipped  {counts['skipped']:>5}")
    if counts["error"]:
        print(f"  error    {counts['error']:>5}")
    print("─" * 60)
    print(f"  Pass rate: {pass_rate:.1f}%  ({attempted} attempted)\n")

    # ── Failure details ───────────────────────────────────────────────────────
    failures = [r for r in results if r["dbt_validation_status"] == "fail"]  # type: ignore[union-attr]
    if failures:
        show = failures if args.only_failures or not args.output else failures[:10]
        print(f"{'─' * 60}")
        print(f"Failures ({len(failures)} total, showing {len(show)}):")
        for r in show:
            q = r.get("question", "?")[:80]
            cplx = r.get("complexity", "")
            print(f"\n  [{cplx}] {q}")
            for e in r.get("dbt_validation_errors", [])[:5]:
                print(f"    ✗ {e}")
        print()

    # ── Write output ──────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved annotated results → {args.output}")

    sys.exit(0 if counts["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
