#!/usr/bin/env python3
"""
Convert finetune_dataset.jsonl → Mistral chat-format JSONL for SFT.

Output format (one JSON object per line):
  {"messages": [
      {"role": "system",  "content": SYSTEM_PROMPT},
      {"role": "user",    "content": "<question>\n\n<sql_schemas>"},
      {"role": "assistant","content": "<dbt_dag XML>"}
  ]}

Also writes a train/eval split (90/10 by default).

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --split 0.05   # 5% eval
    python prepare_dataset.py --input ../01_data_generation/data/finetune_dataset.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

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


def build_user_message(row: dict) -> str:
    return f"Business question: {row['question']}\n\nSQL schemas:\n{row['context']}"


def convert(input_path: Path, output_dir: Path, eval_split: float, seed: int) -> None:
    rows = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} rows from {input_path}")

    random.seed(seed)
    random.shuffle(rows)

    n_eval = max(1, int(len(rows) * eval_split))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]
    print(f"Split: {len(train_rows)} train / {len(eval_rows)} eval")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in [("train", train_rows), ("eval", eval_rows)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with out_path.open("w") as f:
            for row in split_rows:
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_message(row)},
                        {"role": "assistant", "content": row["dbt_dag"]},
                    ]
                }
                f.write(json.dumps(record) + "\n")
        print(f"  → {out_path}  ({len(split_rows)} rows)")

    # Token estimate
    all_chars = sum(
        len(SYSTEM_PROMPT) + len(build_user_message(r)) + len(r["dbt_dag"])
        for r in rows
    )
    print(f"\nEst. total tokens (chars/4): {all_chars // 4:,}")
    print(f"Est. avg tokens per row:     {all_chars // 4 // len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../01_data_generation/data/finetune_dataset.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=float, default=0.10, help="Eval fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert(args.input, args.output_dir, args.split, args.seed)


if __name__ == "__main__":
    main()
