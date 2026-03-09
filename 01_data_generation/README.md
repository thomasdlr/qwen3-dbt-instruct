# dbt Synthetic Dataset Generator

Fine-tuning dataset pipeline: SQL queries → multi-file dbt DAGs (SQL + YAML).
Source dataset: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (HuggingFace).

---

## Prerequisites

```bash
uv sync                  # install all dependencies including dbt-duckdb
cp .env.example .env     # set LOCAL_LM_BASE_URL and LM_API_KEY
```

Start LM Studio and load the model before running any transform step.

---

## Pipeline — run in order

### Step 1 — Select queries
Downloads the HuggingFace dataset, scores complexity (24 features), filters data
quality issues, and outputs a balanced 1 000-query sample.

```bash
uv run select_queries.py --target 1000 --output data/selected_queries.jsonl --stats
```

Output: `data/selected_queries.jsonl`
Fields: `question`, `context`, `answer`, `complexity`, `complexity_score`, `features`, `table_count`
Distribution: 10% simple / 25% moderate / 40% complex / 25% advanced, all with 2+ tables.

---

### Step 2 — (Optional) Smoke test the LLM
Run the transformer on a small slice before committing to the full dataset.

```bash
uv run test_transform.py --complexity advanced --limit 5 --output data/results_smoke.jsonl
```

Review the output visually (see Step 4) before proceeding.

---

### Step 3 — Generate the full dataset
Reads `data/selected_queries.jsonl` and calls the local LLM to transform each
query into a multi-file dbt DAG (staging SQL + YAML + intermediate + marts).
Each row is written immediately — **stop anytime with Ctrl+C and re-run the
exact same command to resume** from where it left off.

```bash
uv run transform_sql_to_dbt.py \
    --sample-size 1000 \
    --concurrency 4 \
    --output data/dbt_dag_dataset.jsonl
```

Output: `data/dbt_dag_dataset.jsonl`
Fields: everything from Step 1 + `original_sql`, `dbt_dag` (XML-tagged file contents), `status`.

Checkpoint: rows already present in the output file with `status=success` are
automatically skipped on the next run.

---

### Step 4 — Validate with dbt parse
Scaffolds each DAG into a temp dbt project and runs `dbt parse` (no database
required) to catch ref/source resolution errors, YAML syntax issues, and
circular dependencies.

```bash
uv run validate_dbt_dag.py data/dbt_dag_dataset.jsonl \
    --workers 8 \
    --output data/validated_dataset.jsonl
```

Output: `data/validated_dataset.jsonl`
Added fields: `dbt_validation_status` (pass/fail/skipped), `dbt_validation_errors`, `dbt_validation_warnings`.  
Exit code `1` if any row fails — usable in CI.

---

### Step 5 — Human review
TUI to browse, inspect, and annotate any JSONL file row by row.
SQL files render with blue borders; YAML files with green borders.

```bash
uv run review_dataset.py data/validated_dataset.jsonl
```

Keybindings: `←/→` navigate · `a` approve · `r` reject · `f` flag · `s` skip · `e` edit-needed · `u` clear · `n` next unannotated · `q` quit  
Annotations are saved to `<file>.annotations.json` (gitignored).

---

## Data files

| File | Description |
|---|---|
| `data/selected_queries.jsonl` | 1 000 scored + filtered source queries |
| `data/dbt_dag_dataset.jsonl` | Generated dbt DAGs (raw, pre-validation) |
| `data/validated_dataset.jsonl` | DAGs annotated with dbt parse results |

---

## Scripts at a glance

| Script | Role |
|---|---|
| `select_queries.py` | Query selection + complexity scoring |
| `transform_sql_to_dbt.py` | LLM transformer (SQL → dbt DAG) |
| `test_transform.py` | Smoke test: run transformer on a small slice |
| `validate_dbt_dag.py` | dbt parse validation (no DB needed) |
| `review_dataset.py` | Interactive TUI reviewer + annotator |
