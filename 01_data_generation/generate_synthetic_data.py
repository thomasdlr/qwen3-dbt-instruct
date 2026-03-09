#!/usr/bin/env python3
"""
Synthetic data generator for text-to-dbt fine-tuning.

Uses a local LLM via LM Studio to generate diverse (user_prompt, dbt_model) pairs
and saves them as a JSONL file ready for Mistral fine-tuning.

Configuration is loaded from a .env file (see .env.example).

Usage:
    uv run generate_synthetic_data.py
    uv run generate_synthetic_data.py --variations 5 --concurrency 2 --output data/my_dataset.jsonl
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
from datasets import load_dataset
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

# Load from .env in the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

LOCAL_LM_BASE_URL = os.getenv("LOCAL_LM_BASE_URL", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
DEFAULT_OUTPUT = Path(
    os.getenv("OUTPUT_FILE", "01_data_generation/data/dbt_synthetic_dataset.jsonl")
)

# Tune --concurrency based on your Mac (2 for M1 Pro, 4–8 for M-Max/Ultra)
DEFAULT_CONCURRENCY = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
DEFAULT_VARIATIONS = int(os.getenv("DEFAULT_VARIATIONS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# ── Sampling parameters ──────────────────────────────────────────────────────
# temperature      – randomness scale; lower = more deterministic SQL output
# top_p            – nucleus sampling; only tokens summing to this probability are considered
# top_k            – hard cap on candidate tokens per step; keeps SQL syntax tight
# min_p            – drops tokens below this fraction of the best-token probability (0.0 = off)
# presence_penalty – discourages tokens already present (0.0 = off, safe for SQL keywords)
# repetition_penalty – llama.cpp multiplicative repeat penalty (1.0 = neutral)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "20"))
MIN_P = float(os.getenv("MIN_P", "0.0"))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", "0.0"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.0"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "true").lower() == "true"

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── LM Studio client (shared across coroutines) ──────────────────────────────

client = AsyncOpenAI(
    base_url=LOCAL_LM_BASE_URL,
    api_key=LM_API_KEY,  # Required by the SDK but ignored by LM Studio
)

# ─── Seed scenarios ───────────────────────────────────────────────────────────
# Each seed provides a business domain, available dbt source tables, key dbt
# concepts to exercise, and a high-level business task.  The local model will
# freely paraphrase the task and write a full dbt model, so multiple
# --variations per seed produce genuinely different training examples.

SEED_SCENARIOS: list[dict] = [
    {
        "domain": "E-commerce",
        "sources": ["orders", "customers", "products", "order_items", "returns"],
        "dbt_concepts": ["incremental model", "aggregation", "multi-table join"],
        "task": "Calculate monthly revenue per customer segment, excluding returned items",
    },
    {
        "domain": "SaaS",
        "sources": ["subscriptions", "users", "events", "invoices", "plans"],
        "dbt_concepts": ["CTE chaining", "window functions", "date filtering"],
        "task": "Identify churned customers as those with no activity in the last 30 days",
    },
    {
        "domain": "Finance",
        "sources": ["transactions", "accounts", "categories", "budgets"],
        "dbt_concepts": [
            "running totals",
            "period-over-period comparison",
            "aggregation",
        ],
        "task": "Compute monthly spending by category compared to budget targets",
    },
    {
        "domain": "Marketing",
        "sources": ["campaigns", "clicks", "conversions", "ad_spend", "channels"],
        "dbt_concepts": ["attribution", "funnel analysis", "aggregation"],
        "task": "Calculate ROI and conversion rate by marketing channel and campaign",
    },
    {
        "domain": "HR",
        "sources": ["employees", "departments", "salaries", "performance_reviews"],
        "dbt_concepts": ["self-join / hierarchy", "aggregation", "conditional logic"],
        "task": "Compute average salary and headcount growth quarter-over-quarter by department",
    },
    {
        "domain": "Logistics",
        "sources": ["shipments", "warehouses", "carriers", "delivery_events"],
        "dbt_concepts": ["SLA tracking", "date arithmetic", "status pivoting"],
        "task": "Track on-time delivery rate and average transit time per carrier",
    },
    {
        "domain": "Product Analytics",
        "sources": ["sessions", "pageviews", "users", "feature_flags", "events"],
        "dbt_concepts": ["sessionization", "cohort analysis", "retention"],
        "task": "Build a weekly user retention cohort model grouped by signup month",
    },
    {
        "domain": "Healthcare",
        "sources": ["patients", "appointments", "diagnoses", "procedures", "providers"],
        "dbt_concepts": ["aggregation", "date spine", "conditional pivoting"],
        "task": "Summarize appointment volume and no-show rate by provider and specialty",
    },
    {
        "domain": "Retail",
        "sources": ["stores", "sales", "inventory", "suppliers", "products"],
        "dbt_concepts": ["slowly changing dimensions", "stock alerts", "aggregation"],
        "task": "Flag products with fewer than 10 units in stock and their last restock date",
    },
    {
        "domain": "Banking",
        "sources": ["accounts", "transactions", "loans", "customers", "risk_scores"],
        "dbt_concepts": ["risk scoring", "window functions", "multi-join"],
        "task": "Compute 90-day average balance and flag accounts with unusual transaction spikes",
    },
]

# ─── Prompts ──────────────────────────────────────────────────────────────────

# System prompt for SQL → dbt transformation
TRANSFORMATION_SYSTEM_PROMPT = """\
You are an expert Analytics Engineer specializing in dbt (data build tool).

Your task: Transform vanilla SQL queries into production-ready dbt models.

### Transformation rules (FOLLOW STRICTLY):

1. **Wrap ALL table references** in {{ source('schema', 'table_name') }}
   - Infer a logical schema name from context (e.g., 'raw', 'analytics', 'sales')
   - Example: `FROM users` → `FROM {{ source('raw', 'users') }}`

2. **Refactor into CTEs** if the query has subqueries or complexity
   - Use descriptive CTE names (e.g., `active_users`, `monthly_revenue`)
   - Keep simple queries simple

3. **Add SQL comments** for non-obvious business logic

4. **Keep all column references** exactly as they appear in the original SQL

5. **Preserve the query logic** - only change syntax, not semantics

6. **Output format**: Return ONLY the dbt SQL model as plain text
   - NO JSON, NO markdown fences, NO explanations
   - Just the SQL with dbt macros

### Example:
Input SQL:
```
SELECT u.user_id, u.email, COUNT(o.order_id) AS order_count
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE u.status = 'active'
GROUP BY u.user_id, u.email
```

Output dbt model:
```
WITH active_users AS (
    SELECT user_id, email, status
    FROM {{ source('raw', 'users') }}
    WHERE status = 'active'
),
user_orders AS (
    SELECT
        u.user_id,
        u.email,
        COUNT(o.order_id) AS order_count
    FROM active_users u
    LEFT JOIN {{ source('raw', 'orders') }} o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.email
)
SELECT * FROM user_orders
```
"""

# System prompt embedded in the *fine-tuning dataset* (what the fine-tuned model will see)
FINETUNE_SYSTEM_PROMPT = """\
You are an expert Analytics Engineer specialising in dbt (data build tool).
Convert natural language data transformation requests into production-ready dbt models.
You are given the source table schemas as CREATE TABLE statements.
Always use named CTEs, {{ ref() }} macros for model references, and \
{{ source() }} macros for raw tables.
Only reference columns that exist in the provided schemas."""


def _build_user_message(scenario: dict) -> str:
    return (
        f"Domain: {scenario['domain']}\n"
        f"Available source tables: {', '.join(scenario['sources'])}\n"
        f"Key dbt concepts to demonstrate: {', '.join(scenario['dbt_concepts'])}\n"
        f"Business task: {scenario['task']}\n\n"
        "Generate one realistic training example."
    )


# ─── Data model ───────────────────────────────────────────────────────────────


@dataclass
class TrainingExample:
    user_prompt: str
    context: str  # CREATE TABLE statements grounding the dbt model
    dbt_model: str

    def to_jsonl_record(self) -> dict:
        """SQL-create-context format: question, context, answer."""
        return {
            "question": self.user_prompt,
            "context": self.context,
            "answer": self.dbt_model,
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _strip_reasoning(text: str) -> str:
    """Remove reasoning/thinking preambles emitted by reasoning models."""
    # Remove <think>...</think> or <thinking>...</thinking> blocks
    text = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    # If the response starts with a prose preamble (no leading '{'), drop everything before the first '{'
    first_brace = text.find("{")
    if first_brace > 0:
        # Only strip if there's a meaningful amount of prose before the JSON
        preamble = text[:first_brace]
        if len(preamble.strip()) > 20:
            text = text[first_brace:]
    return text.strip()


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from model output, tolerating markdown fences and reasoning traces."""
    text = _strip_reasoning(text)

    # Direct parse (ideal case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip ```json ... ``` fences (with greedy matching to capture full object)
    match = re.search(r"```(?:json)?\s*(\{.+\})\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try fixing common escaping issues
            text = candidate

    # Fallback: grab the outermost { ... } block (greedy to get full object)
    if not match:
        match = re.search(r"\{.+\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    # Try to fix unescaped newlines in string values:
    # Replace literal newlines inside quoted strings with \n
    try:
        # This is a crude fix: split on quotes and fix newlines only within strings
        parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
        fixed_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Inside a quoted string
                # Replace literal newlines/tabs with escaped versions
                part = (
                    part.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
                )
            fixed_parts.append(part)
        text = "".join(fixed_parts)
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def _is_valid_dbt_model(model: str) -> bool:
    """Ensure the generated SQL contains at least one dbt macro and a SELECT, and no SELECT *."""
    has_macro = "ref(" in model or "source(" in model
    has_select = "SELECT" in model.upper()
    has_bare_select_star = re.search(r"SELECT\s+\*(?:\s|,|FROM)", model, re.IGNORECASE)
    return has_macro and has_select and not has_bare_select_star


# ─── Async generation ─────────────────────────────────────────────────────────


def _make_retry_decorator(semaphore_ref: asyncio.Semaphore):
    """Build the @retry decorator so it can reference the runtime semaphore."""

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        # Don't retry hard model crashes (400); only retry transient network/timeout errors
        retry=retry_if_exception_type(
            (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError)
        ),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    async def _generate(scenario: dict, label: str) -> Optional[TrainingExample]:
        async with semaphore_ref:
            log.info(f"{label} – generating…")
            try:
                response = await client.chat.completions.create(
                    model="local-model",  # LM Studio ignores the model name
                    messages=[
                        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_message(scenario)},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    presence_penalty=PRESENCE_PENALTY,
                    max_tokens=MAX_TOKENS,
                    # LM Studio / llama.cpp extensions (not in the OpenAI spec)
                    extra_body={
                        "top_k": TOP_K,
                        "min_p": MIN_P,
                        "repetition_penalty": REPETITION_PENALTY,
                        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
                    },
                )
            except BadRequestError as e:
                # 400 usually means the model crashed in LM Studio – log and skip, don't retry
                log.error(f"{label} – model returned 400 (crash?), skipping: {e}")
                return None
            raw = response.choices[0].message.content.strip()

        parsed = _extract_json(raw)
        if not parsed:
            log.warning(
                f"{label} – could not parse JSON from response.\n"
                f"=== RAW RESPONSE ===\n{raw}\n=== END RAW RESPONSE ==="
            )
            return None

        user_prompt = parsed.get("user_prompt", "").strip()
        context = parsed.get("context", "").strip()
        dbt_model = parsed.get("dbt_model", "").strip()

        if not user_prompt or not context or not dbt_model:
            log.warning(
                f"{label} – missing fields in parsed response. Keys found: {list(parsed.keys())}"
            )
            return None

        if "CREATE TABLE" not in context.upper():
            log.warning(
                f"{label} – context missing CREATE TABLE statements.\n"
                f"=== RAW CONTEXT ===\n{context}\n=== END CONTEXT ==="
            )
            return None

        if not _is_valid_dbt_model(dbt_model):
            log.warning(
                f"{label} – dbt model failed validation (no ref/source or SELECT).\n"
                f"=== RAW DBT MODEL ===\n{dbt_model}\n=== END DBT MODEL ==="
            )
            return None

        log.info(f"{label} ✓")
        return TrainingExample(
            user_prompt=user_prompt, context=context, dbt_model=dbt_model
        )

    return _generate


async def generate_dataset(
    scenarios: list[dict],
    variations_per_scenario: int,
    concurrency: int,
) -> list[TrainingExample]:
    semaphore = asyncio.Semaphore(concurrency)
    _generate = _make_retry_decorator(semaphore)

    tasks = [
        _generate(
            scenario,
            label=f"[{s_idx}.{v}] {scenario['domain']}",
        )
        for s_idx, scenario in enumerate(scenarios)
        for v in range(variations_per_scenario)
    ]

    total = len(tasks)
    log.info(
        f"Launching {total} generation tasks "
        f"({concurrency} max concurrent, {MAX_RETRIES} retries each)…"
    )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    examples: list[TrainingExample] = []
    for result in results:
        if isinstance(result, Exception):
            log.error(f"Task failed permanently: {result}")
        elif result is not None:
            examples.append(result)

    return examples


# ─── Output ───────────────────────────────────────────────────────────────────


def save_dataset(examples: list[TrainingExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_jsonl_record(), ensure_ascii=False) + "\n")
    log.info(f"Saved {len(examples)} examples → {output_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic text-to-dbt training data using a local LLM."
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=DEFAULT_VARIATIONS,
        help=f"Variations per seed scenario (default: {DEFAULT_VARIATIONS})",
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
        help=f"Max concurrent LLM requests — tune for your Mac (default: {DEFAULT_CONCURRENCY})",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    examples = await generate_dataset(
        scenarios=SEED_SCENARIOS,
        variations_per_scenario=args.variations,
        concurrency=args.concurrency,
    )

    if not examples:
        log.error(
            "No valid examples generated. Is LM Studio running at %s?",
            LOCAL_LM_BASE_URL,
        )
        return

    save_dataset(examples, args.output)

    total_tasks = len(SEED_SCENARIOS) * args.variations
    log.info(
        "Done. %d/%d examples generated successfully (%.0f%% success rate).",
        len(examples),
        total_tasks,
        100 * len(examples) / total_tasks,
    )


if __name__ == "__main__":
    asyncio.run(main())
