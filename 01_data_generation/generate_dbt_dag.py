#!/usr/bin/env python3
"""
Generate dbt DAGs from SQL queries using SQL parsing + template generation.
Claude Sonnet 4.6 inline generation - no external API required.
"""

import json
import re
import sys
from pathlib import Path

INPUT_FILE  = Path("data/selected_queries.jsonl")
OUTPUT_FILE = Path("data/dbt_dag_dataset_sonnet.jsonl")

# ─── SQL keywords (never treated as column names) ─────────────────────────────

SQL_KW = frozenset({
    'SELECT', 'DISTINCT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL',
    'CROSS', 'OUTER', 'ON', 'AND', 'OR', 'NOT', 'IN', 'AS', 'IS', 'NULL', 'TRUE',
    'FALSE', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET', 'TOP', 'WITH',
    'RECURSIVE', 'UNION', 'INTERSECT', 'EXCEPT', 'ALL', 'EXISTS', 'BETWEEN', 'LIKE',
    'ILIKE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'IF', 'COALESCE', 'NULLIF',
    'OVER', 'PARTITION', 'ROWS', 'RANGE', 'UNBOUNDED', 'PRECEDING', 'FOLLOWING',
    'CURRENT', 'ROW', 'ASC', 'DESC', 'NULLS', 'FIRST', 'LAST',
    'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'VARIANCE', 'TOTAL',
    'RANK', 'DENSE_RANK', 'ROW_NUMBER', 'NTILE', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE',
    'CAST', 'CONVERT', 'DATE', 'TIMESTAMP', 'INTERVAL', 'EXTRACT', 'YEAR', 'MONTH',
    'DAY', 'HOUR', 'MINUTE', 'SECOND', 'EPOCH',
    'INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT', 'FLOAT', 'DOUBLE', 'DECIMAL',
    'NUMERIC', 'VARCHAR', 'CHAR', 'TEXT', 'STRING', 'BOOLEAN', 'BOOL', 'REAL',
    'CREATE', 'TABLE', 'INDEX', 'VIEW', 'SCHEMA', 'DATABASE',
    'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'CONSTRAINT', 'CHECK', 'DEFAULT',
    'INSERT', 'UPDATE', 'DELETE', 'SET', 'INTO', 'VALUES',
    'TRIM', 'LOWER', 'UPPER', 'LENGTH', 'SUBSTR', 'SUBSTRING', 'CONCAT', 'REPLACE',
    'ROUND', 'FLOOR', 'CEIL', 'ABS', 'MOD', 'POWER', 'SQRT',
    'NOW', 'CURRENT_DATE', 'CURRENT_TIMESTAMP', 'STRFTIME', 'DATETIME',
    # common short aliases
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
    't1', 't2', 't3', 't4', 't5', 't6',
})


# ─── Utilities ────────────────────────────────────────────────────────────────

def to_snake(name: str) -> str:
    """CamelCase / PascalCase / UPPERCASE_ID → snake_case."""
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def clean_name(name: str) -> str:
    """Ensure a valid Python/SQL identifier."""
    n = re.sub(r'[^a-z0-9_]', '_', to_snake(name))
    return re.sub(r'_+', '_', n).strip('_') or 'col'


def parse_schema(context: str) -> dict[str, list[str]]:
    """Return {TableName: [col, ...]} from CREATE TABLE statements."""
    tables: dict[str, list[str]] = {}
    for m in re.finditer(r'CREATE\s+TABLE\s+(\w+)\s*\(([^)]+)\)', context, re.IGNORECASE):
        table = m.group(1)
        cols: list[str] = []
        for chunk in m.group(2).split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            first = chunk.split()[0] if chunk.split() else ''
            if first and first.upper() not in (
                'PRIMARY', 'FOREIGN', 'UNIQUE', 'KEY', 'INDEX', 'CONSTRAINT', 'CHECK',
            ):
                cols.append(first)
        if cols:
            tables[table] = cols
    return tables


def guess_pk(table: str, cols: list[str]) -> str | None:
    """Heuristically find the primary-key column."""
    t = table.upper()
    for col in cols:
        cu = col.upper()
        # Exact "ID"
        if cu == 'ID':
            return to_snake(col)
        # "TABLEID" or "TABLE_ID"
        if cu in (f'{t}ID', f'{t}_ID'):
            return to_snake(col)
    # Any column ending in _ID
    for col in cols:
        if col.upper().endswith('_ID') or col.upper().endswith('ID'):
            return to_snake(col)
    return to_snake(cols[0]) if cols else None


# ─── SQL analysis ─────────────────────────────────────────────────────────────

def analyze_sql(sql: str) -> dict:
    u = sql.upper()

    # Extract table aliases  alias_map[ALIAS_UPPER] = original_table_name
    alias_map: dict[str, str] = {}
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(\w+)(?:\s+AS\s+(\w+)|\s+([A-Za-z]\w*)(?=\s))?', sql, re.IGNORECASE):
        table = m.group(1)
        alias = m.group(2) or m.group(3) or ''
        alias_map[table.upper()] = table  # table maps to itself
        if alias and alias.upper() not in SQL_KW:
            alias_map[alias.upper()] = table

    from_tables: list[str] = re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
    join_tables: list[str] = re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
    all_tables: list[str] = list(dict.fromkeys(from_tables + join_tables))

    return {
        'all_tables':   all_tables,
        'alias_map':    alias_map,
        'has_join':     bool(re.search(r'\bJOIN\b', sql, re.IGNORECASE)),
        'has_group_by': bool(re.search(r'\bGROUP\s+BY\b', sql, re.IGNORECASE)),
        'has_having':   bool(re.search(r'\bHAVING\b', sql, re.IGNORECASE)),
        'has_where':    bool(re.search(r'\bWHERE\b', sql, re.IGNORECASE)),
        'has_intersect':bool(re.search(r'\bINTERSECT\b', sql, re.IGNORECASE)),
        'has_except':   bool(re.search(r'\bEXCEPT\b', sql, re.IGNORECASE)),
        'has_union':    bool(re.search(r'\bUNION\b', sql, re.IGNORECASE)),
        'has_subquery': bool(re.search(r'\bIN\s*\(\s*SELECT\b', sql, re.IGNORECASE)),
        'has_not_in':   bool(re.search(r'\bNOT\s+IN\s*\(\s*SELECT\b', sql, re.IGNORECASE)),
        'has_order_by': bool(re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE)),
        'has_limit':    bool(re.search(r'\bLIMIT\b', sql, re.IGNORECASE)),
        'has_distinct': bool(re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE)),
        'has_count':    bool(re.search(r'\bCOUNT\s*\(', sql, re.IGNORECASE)),
        'has_sum':      bool(re.search(r'\bSUM\s*\(', sql, re.IGNORECASE)),
        'has_avg':      bool(re.search(r'\bAVG\s*\(', sql, re.IGNORECASE)),
        'has_max_min':  bool(re.search(r'\b(?:MAX|MIN)\s*\(', sql, re.IGNORECASE)),
        'has_window':   bool(re.search(r'\bOVER\s*\(', sql, re.IGNORECASE)),
    }


def determine_mart(question: str, sql: str, st: dict, schema: dict) -> tuple[str, str, str]:
    """Return (mart_type, mart_name, grain)."""
    q = question.lower()
    tables = st['all_tables']

    # ── type ────────────────────────────────────────────────────────────────
    if st['has_intersect'] or st['has_except']:
        mart_type = 'rpt'
    elif st['has_not_in'] or st['has_subquery']:
        mart_type = 'rpt'
    elif st['has_order_by'] and st['has_limit']:
        mart_type = 'rpt'
    elif st['has_having'] and not (st['has_sum'] or st['has_avg']):
        mart_type = 'rpt'
    elif st['has_group_by'] and (st['has_count'] or st['has_sum'] or st['has_avg'] or st['has_max_min']):
        mart_type = 'fct'
    elif st['has_group_by']:
        mart_type = 'fct'
    elif re.match(r'how many', q):
        mart_type = 'fct'
    elif st['has_where'] and not st['has_join'] and not st['has_group_by']:
        mart_type = 'rpt' if not st['has_count'] else 'fct'
    elif st['has_join'] and not st['has_group_by']:
        mart_type = 'rpt'
    else:
        mart_type = 'dim'

    # ── grain ────────────────────────────────────────────────────────────────
    if st['has_group_by']:
        gb = re.search(r'\bGROUP\s+BY\s+(.+?)(?=\bHAVING\b|\bORDER\b|\bLIMIT\b|$)',
                       sql, re.IGNORECASE | re.DOTALL)
        if gb:
            first_gb = gb.group(1).strip().split(',')[0].split('.')[-1].strip()
            grain = f"1 row per {to_snake(first_gb)}"
        else:
            grain = "1 row per group"
    elif st['has_distinct']:
        sel = re.search(r'\bSELECT\s+DISTINCT\s+(.+?)\s+FROM\b', sql, re.IGNORECASE | re.DOTALL)
        if sel:
            c = sel.group(1).strip().split(',')[0].split('.')[-1].strip()
            grain = f"1 row per distinct {to_snake(c)}"
        else:
            grain = "1 row per distinct result"
    elif st['has_intersect'] or st['has_except']:
        grain = "1 row per distinct result in set operation"
    elif st['has_limit']:
        grain = "1 row (top result)"
    elif st['has_join'] and not st['has_group_by'] and len(tables) >= 2:
        grain = f"1 row per {to_snake(tables[0])}-{to_snake(tables[1])} pair"
    else:
        grain = "1 row per result"

    # ── name ────────────────────────────────────────────────────────────────
    primary = to_snake(tables[0]) if tables else 'result'

    if mart_type == 'fct':
        if st['has_sum']:
            m = re.search(r'\bSUM\s*\(\s*(?:\w+\.)?(\w+)\s*\)', sql, re.IGNORECASE)
            sfx = to_snake(m.group(1)) if m else 'totals'
            name = f"fct_{primary}_{sfx}_totals"
        elif st['has_avg']:
            name = f"fct_{primary}_averages"
        elif st['has_max_min']:
            name = f"fct_{primary}_extremes"
        else:
            name = f"fct_{primary}_counts"
    elif mart_type == 'dim':
        name = f"dim_{primary}"
    else:
        kw = _q_keywords(q)
        name = f"rpt_{'_'.join(kw[:3])}" if kw else f"rpt_{primary}_report"

    name = re.sub(r'_+', '_', re.sub(r'[^a-z0-9_]', '_', name)).strip('_')
    return mart_type, name, grain


def _q_keywords(q: str) -> list[str]:
    stop = {
        'what', 'are', 'the', 'a', 'an', 'of', 'is', 'show', 'find', 'list', 'all',
        'how', 'many', 'which', 'who', 'have', 'has', 'been', 'for', 'in', 'on', 'at',
        'and', 'or', 'with', 'not', 'each', 'per', 'by', 'do', 'does', 'where', 'when',
        'give', 'get', 'from', 'to', 'than', 'more', 'less', 'most', 'least', 'their',
        'them', 'he', 'she', 'it', 'its', 'this', 'those', 'that', 'also', 'using',
    }
    words = re.sub(r'[^a-z0-9\s]', ' ', q).split()
    return [w for w in words if w not in stop and len(w) > 2]


# ─── SQL translation ──────────────────────────────────────────────────────────

def translate_sql(sql: str, schema: dict, st: dict) -> str:
    """
    Translate original SQL to reference snake_case columns and CTE names.
    Returns a cleaned SELECT ... statement (no config/WITH block).
    """
    alias_map = st['alias_map']

    # Build col_map: (TABLE_UPPER, COL_UPPER) → snake_col
    col_map: dict[tuple, str] = {}
    all_cols: dict[str, str] = {}   # COL_UPPER → snake_col (fallback)
    for tname, cols in schema.items():
        for c in cols:
            sc = to_snake(c)
            col_map[(tname.upper(), c.upper())] = sc
            all_cols[c.upper()] = sc

    # Table → snake table (for CTE names)
    tbl_snake: dict[str, str] = {t.upper(): to_snake(t) for t in schema}

    # ── pass 1: resolve alias.col  ────────────────────────────────────────
    def repl_qualified(m: re.Match) -> str:
        prefix, col = m.group(1), m.group(2)
        orig_tbl = alias_map.get(prefix.upper())
        if orig_tbl:
            st_col = col_map.get((orig_tbl.upper(), col.upper()), to_snake(col))
            st_tbl = tbl_snake.get(orig_tbl.upper(), to_snake(orig_tbl))
            return f"{st_tbl}.{st_col}"
        # prefix is maybe a table
        for tname in schema:
            key = (tname.upper(), col.upper())
            if key in col_map:
                return f"{tbl_snake[tname.upper()]}.{col_map[key]}"
        return f"{to_snake(prefix)}.{to_snake(col)}"

    out = re.sub(r'\b(\w+)\.(\w+)\b', repl_qualified, sql)

    # ── pass 2: strip aliases from FROM/JOIN  ─────────────────────────────
    def repl_from(m: re.Match) -> str:
        kw  = m.group(1)   # e.g. "FROM" / "LEFT JOIN"
        tbl = m.group(2)
        orig = alias_map.get(tbl.upper(), tbl)
        snake = tbl_snake.get(orig.upper(), to_snake(orig))
        return f"{kw} {snake}"

    # Remove "FROM/JOIN table [AS] alias" alias portion
    out = re.sub(
        r'((?:FROM'
        r'|(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+(?:OUTER\s+)?JOIN'
        r'|JOIN))\s+(\w+)'
        r'(?:\s+AS\s+\w+|\s+(?!(?:ON|WHERE|AND|OR|INNER|LEFT|RIGHT|FULL|CROSS'
        r'|JOIN|GROUP|ORDER|HAVING|LIMIT|SET|\Z)\b)[A-Za-z]\w*)?',
        repl_from, out, flags=re.IGNORECASE,
    )

    # ── pass 3: replace bare column references  ───────────────────────────
    def repl_bare(m: re.Match) -> str:
        w = m.group(0)
        if w.upper() in SQL_KW:
            return w
        return all_cols.get(w.upper(), w)

    out = re.sub(r'\b[A-Za-z_]\w*\b', repl_bare, out)

    # ── pass 4: light formatting  ──────────────────────────────────────────
    out = re.sub(r'\s+', ' ', out).strip()
    for kw in ('SELECT', 'FROM', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
                'FULL JOIN', 'CROSS JOIN', 'JOIN', 'WHERE', 'GROUP BY',
                'HAVING', 'ORDER BY', 'LIMIT', 'UNION ALL', 'UNION',
                'INTERSECT', 'EXCEPT'):
        out = re.sub(rf'\b{kw}\b', f'\n{kw}', out, flags=re.IGNORECASE)

    # Indent SELECT columns
    lines, formatted = out.split('\n'), []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        u = line.upper()
        if u.startswith(('SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP', 'HAVING',
                          'ORDER', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT',
                          'INNER', 'LEFT', 'RIGHT', 'FULL', 'CROSS')):
            formatted.append(line)
        else:
            formatted.append('    ' + line)

    return '\n'.join(formatted).strip()


# ─── File generators ──────────────────────────────────────────────────────────

def gen_staging_sql(table: str, cols: list[str]) -> str:
    lines = []
    for c in cols:
        sc = to_snake(c)
        lines.append(f"    {c} AS {sc}" if c != sc else f"    {c}")
    body = ',\n'.join(lines)
    return f"""{{% config(materialized='view') %}}

SELECT
{body}
FROM {{{{ source('raw', '{table}') }}}}
"""


def gen_staging_sources_yml(schema: dict) -> str:
    tbl_entries = []
    for t, cols in schema.items():
        pk = guess_pk(t, cols)
        col_tests = ''
        if pk:
            col_tests = f"""      columns:
        - name: {pk}
          tests:
            - not_null
            - unique
"""
        tbl_entries.append(f"      - name: {t}\n{col_tests}" if col_tests else f"      - name: {t}")
    tables_block = '\n'.join(tbl_entries)
    return f"""version: 2

sources:
  - name: raw
    tables:
{tables_block}
"""


def gen_staging_models_yml(schema: dict) -> str:
    model_blocks = []
    for t, cols in schema.items():
        snake_t = to_snake(t)
        pk = guess_pk(t, cols)
        col_lines = []
        for c in cols:
            sc = to_snake(c)
            if sc == pk:
                col_lines.append(
                    f"      - name: {sc}\n"
                    f"        tests:\n"
                    f"          - not_null\n"
                    f"          - unique"
                )
            else:
                col_lines.append(f"      - name: {sc}")
        cols_block = '\n'.join(col_lines)
        model_blocks.append(
            f"  - name: stg_{snake_t}\n"
            f"    columns:\n"
            f"{cols_block}"
        )
    return f"""version: 2

models:
{''.join(m + chr(10) for m in model_blocks)}"""


def gen_mart_sql(sql: str, schema: dict, st: dict, mart_name: str) -> str:
    tables = [t for t in st['all_tables'] if t in schema]
    if not tables:
        tables = list(schema.keys())

    # Import CTEs
    ctes = []
    for t in tables:
        s = to_snake(t)
        ctes.append(f"""{s} AS (

    SELECT *
    FROM {{{{ ref('stg_{s}') }}}}

)""")

    body = translate_sql(sql, schema, st)

    ctes_sql = ',\n\n'.join(ctes)
    return f"""{{% config(materialized='table') %}}

WITH {ctes_sql}

{body}
"""


def gen_core_models_yml(
    mart_name: str, mart_type: str, grain: str,
    question: str, sql: str, schema: dict, st: dict
) -> str:
    # Collect SELECT columns from mart
    sel_m = re.search(r'\bSELECT\b\s+(.*?)\s+\bFROM\b', sql, re.IGNORECASE | re.DOTALL)
    mart_cols = []
    if sel_m:
        raw_cols = sel_m.group(1).strip()
        for part in raw_cols.split(','):
            part = part.strip()
            if not part or part.upper() == 'DISTINCT':
                continue
            # AS alias → take alias
            alias_m = re.search(r'\bAS\s+(\w+)\s*$', part, re.IGNORECASE)
            if alias_m:
                mart_cols.append(to_snake(alias_m.group(1)))
                continue
            # table.col → col
            dot_m = re.search(r'\w+\.(\w+)\s*$', part)
            if dot_m:
                c = dot_m.group(1)
                mart_cols.append(to_snake(c))
                continue
            # plain col / function
            plain = part.split('(')[-1].rstrip(')').strip()
            if plain and plain.upper() not in SQL_KW:
                mart_cols.append(to_snake(plain))

    if not mart_cols:
        # fallback: all cols from first table
        t0 = list(schema.values())[0]
        mart_cols = [to_snake(c) for c in t0[:3]]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_cols = []
    for c in mart_cols:
        if c and c not in seen:
            seen.add(c)
            unique_cols.append(c)
    mart_cols = unique_cols

    # Determine PK col for tests
    pk_col = mart_cols[0] if mart_cols else None

    # Human-readable description
    desc = question.rstrip('?').strip()
    desc = desc[0].upper() + desc[1:] if desc else "Business metric."

    type_label = {'fct': 'fact', 'dim': 'dimension', 'rpt': 'report'}[mart_type]

    col_lines = []
    for i, c in enumerate(mart_cols):
        if i == 0 and pk_col:
            col_lines.append(
                f"      - name: {c}\n"
                f"        tests:\n"
                f"          - not_null"
            )
        else:
            col_lines.append(f"      - name: {c}")
    cols_block = '\n'.join(col_lines) if col_lines else '      - name: result'

    return f"""version: 2

models:
  - name: {mart_name}
    description: "{desc}."
    meta:
      model_type: {type_label}
      grain: "{grain}"
    columns:
{cols_block}
"""


# ─── DAG assembler ────────────────────────────────────────────────────────────

def generate_dbt_dag(question: str, context: str, sql: str) -> str | None:
    """Return XML-tagged dbt DAG string, or None on failure."""
    schema = parse_schema(context)
    if not schema:
        return None

    st = analyze_sql(sql)
    mart_type, mart_name, grain = determine_mart(question, sql, st, schema)

    files: list[tuple[str, str]] = []

    # Staging SQL (one per table)
    for tname, cols in schema.items():
        snake_t = to_snake(tname)
        files.append((f"models/staging/stg_{snake_t}.sql", gen_staging_sql(tname, cols)))

    # Staging YAML
    files.append(("models/staging/_staging_sources.yml", gen_staging_sources_yml(schema)))
    files.append(("models/staging/_staging_models.yml", gen_staging_models_yml(schema)))

    # Mart SQL + YAML
    files.append((f"models/marts/core/{mart_name}.sql", gen_mart_sql(sql, schema, st, mart_name)))
    files.append(("models/marts/core/_core_models.yml",
                  gen_core_models_yml(mart_name, mart_type, grain, question, sql, schema, st)))

    parts = [f'<file path="{path}">\n{content.strip()}\n</file>' for path, content in files]
    return '\n\n'.join(parts)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_dag(dag: str) -> bool:
    if not dag:
        return False
    files = re.findall(r'<file\s+path="([^"]+)">(.*?)</file>', dag, re.DOTALL)
    if not files:
        return False
    has_stg_sql = has_mart_sql = has_source = has_ref = has_config = False
    has_src_yml = has_stg_yml = has_core_yml = False
    for path, content in files:
        if path.endswith('.sql'):
            if '/staging/' in path:
                has_stg_sql = True
            if '/marts/' in path:
                has_mart_sql = True
            if 'source(' in content:
                has_source = True
            if 'ref(' in content:
                has_ref = True
            if 'config(' in content:
                has_config = True
            if 'SELECT' not in content.upper():
                return False
        elif path.endswith(('.yml', '.yaml')):
            if '_staging_sources' in path:
                has_src_yml = True
            if '_staging_models' in path:
                has_stg_yml = True
            if '_core_models' in path:
                has_core_yml = True
    return has_stg_sql and has_mart_sql and has_source and has_ref and has_config \
           and has_src_yml and has_stg_yml and has_core_yml


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load input
    rows = [json.loads(l) for l in INPUT_FILE.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} rows from {INPUT_FILE}", flush=True)

    # Checkpoint
    done: set[str] = set()
    if OUTPUT_FILE.exists():
        for l in OUTPUT_FILE.read_text().splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
                if r.get('status') == 'success':
                    done.add(r['question'])
            except json.JSONDecodeError:
                pass
    if done:
        print(f"Checkpoint: {len(done)} rows already done — resuming…", flush=True)

    pending = [r for r in rows if r['question'] not in done]
    if not pending:
        print("All rows already processed.")
        return

    print(f"Processing {len(pending)} rows…", flush=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    success = failed = 0
    n_done = len(done)
    total = len(pending)

    with OUTPUT_FILE.open('a', encoding='utf-8') as f:
        for i, row in enumerate(pending, 1):
            try:
                dag = generate_dbt_dag(row['question'], row['context'], row['answer'])
                if dag and validate_dag(dag):
                    out = {k: v for k, v in row.items() if k != 'answer'}
                    out['original_sql'] = row['answer']
                    out['dbt_dag'] = dag
                    out['status'] = 'success'
                    f.write(json.dumps(out, ensure_ascii=False) + '\n')
                    f.flush()
                    success += 1
                else:
                    failed += 1
                    if dag:
                        print(f"  [{n_done+i}/{n_done+total}] ✗ validation failed", flush=True)
                    else:
                        print(f"  [{n_done+i}/{n_done+total}] ✗ generation failed", flush=True)
            except Exception as e:
                failed += 1
                print(f"  [{n_done+i}/{n_done+total}] ✗ error: {e}", flush=True)

            if i % 50 == 0 or i == total:
                pct = 100 * success / i
                print(f"  Progress {n_done+i}/{n_done+total} | ✓ {success}  ✗ {failed}  ({pct:.0f}%)", flush=True)

    print(f"\nDone. {success}/{success+failed} rows succeeded ({100*success/(success+failed) if success+failed else 0:.0f}%)")


if __name__ == '__main__':
    main()
