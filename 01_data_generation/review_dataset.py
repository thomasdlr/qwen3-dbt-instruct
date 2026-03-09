#!/usr/bin/env python3
"""
Interactive JSONL dataset reviewer for quality monitoring.

Displays each row with syntax highlighting, keyboard navigation,
and persistent annotation support.

Usage:
    uv run review_dataset.py data/test_dag.jsonl
    uv run review_dataset.py data/test_dag.jsonl --annotations my_notes.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

# ─── Annotation labels ───────────────────────────────────────────────────────

LABELS: dict[str, tuple[str, str, str]] = {
    # key: (name, badge_markup, style)
    "a": ("approved", "✅ APPROVED", "green bold"),
    "r": ("rejected", "❌ REJECTED", "red bold"),
    "f": ("flagged", "🚩 FLAGGED", "yellow bold"),
    "s": ("skipped", "⏭  SKIPPED", "dim"),
    "e": ("edit needed", "✏️  EDIT NEEDED", "magenta bold"),
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def parse_file_tags(text: str) -> list[tuple[str, str]]:
    """Extract (filepath, content) pairs from <file path="...">...</file> blocks."""
    pattern = r'<file\s+path="([^"]+)">\s*\n?(.*?)\s*</file>'
    return [
        (path, content.strip())
        for path, content in re.findall(pattern, text, re.DOTALL)
    ]


def detect_language(filepath: str) -> str:
    """Guess syntax language from file extension."""
    if filepath.endswith(".sql"):
        return "sql"
    if filepath.endswith((".yml", ".yaml")):
        return "yaml"
    if filepath.endswith(".py"):
        return "python"
    return "text"


def is_yaml_file(filepath: str) -> bool:
    return filepath.endswith((".yml", ".yaml"))


def looks_like_sql(text: str) -> bool:
    """Heuristic: does this text look like SQL?"""
    return bool(re.search(r"\b(SELECT|CREATE\s+TABLE|INSERT|UPDATE|DELETE)\b", text, re.IGNORECASE))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_annotations(path: Path) -> dict[int, str]:
    if path.exists():
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_annotations(path: Path, annotations: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {str(k): v for k, v in sorted(annotations.items())},
            f,
            indent=2,
        )


def annotation_summary(annotations: dict[int, str], total: int) -> str:
    """Build a compact summary string of annotation counts."""
    counts: dict[str, int] = {}
    for v in annotations.values():
        counts[v] = counts.get(v, 0) + 1
    parts = []
    for key, (name, _, _) in LABELS.items():
        c = counts.get(key, 0)
        if c:
            parts.append(f"{name}: {c}")
    remaining = total - len(annotations)
    parts.append(f"remaining: {remaining}")
    return "  │  ".join(parts)


# ─── Widgets ──────────────────────────────────────────────────────────────────


class RowViewer(Static):
    """Renders a single JSONL row with rich formatting and syntax highlighting."""

    def __init__(self, row: dict, index: int, total: int, annotation: str | None):
        super().__init__()
        self.row = row
        self.index = index
        self.total = total
        self.annotation = annotation

    def _render_value(self, key: str, value: str) -> list:
        """Render a field value, with special handling for dbt file tags and SQL."""
        parts: list = []

        # Try parsing <file> XML tags for answer-like fields
        if key in ("answer", "output", "response", "completion", "dbt_dag"):
            files = parse_file_tags(value)
            if files:
                # Group into SQL files and YAML files for cleaner display
                sql_files = [(p, c) for p, c in files if not is_yaml_file(p)]
                yml_files = [(p, c) for p, c in files if is_yaml_file(p)]

                for filepath, content in sql_files:
                    lang = detect_language(filepath)
                    parts.append(
                        Panel(
                            Syntax(
                                content,
                                lang,
                                theme="monokai",
                                line_numbers=True,
                                word_wrap=True,
                            ),
                            title=f"[dim italic]{filepath}[/]",
                            border_style="blue",
                            padding=(0, 1),
                        )
                    )
                for filepath, content in yml_files:
                    parts.append(
                        Panel(
                            Syntax(
                                content,
                                "yaml",
                                theme="monokai",
                                line_numbers=True,
                                word_wrap=True,
                            ),
                            title=f"[dim italic]{filepath}[/]",
                            border_style="green",
                            padding=(0, 1),
                        )
                    )
                return parts

        # SQL heuristic
        if looks_like_sql(value):
            parts.append(
                Panel(
                    Syntax(value, "sql", theme="monokai", word_wrap=True),
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        else:
            # Plain text — just render with proper newlines
            parts.append(
                Panel(Text(value), border_style="dim", padding=(0, 1))
            )
        return parts

    def render(self):
        parts: list = []

        # ── Row header with annotation badge ──
        header = Text(f"  Row {self.index + 1} / {self.total}", style="bold cyan")
        if self.annotation and self.annotation in LABELS:
            _, badge, style = LABELS[self.annotation]
            header.append("    ")
            header.append(badge, style=style)
        parts.append(header)
        parts.append(Text(""))

        # ── Render each field ──
        for key, value in self.row.items():
            # Section label
            parts.append(Text(f"  ▎ {key.upper()}", style="bold yellow"))

            if isinstance(value, str):
                # Complexity badge inline
                if key == "complexity":
                    style_map = {
                        "simple": "green",
                        "moderate": "yellow",
                        "complex": "red",
                        "advanced": "magenta bold",
                    }
                    parts.append(
                        Text(f"    {value.upper()}", style=style_map.get(value, ""))
                    )
                else:
                    parts.extend(self._render_value(key, value))
            elif isinstance(value, list):
                # Handle messages-style arrays or feature lists
                if value and isinstance(value[0], str):
                    # Simple string list (e.g., features)
                    parts.append(
                        Panel(
                            Text("  ".join(value)),
                            border_style="dim",
                            padding=(0, 1),
                        )
                    )
                else:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            role = item.get("role", f"item {i}")
                            content = item.get("content", json.dumps(item, indent=2))
                            parts.append(
                                Text(f"    [{role}]", style="bold magenta")
                            )
                            parts.extend(self._render_value(role, content))
                        else:
                            parts.append(
                                Panel(
                                    Text(str(item)),
                                    border_style="dim",
                                    padding=(0, 1),
                                )
                            )
            else:
                parts.append(
                    Panel(
                        Text(json.dumps(value, indent=2)),
                        border_style="dim",
                        padding=(0, 1),
                    )
                )
            parts.append(Text(""))

        return Group(*parts)


# ─── Application ──────────────────────────────────────────────────────────────


class ReviewApp(App):
    """Interactive TUI for reviewing JSONL datasets."""

    CSS = """
    Screen {
        background: $surface;
    }
    #scroll {
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("right", "next_row", "Next →", priority=True),
        Binding("left", "prev_row", "← Prev", priority=True),
        Binding("a", "annotate('a')", "Approve"),
        Binding("r", "annotate('r')", "Reject"),
        Binding("f", "annotate('f')", "Flag"),
        Binding("s", "annotate('s')", "Skip"),
        Binding("e", "annotate('e')", "Edit"),
        Binding("u", "clear_annotation", "Clear"),
        Binding("n", "next_unannotated", "Next ✦"),
        Binding("q", "quit", "Quit"),
    ]

    idx: reactive[int] = reactive(0)

    def __init__(self, rows: list[dict], ann_path: Path):
        super().__init__()
        self.rows = rows
        self.ann_path = ann_path
        self.ann = load_annotations(ann_path)

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="scroll")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()

    def watch_idx(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        """Re-render the current row and update the subtitle."""
        scroll = self.query_one("#scroll")
        scroll.remove_children()
        viewer = RowViewer(
            self.rows[self.idx], self.idx, len(self.rows), self.ann.get(self.idx)
        )
        scroll.mount(viewer)
        scroll.scroll_home(animate=False)
        self.sub_title = (
            f"[{self.idx + 1}/{len(self.rows)}]  "
            + annotation_summary(self.ann, len(self.rows))
        )

    # ── Navigation ──

    def action_next_row(self) -> None:
        if self.idx < len(self.rows) - 1:
            self.idx += 1

    def action_prev_row(self) -> None:
        if self.idx > 0:
            self.idx -= 1

    def action_next_unannotated(self) -> None:
        """Jump to the next row that has no annotation (wraps around)."""
        for i in range(self.idx + 1, len(self.rows)):
            if i not in self.ann:
                self.idx = i
                return
        for i in range(0, self.idx):
            if i not in self.ann:
                self.idx = i
                return
        self.notify("All rows annotated!", severity="information")

    # ── Annotation ──

    def action_annotate(self, key: str) -> None:
        """Toggle annotation: same key removes it, otherwise sets it."""
        if self.ann.get(self.idx) == key:
            del self.ann[self.idx]
            save_annotations(self.ann_path, self.ann)
            self._refresh()
            self.notify(f"Row {self.idx + 1}: annotation cleared")
            return

        self.ann[self.idx] = key
        save_annotations(self.ann_path, self.ann)
        name = LABELS[key][0]
        self.notify(f"Row {self.idx + 1}: {name}")

        # Auto-advance to next row
        if self.idx < len(self.rows) - 1:
            self.idx += 1
        else:
            self._refresh()

    def action_clear_annotation(self) -> None:
        """Remove annotation from current row."""
        if self.idx in self.ann:
            del self.ann[self.idx]
            save_annotations(self.ann_path, self.ann)
            self._refresh()
            self.notify(f"Row {self.idx + 1}: annotation cleared")


# ─── CLI entry point ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Interactive JSONL dataset reviewer")
    parser.add_argument("file", type=Path, help="Path to a JSONL file")
    parser.add_argument(
        "--annotations",
        "-a",
        type=Path,
        default=None,
        help="Path to annotations sidecar JSON (default: <file>.annotations.json)",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: {args.file} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(args.file)
    if not rows:
        print("Error: no rows in file", file=sys.stderr)
        sys.exit(1)

    ann_path = args.annotations or args.file.with_suffix(".annotations.json")

    print(f"Loading {len(rows)} rows from {args.file}")
    print(f"Annotations: {ann_path}")

    app = ReviewApp(rows, ann_path)
    app.run()


if __name__ == "__main__":
    main()
