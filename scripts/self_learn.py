"""Self-learning: analyze Claude Code history → extract patterns → update memory.

Run periodically to make Claude smarter over time:
    python scripts/self_learn.py
"""

import json
import re
from collections import Counter
from pathlib import Path

HISTORY_FILE = Path.home() / ".claude" / "history.jsonl"
MEMORY_DIR = Path.home() / ".claude" / "projects" / "C--Users-Administrator-Desktop" / "memory"


def load_history(path: Path, limit: int = 500) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries[-limit:]


def analyze(entries: list[dict]) -> dict:
    """Extract patterns from history entries."""
    patterns = {
        "projects": Counter(),
        "commands": Counter(),
        "short_responses": Counter(),
        "total_messages": len(entries),
    }

    for e in entries:
        display = e.get("display", "")
        project = e.get("project", "")

        patterns["projects"][project] += 1

        # Detect short responses (user making choices)
        if display.strip() in ("A", "B", "C", "继续", "继续吧", "好继续把", "可以", "是的"):
            patterns["short_responses"][display.strip()] += 1

        # Detect commands
        if display.startswith("/"):
            cmd = display.split()[0]
            patterns["commands"][cmd] += 1
        if display.startswith("cd "):
            patterns["commands"]["cd"] += 1

    return patterns


def generate_report(patterns: dict) -> str:
    """Generate a human-readable report from patterns."""
    lines = ["# Claude Code Self-Learning Report", ""]

    lines.append(f"Analyzed {patterns['total_messages']} history entries.")
    lines.append("")

    lines.append("## Active Projects")
    for proj, count in patterns["projects"].most_common(5):
        lines.append(f"- {proj}: {count} messages")
    lines.append("")

    lines.append("## Frequent Commands")
    for cmd, count in patterns["commands"].most_common(10):
        lines.append(f"- `{cmd}`: {count} times")
    lines.append("")

    lines.append("## Quick Responses (user making choices)")
    for resp, count in patterns["short_responses"].most_common(5):
        lines.append(f"- `{resp}`: {count} times")
    lines.append("")

    return "\n".join(lines)


def main():
    if not HISTORY_FILE.exists():
        print(f"History file not found: {HISTORY_FILE}")
        return

    entries = load_history(HISTORY_FILE)
    patterns = analyze(entries)
    report = generate_report(patterns)

    print(report)

    # Save report to memory
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MEMORY_DIR / "auto-report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
