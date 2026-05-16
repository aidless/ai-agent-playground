"""Code Review Agent — AI-powered code quality analysis.

Usage:
  uv run python code_review_agent/main.py <path>
  uv run python code_review_agent/main.py <github-url>   (coming soon)
  uv run python code_review_agent/main.py                (reviews itself as demo)
"""

import os
import sys
import tempfile
from pathlib import Path

try:
    from .scanner import scan_directory
    from .reviewer import review_files
    from .report import generate_report
except ImportError:
    from code_review_agent.scanner import scan_directory
    from code_review_agent.reviewer import review_files
    from code_review_agent.report import generate_report

OUTPUT_DIR = Path(__file__).parent.parent / "reports"


def review_project(root: str, name: str | None = None) -> str:
    """Run full review pipeline: scan → review → report. Returns report path."""
    project_name = name or Path(root).name
    print(f"\nScanning: {root}\n")

    files = scan_directory(root)
    print(f"Found {len(files)} code files to review\n")

    if not files:
        print("No reviewable code files found.")
        sys.exit(0)

    results = review_files(files)
    report_md = generate_report(results, project_name)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"review-{project_name}-{_timestamp()}.md"
    out_path.write_text(report_md, encoding="utf-8")

    print(f"\nReport saved → {out_path}")
    return str(out_path)


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default: review ourselves (the ai-agent-playground project)
        target = str(Path(__file__).parent.parent)

    if os.path.isdir(target):
        out = review_project(target)
        # Print a preview
        content = Path(out).read_text(encoding="utf-8")
        print("\n" + "=" * 60)
        print(content[:3000])
        if len(content) > 3000:
            print(f"\n... (report truncated, full file: {out})")
    else:
        print(f"Not a directory: {target}")
        print("Usage: uv run python code_review_agent/main.py [<path>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
