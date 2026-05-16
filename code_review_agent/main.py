"""Code Review Agent — AI-powered code quality analysis.

Usage:
  uv run python -m code_review_agent.main <path>
  uv run python -m code_review_agent.main                (reviews itself)
"""

import os
import sys
from pathlib import Path

from .agent import CodeReviewAgent


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = str(Path(__file__).parent.parent)

    if not os.path.isdir(target):
        print(f"Not a directory: {target}")
        print("Usage: uv run python -m code_review_agent.main [<path>]")
        sys.exit(1)

    agent = CodeReviewAgent()
    print(f"\nScanning: {target}\n")

    report_path = agent.review(target)
    print(f"\nReport saved → {report_path}")

    content = Path(report_path).read_text(encoding="utf-8")
    print("\n" + "=" * 60)
    print(content[:3000])
    if len(content) > 3000:
        print(f"\n... (report truncated, full file: {report_path})")


if __name__ == "__main__":
    main()
