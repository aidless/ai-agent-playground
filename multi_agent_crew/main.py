"""Multi-Agent Crew CLI.

Usage:
  uv run python -m multi_agent_crew.main "Build a todo app API"
  uv run python -m multi_agent_crew.main                    (demo mode)
"""

import sys
from pathlib import Path

from .agent import CrewAgent


def main():
    if len(sys.argv) > 1:
        requirement = " ".join(sys.argv[1:])
    else:
        requirement = (
            "Build a REST API for a personal blog with CRUD operations "
            "for posts and comments. Use FastAPI and SQLite."
        )

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Crew")
    print(f"  PM → Dev → QA → DevOps")
    print(f"{'='*60}\n")

    agent = CrewAgent()
    report = agent.build(requirement)

    # Save report
    out_dir = Path(__file__).parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"crew-report-{ts}.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"\nReport saved → {out_path}")
    print(f"\nPreview:\n{report[:2000]}")
    if len(report) > 2000:
        print(f"\n... (full report: {out_path})")


if __name__ == "__main__":
    main()
