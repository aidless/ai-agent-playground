#!/usr/bin/env python3
"""
Eval Gate — automated quality gate for AI agent changes.

Usage:
  # As a pre-commit hook (add to .git/hooks/pre-commit):
  uv run python scripts/eval_gate.py

  # Manual check:
  uv run python scripts/eval_gate.py --agent mcp-agent --threshold 0.7

  # CI mode (exit 1 on failure, output JSON):
  uv run python scripts/eval_gate.py --ci

What it does:
  1. Runs eval_harness on specified agents
  2. Compares scores against baseline (stored in reports/baseline.json)
  3. If any agent drops below threshold → FAIL, exit 1
  4. If scores improved → update baseline, pass
  5. Generates a gate report

This is the "指标掉了代码直接拒绝 merge" mechanism.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
BASELINE_PATH = REPO_ROOT / "reports" / "baseline.json"
GATE_REPORT_PATH = REPO_ROOT / "reports" / "gate_report.md"


def load_baseline() -> dict:
    """Load baseline scores from disk."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {}


def save_baseline(data: dict):
    """Save baseline scores to disk."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_evals(agent_filter: str | None = None) -> dict:
    """Run eval harness and return agent_name → avg_score mapping."""
    from eval_harness.runner import run_evaluation

    reports = run_evaluation(
        agent_filter=agent_filter,
        scorers=["contains", "llm_judge"],
        pass_threshold=0.0,  # Get all results, we'll threshold ourselves
    )

    scores = {}
    for agent_name, report in reports.items():
        scores[agent_name] = {
            "avg_score": round(report.avg_score, 3),
            "pass_rate": round(report.pass_rate, 3),
            "total_cases": report.total_cases,
            "passed_cases": report.passed_cases,
            "avg_duration": round(report.avg_duration, 2),
        }
        summary = report.agent_summary
        if summary:
            scores[agent_name]["tool_error_rate"] = round(summary["tool_error_rate"], 3)
            scores[agent_name]["avg_tool_calls"] = round(summary["avg_tool_calls"], 1)

    return scores


def gate_check(
    current: dict,
    baseline: dict,
    threshold: float = 0.6,
    regression_threshold: float = 0.05,
) -> tuple[bool, list[dict]]:
    """Compare current scores against baseline.

    A regression is defined as: current score dropped > regression_threshold
    below baseline.

    Returns (passed, list of issues).
    """
    issues = []

    for agent_name, curr in current.items():
        curr_score = curr["avg_score"]

        # Check absolute threshold
        if curr_score < threshold:
            issues.append({
                "agent": agent_name,
                "type": "below_threshold",
                "current": curr_score,
                "threshold": threshold,
                "message": f"{agent_name}: {curr_score:.2f} < threshold {threshold}",
            })
            continue

        # Check regression against baseline
        if agent_name in baseline:
            base_score = baseline[agent_name]["avg_score"]
            delta = curr_score - base_score
            if delta < -regression_threshold:
                issues.append({
                    "agent": agent_name,
                    "type": "regression",
                    "baseline": base_score,
                    "current": curr_score,
                    "delta": round(delta, 3),
                    "message": f"{agent_name}: {curr_score:.2f} dropped from {base_score:.2f} ({delta:+.3f})",
                })

    return len(issues) == 0, issues


def generate_report(current: dict, baseline: dict, passed: bool, issues: list[dict]):
    """Generate a Markdown gate report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "PASS" if passed else "FAIL"

    lines = [
        f"# Eval Gate Report — {status}",
        "",
        f"**Timestamp**: {timestamp}",
        f"**Gate Status**: {'✅ PASS' if passed else '❌ FAIL'}",
        "",
        "## Agent Scores",
        "",
        "| Agent | Score | Pass Rate | Cases | vs Baseline |",
        "|-------|-------|-----------|-------|-------------|",
    ]

    for agent_name, curr in current.items():
        base = baseline.get(agent_name, {})
        base_score = base.get("avg_score")
        if base_score is not None:
            delta = curr["avg_score"] - base_score
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "new"
        line = (
            f"| {agent_name} | {curr['avg_score']:.3f} | "
            f"{curr['pass_rate']:.0%} | {curr['total_cases']} | {delta_str} |"
        )
        lines.append(line)

    if issues:
        lines.extend([
            "",
            "## Issues",
            "",
        ])
        for issue in issues:
            lines.append(f"- ❌ {issue['message']}")

    if passed:
        lines.extend([
            "",
            "## Action",
            "",
            "No regressions detected. Safe to merge.",
        ])
    else:
        lines.extend([
            "",
            "## Action Required",
            "",
            "Regressions detected. Fix the issues before merging:",
            "",
            "1. Review agent output for the failing test cases",
            "2. Check if the code change introduced a bug",
            "3. If the baseline is stale (legitimate improvement), update it:",
            "   `uv run python scripts/eval_gate.py --update-baseline`",
            "4. Re-run the gate: `uv run python scripts/eval_gate.py`",
        ])

    GATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(GATE_REPORT_PATH)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Agent Eval Gate")
    parser.add_argument("--agent", help="Filter to one agent")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum acceptable score (default: 0.6)")
    parser.add_argument("--regression", type=float, default=0.05,
                        help="Max acceptable score drop (default: 0.05)")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: exit 1 on failure, output JSON")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Update baseline to current scores")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Agent Eval Gate")
    print(f"  Threshold: {args.threshold}  |  Max regression: {args.regression}")
    print("=" * 60)
    print()

    # Run evaluations
    try:
        current = run_evals(agent_filter=args.agent)
    except Exception as e:
        print(f"Eval harness failed: {e}")
        sys.exit(2)

    if not current:
        print("No agents evaluated. Check eval_harness setup.")
        sys.exit(2)

    # Print current scores
    for agent_name, scores in current.items():
        print(f"  {agent_name}: {scores['avg_score']:.3f} "
              f"({scores['passed_cases']}/{scores['total_cases']} passed)")

    # Update baseline if requested
    if args.update_baseline:
        save_baseline(current)
        print(f"\nBaseline updated: {BASELINE_PATH}")
        sys.exit(0)

    # Gate check
    baseline = load_baseline()
    if not baseline:
        print("\nNo baseline found. Creating initial baseline...")
        save_baseline(current)
        print(f"Baseline saved to {BASELINE_PATH}")
        print("Run eval gate again to compare against baseline.")
        sys.exit(0)

    passed, issues = gate_check(
        current, baseline,
        threshold=args.threshold,
        regression_threshold=args.regression,
    )

    # Report
    report_path = generate_report(current, baseline, passed, issues)

    if args.ci:
        result = {
            "passed": passed,
            "current": current,
            "baseline": baseline,
            "issues": issues,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if not passed:
        print(f"\n❌ GATE FAILED — {len(issues)} issue(s)")
        for issue in issues:
            print(f"  - {issue['message']}")
        print(f"\nReport: {report_path}")
        sys.exit(1)

    print(f"\n✅ GATE PASSED — No regressions detected")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
