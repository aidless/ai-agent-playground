"""
Report generator — turns eval results into readable output.

Shows both answer quality AND agent process metrics (tool usage, efficiency).
"""

from .runner import EvalReport


def print_report(reports: dict[str, EvalReport]):
    """Print a human-readable evaluation report to the console."""
    if not reports:
        print("No reports to show.")
        return

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)

    total_cases = 0
    total_passed = 0

    for agent_name, report in reports.items():
        total_cases += report.total_cases
        total_passed += report.passed_cases

        summary = report.agent_summary

        print(f"\n{'─' * 70}")
        print(f"  Agent: {agent_name}")
        print(f"  Passed: {report.passed_cases}/{report.total_cases} "
              f"({report.pass_rate:.0%})")
        print(f"  Avg Score: {report.avg_score:.2f}  |  "
              f"Avg Time: {report.avg_duration:.1f}s")
        if summary:
            print(f"  Tool Calls: {summary['total_tool_calls']} total, "
                  f"{summary['avg_tool_calls']:.1f}/case avg  |  "
                  f"Rounds: {summary['avg_rounds']:.1f}/case avg  |  "
                  f"Tool Error Rate: {summary['tool_error_rate']:.0%}")
        print(f"{'─' * 70}")

        for r in report.results:
            icon = "PASS" if r.passed else "FAIL"
            meta = r.agent_meta

            # Build score detail string
            score_parts = []
            for k, v in r.scores.items():
                if k != "total":
                    score_parts.append(f"{k}={v:.2f}")
            score_detail = ", ".join(score_parts)

            print(f"\n  [{icon}] {r.test.id}: {r.test.name}")
            print(f"  Score: {r.scores.get('total', 0):.2f} ({score_detail})")

            # Agent process detail
            if meta:
                tools_str = ", ".join(meta.get("tool_names", [])) or "none"
                rounds = meta.get("rounds", 0)
                success_rate = meta.get("tool_success_rate", 1.0)
                maxed = " (MAXED)" if meta.get("max_rounds_reached") else ""
                print(f"  Tools used: [{tools_str}] | Rounds: {rounds}{maxed} | "
                      f"Tool success: {success_rate:.0%}")

            if not r.passed and r.error:
                print(f"  Error: {r.error}")
            elif not r.passed:
                if r.scores.get("contains", 1.0) < 0.5:
                    missing = [
                        kw
                        for kw in r.test.expected_keywords
                        if kw.lower() not in r.output.lower()
                    ]
                    print(f"  Missing keywords: {missing}")

            snippet = r.output[:200].replace("\n", " ")
            print(f"  Output: {snippet}...")

    # Grand total
    print(f"\n{'=' * 70}")
    if total_cases > 0:
        print(f"  GRAND TOTAL: {total_passed}/{total_cases} passed "
              f"({total_passed / total_cases:.0%})")
    else:
        print("  No cases run")
    print(f"{'=' * 70}\n")


def save_markdown(reports: dict[str, EvalReport], path: str = "eval_report.md"):
    """Save evaluation report as a Markdown file with agent process metrics."""
    from pathlib import Path

    lines = [
        "# AI Agent Evaluation Report",
        "",
        f"**Total Agents Evaluated**: {len(reports)}",
        "",
    ]

    for agent_name, report in reports.items():
        summary = report.agent_summary

        lines.extend([
            f"## {agent_name}",
            "",
            f"- **Passed**: {report.passed_cases}/{report.total_cases} "
            f"({report.pass_rate:.0%})",
            f"- **Avg Score**: {report.avg_score:.2f}",
            f"- **Avg Duration**: {report.avg_duration:.1f}s",
        ])

        if summary:
            lines.extend([
                f"- **Total Tool Calls**: {summary['total_tool_calls']}",
                f"- **Avg Tool Calls/Case**: {summary['avg_tool_calls']:.1f}",
                f"- **Avg Rounds/Case**: {summary['avg_rounds']:.1f}",
                f"- **Tool Error Rate**: {summary['tool_error_rate']:.0%}",
            ])

        lines.extend([
            "",
            "| Case | Result | Score | Tools | Rounds | Output |",
            "|------|--------|-------|-------|--------|--------|",
        ])

        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            meta = r.agent_meta
            tools_str = ", ".join(meta.get("tool_names", [])) if meta else "-"
            rounds_str = str(meta.get("rounds", "-")) if meta else "-"
            snippet = r.output[:100].replace("\n", " ").replace("|", "/")
            lines.append(
                f"| {r.test.id} | {status} | {r.scores.get('total', 0):.2f} | "
                f"{tools_str} | {rounds_str} | {snippet}... |"
            )

        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
