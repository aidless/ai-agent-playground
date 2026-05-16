"""Report generator — turns eval results into readable output.

Like a test report you'd see in CI: what passed, what failed, and why.
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

        print(f"\n{'─'*70}")
        print(f"  Agent: {agent_name}")
        print(f"  Passed: {report.passed_cases}/{report.total_cases} "
              f"({report.pass_rate:.0%})")
        print(f"  Avg Score: {report.avg_score:.2f}  |  "
              f"Avg Time: {report.avg_duration:.1f}s")
        print(f"{'─'*70}")

        for r in report.results:
            icon = "PASS" if r.passed else "FAIL"
            print(f"\n  [{icon}] {r.test.id}: {r.test.name}")
            print(f"  Score: {r.scores.get('total', 0):.2f} "
                  f"({', '.join(f'{k}={v:.2f}' for k, v in r.scores.items() if k != 'total')})")

            if not r.passed and r.error:
                print(f"  Error: {r.error}")
            elif not r.passed:
                # Show why it failed
                if r.scores.get("contains", 1.0) < 0.5:
                    missing = [kw for kw in r.test.expected_keywords
                               if kw.lower() not in r.output.lower()]
                    print(f"  Missing keywords: {missing}")

            # Show a snippet of the output
            snippet = r.output[:200].replace("\n", " ")
            print(f"  Output: {snippet}...")

    # Grand total
    print(f"\n{'='*70}")
    print(f"  GRAND TOTAL: {total_passed}/{total_cases} passed "
          f"({total_passed/total_cases:.0%})"
          if total_cases > 0 else "  No cases run")
    print(f"{'='*70}\n")


def save_markdown(reports: dict[str, EvalReport], path: str = "eval_report.md"):
    """Save evaluation report as a Markdown file."""
    lines = [
        "# AI Agent Evaluation Report",
        "",
        f"**Total Agents Evaluated**: {len(reports)}",
        "",
    ]

    for agent_name, report in reports.items():
        lines.extend([
            f"## {agent_name}",
            "",
            f"- **Passed**: {report.passed_cases}/{report.total_cases} "
            f"({report.pass_rate:.0%})",
            f"- **Avg Score**: {report.avg_score:.2f}",
            f"- **Avg Duration**: {report.avg_duration:.1f}s",
            "",
            "| Case | Result | Score | Output |",
            "|------|--------|-------|--------|",
        ])

        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            snippet = r.output[:100].replace("\n", " ").replace("|", "/")
            lines.append(
                f"| {r.test.id} | {status} | {r.scores.get('total', 0):.2f} | {snippet}... |"
            )

        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
