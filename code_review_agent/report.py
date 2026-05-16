"""Report generator: turn review results into a structured Markdown report."""

from datetime import datetime

try:
    from .reviewer import Issue, ReviewResult
except ImportError:
    from code_review_agent.reviewer import Issue, ReviewResult

SEVERITY_EMOJI = {"critical": "Critical", "warning": "Warning", "info": "Info"}
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def generate_report(results: list[ReviewResult], project_name: str) -> str:
    """Build a full Markdown code review report."""
    all_issues: list[tuple[ReviewResult, Issue]] = []
    for r in results:
        for issue in r.issues:
            all_issues.append((r, issue))

    # Sort: severity desc, then category, then file
    all_issues.sort(key=lambda ri: (
        SEVERITY_ORDER.get(ri[1].severity, 99),
        ri[1].category,
        ri[0].file.rel_path,
    ))

    critical = sum(1 for _, i in all_issues if i.severity == "critical")
    warnings = sum(1 for _, i in all_issues if i.severity == "warning")
    infos = sum(1 for _, i in all_issues if i.severity == "info")
    files_with_issues = len({r.file.rel_path for r, _ in all_issues})

    lines = []
    lines.append(f"# Code Review Report: {project_name}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Files reviewed**: {len(results)}")
    lines.append(f"**Files with issues**: {files_with_issues}")
    lines.append(f"**Total issues**: {len(all_issues)} "
                 f"({critical} critical, {warnings} warning, {infos} info)")
    lines.append("")

    if not all_issues:
        lines.append("## No issues found")
        lines.append("")
        lines.append("All reviewed files look good. No actionable issues detected.")
        return "\n".join(lines)

    # ---- Summary by category ----
    lines.append("---")
    lines.append("")
    lines.append("## Summary by Category")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    cat_counts: dict[str, int] = {}
    for _, issue in all_issues:
        cat_counts[issue.category] = cat_counts.get(issue.category, 0) + 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    # ---- Critical issues ----
    _append_severity_section(lines, all_issues, "critical", results)

    # ---- Warnings ----
    _append_severity_section(lines, all_issues, "warning", results)

    # ---- Info ----
    _append_severity_section(lines, all_issues, "info", results)

    # ---- Per-file summary ----
    lines.append("---")
    lines.append("")
    lines.append("## Per-File Summary")
    lines.append("")
    lines.append("| File | Issues |")
    lines.append("|------|--------|")
    file_issue_counts: dict[str, int] = {}
    for r, _ in all_issues:
        file_issue_counts[r.file.rel_path] = file_issue_counts.get(r.file.rel_path, 0) + 1
    for r in results:
        count = file_issue_counts.get(r.file.rel_path, 0)
        if count > 0:
            lines.append(f"| {r.file.rel_path} | {count} |")
    lines.append("")

    return "\n".join(lines)


def _append_severity_section(
    lines: list[str],
    all_issues: list[tuple[any, Issue]],
    severity: str,
    results: list[ReviewResult],
) -> None:
    group = [(r, i) for r, i in all_issues if i.severity == severity]
    if not group:
        return

    labels = {"critical": "Critical Issues", "warning": "Warnings", "info": "Info & Suggestions"}
    lines.append("---")
    lines.append("")
    lines.append(f"## {labels[severity]}")
    lines.append("")

    for r, issue in group:
        lines.append(f"### `{r.file.rel_path}` — {issue.title}")
        lines.append("")
        lines.append(f"- **Line**: {issue.line}")
        lines.append(f"- **Category**: {issue.category}")
        lines.append(f"- **Severity**: {severity}")
        lines.append("")
        lines.append(issue.description)
        lines.append("")
