"""Report generator: turn review results into structured Markdown."""

from datetime import datetime

from .reviewer import Issue, ReviewResult

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


class ReportGenerator:
    """Formats review results into a clean Markdown report.

    Like Pipeline.postprocess(): raw model output → user-facing format.
    """

    def generate(self, results: list[ReviewResult], project_name: str) -> str:
        """Build a full Markdown code review report."""
        all_issues: list[tuple[ReviewResult, Issue]] = []
        for r in results:
            for issue in r.issues:
                all_issues.append((r, issue))

        # Sort: severity desc → category → file
        all_issues.sort(key=lambda ri: (
            SEVERITY_ORDER.get(ri[1].severity, 99),
            ri[1].category,
            ri[0].file.rel_path,
        ))

        critical = sum(1 for _, i in all_issues if i.severity == "critical")
        warnings = sum(1 for _, i in all_issues if i.severity == "warning")
        infos = sum(1 for _, i in all_issues if i.severity == "info")
        files_with_issues = len({r.file.rel_path for r, _ in all_issues})

        lines = [
            f"# Code Review Report: {project_name}",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Files reviewed**: {len(results)}",
            f"**Files with issues**: {files_with_issues}",
            f"**Total issues**: {len(all_issues)} "
            f"({critical} critical, {warnings} warning, {infos} info)",
            "",
        ]

        if not all_issues:
            lines.extend([
                "## No issues found", "",
                "All reviewed files look good. No actionable issues detected.",
            ])
            return "\n".join(lines)

        # Summary by category
        lines.extend(["---", "", "## Summary by Category", "", "| Category | Count |", "|----------|-------|"])
        cat_counts: dict[str, int] = {}
        for _, issue in all_issues:
            cat_counts[issue.category] = cat_counts.get(issue.category, 0) + 1
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        # Severity sections
        for sev, label in [("critical", "Critical Issues"), ("warning", "Warnings"), ("info", "Info & Suggestions")]:
            group = [(r, i) for r, i in all_issues if i.severity == sev]
            if not group:
                continue
            lines.extend(["---", "", f"## {label}", ""])
            for r, issue in group:
                lines.extend([
                    f"### `{r.file.rel_path}` — {issue.title}", "",
                    f"- **Line**: {issue.line}",
                    f"- **Category**: {issue.category}",
                    f"- **Severity**: {sev}", "",
                    issue.description, "",
                ])

        # Per-file summary
        lines.extend(["---", "", "## Per-File Summary", "", "| File | Issues |", "|------|--------|"])
        file_counts: dict[str, int] = {}
        for r, _ in all_issues:
            file_counts[r.file.rel_path] = file_counts.get(r.file.rel_path, 0) + 1
        for r in results:
            count = file_counts.get(r.file.rel_path, 0)
            if count > 0:
                lines.append(f"| {r.file.rel_path} | {count} |")
        lines.append("")

        return "\n".join(lines)
