"""AI reviewer: send code to the model and collect structured findings."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

try:
    from .scanner import FileInfo
except ImportError:
    from code_review_agent.scanner import FileInfo

load_dotenv(Path(__file__).parent.parent / ".env")

_client = Anthropic(
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

REVIEW_SYSTEM_PROMPT = """\
You are a senior code reviewer. Analyze the code below and report issues.

For each issue you find, use EXACTLY this format (one line per issue):
  SEVERITY|LINE|CATEGORY|TITLE|DESCRIPTION

SEVERITY must be one of: critical, warning, info
CATEGORY must be one of: bug, security, performance, style, best-practice
LINE is the approximate line number (integer, or 0 if file-wide)

Only report real issues. Skip files that look fine — just say "No issues found."
Do NOT suggest generic improvements. Only flag concrete, actionable problems.

Example output:
  warning|23|style|Variable name too short|Variable 'x' should be renamed to something descriptive
  critical|45|bug|Off-by-one error|range(len(items)) should be range(len(items)-1)
"""

MODEL = "deepseek-v4-pro[1m]"
MAX_FILES_PER_RUN = 30  # Safety: don't burn through API budget


@dataclass
class Issue:
    severity: str       # critical / warning / info
    line: int           # approximate line number
    category: str       # bug / security / performance / style / best-practice
    title: str          # one-line summary
    description: str    # full explanation


@dataclass
class ReviewResult:
    file: FileInfo
    issues: list[Issue] = field(default_factory=list)
    raw_response: str = ""


def _parse_issues(text: str) -> list[Issue]:
    """Parse AI response lines into Issue objects."""
    issues = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        try:
            issues.append(Issue(
                severity=parts[0].strip().lower(),
                line=int(parts[1].strip()),
                category=parts[2].strip().lower(),
                title=parts[3].strip(),
                description=parts[4].strip(),
            ))
        except (ValueError, IndexError):
            continue
    return issues


def review_file(file_info: FileInfo) -> ReviewResult:
    """Send a single file to the AI for review."""
    # Truncate content if too long (max ~8000 chars to keep response focused)
    code = file_info.content
    if len(code) > 8000:
        code = code[:8000] + "\n\n... [file truncated, showing first 8000 chars]"

    user_msg = (
        f"Review this {file_info.language} file ({file_info.rel_path}, "
        f"{file_info.lines} lines):\n\n```{file_info.language.lower()}\n{code}\n```"
    )

    try:
        response = _client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=REVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        return ReviewResult(
            file=file_info,
            raw_response=f"[API error: {exc}]",
        )

    text = ""
    for block in response.content:
        if isinstance(block, TextBlock):
            text += block.text

    issues = _parse_issues(text)

    return ReviewResult(
        file=file_info,
        issues=issues,
        raw_response=text.strip(),
    )


def review_files(files: list[FileInfo]) -> list[ReviewResult]:
    """Review multiple files. Respects MAX_FILES_PER_RUN limit."""
    results = []
    to_review = files[:MAX_FILES_PER_RUN]

    if len(files) > MAX_FILES_PER_RUN:
        print(f"  (limiting to first {MAX_FILES_PER_RUN} of {len(files)} files)\n")

    for i, f in enumerate(to_review, 1):
        lang_tag = f.language
        print(f"  [{i}/{len(to_review)}] Reviewing {f.rel_path} ({lang_tag}, {f.lines}L)...", end=" ", flush=True)
        result = review_file(f)
        n = len(result.issues)
        print(f"{n} issue{'s' if n != 1 else ''} found")
        results.append(result)

    return results
