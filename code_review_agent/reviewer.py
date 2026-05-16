"""AI reviewer: send code to the model and collect structured findings.

Now uses LLMClient (shared singleton) and CodeReviewConfig (config-driven).
"""

from dataclasses import dataclass, field

from ai_agent_playground.llm import LLMClient, get_client

from .config import CodeReviewConfig
from .scanner import FileInfo


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


class Reviewer:
    """Sends code files to the AI and collects structured findings.

    Like model.generate(): the core inference step.
    """

    def __init__(self, config: CodeReviewConfig, llm: LLMClient | None = None):
        self.config = config
        self.llm = llm or get_client()

    def review_files(self, files: list[FileInfo]) -> list[ReviewResult]:
        """Review multiple files. Respects max_files_per_run limit."""
        results = []
        to_review = files[:self.config.max_files_per_run]

        if len(files) > self.config.max_files_per_run:
            print(f"  (limiting to first {self.config.max_files_per_run} "
                  f"of {len(files)} files)\n")

        for i, f in enumerate(to_review, 1):
            print(f"  [{i}/{len(to_review)}] Reviewing {f.rel_path} "
                  f"({f.language}, {f.lines}L)...", end=" ", flush=True)
            result = self._review_one(f)
            n = len(result.issues)
            print(f"{n} issue{'s' if n != 1 else ''} found")
            results.append(result)

        return results

    def _review_one(self, file_info: FileInfo) -> ReviewResult:
        """Send a single file to the AI for review."""
        code = file_info.content
        if len(code) > self.config.max_code_chars:
            code = code[:self.config.max_code_chars] + "\n\n... [truncated]"

        user_msg = (
            f"Review this {file_info.language} file ({file_info.rel_path}, "
            f"{file_info.lines} lines):\n\n"
            f"```{file_info.language.lower()}\n{code}\n```"
        )

        try:
            text = self.llm.send(
                messages=[{"role": "user", "content": user_msg}],
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
            )
        except Exception as exc:
            return ReviewResult(file=file_info, raw_response=f"[API error: {exc}]")

        return ReviewResult(
            file=file_info,
            issues=_parse_issues(text),
            raw_response=text.strip(),
        )


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
