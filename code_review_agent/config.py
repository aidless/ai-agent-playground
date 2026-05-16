"""CodeReviewAgent config — like BertConfig, declare params with defaults."""

from dataclasses import dataclass, field
from typing import ClassVar

from ai_agent_playground.config import BaseAgentConfig


@dataclass
class CodeReviewConfig(BaseAgentConfig):
    """Configuration for CodeReviewAgent.

    Like BertConfig declaring hidden_size, num_hidden_layers, etc.
    — we declare scanner settings, reviewer settings, report settings.
    """

    agent_type: ClassVar[str] = "code-review"

    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 2048
    system_prompt: str = (
        "You are a senior code reviewer. Analyze the code below and report issues.\n\n"
        "For each issue you find, use EXACTLY this format (one line per issue):\n"
        "  SEVERITY|LINE|CATEGORY|TITLE|DESCRIPTION\n\n"
        "SEVERITY must be one of: critical, warning, info\n"
        "CATEGORY must be one of: bug, security, performance, style, best-practice\n"
        "LINE is the approximate line number (integer, or 0 if file-wide)\n\n"
        "Only report real issues. Skip files that look fine — just say \"No issues found.\"\n"
        "Do NOT suggest generic improvements. Only flag concrete, actionable problems.\n\n"
        "Example output:\n"
        "  warning|23|style|Variable name too short|Variable 'x' should be renamed\n"
        "  critical|45|bug|Off-by-one error|range(len(items)) should be range(len(items)-1)"
    )

    # Scanner settings
    code_extensions: dict[str, str] = field(default_factory=lambda: {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".jsx": "React JSX", ".tsx": "React TSX", ".java": "Java",
        ".go": "Go", ".rs": "Rust", ".c": "C", ".cpp": "C++",
        ".css": "CSS", ".html": "HTML", ".sql": "SQL",
        ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML", ".json": "JSON",
    })
    skip_dirs: set[str] = field(default_factory=lambda: {
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        ".idea", ".vscode", "dist", "build", "target", ".next",
    })
    max_file_bytes: int = 200_000
    max_code_chars: int = 8_000

    # Reviewer settings
    max_files_per_run: int = 30
