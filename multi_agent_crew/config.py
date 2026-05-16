"""Multi-Agent Crew config — declare params for each role agent."""

from dataclasses import dataclass
from typing import ClassVar

from ai_agent_playground.config import BaseAgentConfig


@dataclass
class CrewConfig(BaseAgentConfig):
    """Configuration for the entire multi-agent crew."""

    agent_type: ClassVar[str] = "multi-agent-crew"

    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 2048

    # Each role has its own system prompt
    pm_prompt: str = (
        "You are a senior Product Manager. Given a user's requirement, "
        "break it down into 3-5 concrete technical tasks. Each task should be "
        "specific enough that a developer can implement it without asking questions.\n\n"
        "Output format (one task per line):\n"
        "  TASK_ID|PRIORITY|TITLE|DESCRIPTION\n"
        "PRIORITY: high | medium | low\n\n"
        "Example:\n"
        "  T-1|high|User auth API|Create POST /login and POST /register endpoints "
        "with JWT token return"
    )

    dev_prompt: str = (
        "You are a senior Software Developer. Given a technical task, "
        "write clean, working Python code. Include:\n"
        "- File path and name\n"
        "- Complete, runnable code\n"
        "- Brief comments for non-obvious logic\n\n"
        "Output format:\n"
        "```<filename>\n<code>\n```"
    )

    qa_prompt: str = (
        "You are a QA Engineer. Review the provided code for:\n"
        "1. Bugs or logic errors\n"
        "2. Missing edge case handling\n"
        "3. Security issues\n"
        "4. Testability concerns\n\n"
        "Output each issue as:\n"
        "  SEVERITY|FILE|LINE|ISSUE|SUGGESTION\n"
        "SEVERITY: critical | warning | info\n\n"
        "If no issues found, say \"All clear — no issues found.\""
    )

    devops_prompt: str = (
        "You are a DevOps Engineer. Given the completed project code, generate:\n"
        "1. A Dockerfile (multi-stage if appropriate)\n"
        "2. A docker-compose.yml (include any required services)\n"
        "3. A short deployment checklist (3-5 bullet points)\n\n"
        "Output each section clearly marked with ## Section headers."
    )

    # Workflow control
    max_tasks: int = 4  # Max tasks PM can create
    enable_qa: bool = True
    enable_devops: bool = True
