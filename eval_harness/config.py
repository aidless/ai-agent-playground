"""Eval configuration — what to test, how to score."""

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """How to run evaluations."""

    # Which test case files to load
    case_files: list[str] = field(default_factory=lambda: ["hello_agent", "code_review"])

    # Scoring methods to use
    scorers: list[str] = field(default_factory=lambda: [
        "contains",    # Check if keywords exist in output
        "llm_judge",   # Use AI to evaluate AI (most reliable for open-ended tasks)
    ])

    # LLM judge settings
    judge_model: str = "deepseek-v4-pro[1m]"
    judge_max_tokens: int = 512

    # Passing threshold
    pass_threshold: float = 0.6  # 60% score = pass
