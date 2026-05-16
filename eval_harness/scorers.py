"""
Scoring methods — how to judge if an AI's answer is good.

Three layers of scoring (from fast/dumb to slow/smart):

1. **Keyword checks** (fast): Does the output contain expected keywords?
2. **LLM Judge** (smart): Another AI evaluates the answer quality.
3. **Agent process metrics** (structural): Tool success rate, round efficiency,
   tool selection correctness — only available for meta-aware agents.
"""

import re

from ai_agent_playground.llm import get_client


def score_contains(output: str, expected_keywords: list[str]) -> float:
    """Check what fraction of expected keywords appear in the output.

    Returns 1.0 if ALL keywords found, 0.0 if NONE found.
    """
    if not expected_keywords:
        return 1.0

    output_lower = output.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in output_lower)
    return found / len(expected_keywords)


def score_forbidden(output: str, forbidden_keywords: list[str]) -> float:
    """Check that forbidden keywords do NOT appear.

    Returns 1.0 if NO forbidden words found, 0.0 if ANY found.
    """
    if not forbidden_keywords:
        return 1.0

    output_lower = output.lower()
    violations = sum(1 for kw in forbidden_keywords if kw.lower() in output_lower)
    return 0.0 if violations > 0 else 1.0


def score_length(output: str, min_len: int, max_len: int) -> float:
    """Check output length is within bounds."""
    actual = len(output)
    if min_len <= actual <= max_len:
        return 1.0
    if actual < min_len:
        return actual / min_len if min_len > 0 else 1.0
    return max_len / actual if actual > 0 else 1.0


def score_llm_judge(
    question: str,
    output: str,
    judge_prompt: str,
    model: str = "deepseek-v4-pro[1m]",
) -> float:
    """Use an AI to judge the quality of another AI's output.

    This is the most reliable scoring method for open-ended tasks.
    Returns a score from 0.0 to 1.0.
    """
    if not judge_prompt:
        return 1.0

    client = get_client()

    judge_system = (
        "You are an expert evaluator of AI responses. "
        "Score the answer on a scale of 0 to 100 based on the rubric. "
        "Reply with ONLY a number (0-100), nothing else."
    )

    judge_message = (
        f"Question: {question}\n\n"
        f"Answer to evaluate: {output}\n\n"
        f"Rubric (what a good answer looks like): {judge_prompt}\n\n"
        f"Score (0-100):"
    )

    try:
        raw = client.send(
            messages=[{"role": "user", "content": judge_message}],
            model=model,
            max_tokens=50,
            system=judge_system,
        )
        match = re.search(r"\d+", raw)
        if match:
            return min(100, max(0, int(match.group()))) / 100.0
        return 0.5
    except Exception:
        return 0.5


# ============================================================
#  Agent process-level scorers
#  These evaluate HOW the agent worked, not just WHAT it said.
# ============================================================


def score_tool_success_rate(agent_meta: dict) -> float:
    """Score based on tool call success rate.

    Each failed tool call (error in result) reduces the score.
    Returns 1.0 if all tools succeeded or no tools were used.
    """
    if not agent_meta:
        return 1.0
    return agent_meta.get("tool_success_rate", 1.0)


def score_round_efficiency(agent_meta: dict, max_rounds: int = 5) -> float:
    """Score based on how efficiently the agent used its rounds.

    - Finishing in 1 round = 1.0 (very efficient)
    - Reaching max_rounds = 0.2 (barely finished, maybe didn't finish)
    - No tool calls = 1.0 (simple answer, no efficiency concern)
    """
    if not agent_meta or not agent_meta.get("tool_calls"):
        return 1.0

    rounds = agent_meta.get("rounds", 0)
    if rounds >= max_rounds and agent_meta.get("max_rounds_reached", False):
        return 0.2  # Hit the limit — agent couldn't finish in time

    # Linear decay: 1 round = 1.0, max_rounds = 0.3
    if max_rounds <= 1:
        return 1.0
    return max(0.3, 1.0 - (rounds - 1) / (max_rounds - 1) * 0.7)


def score_tool_selection(agent_meta: dict, expected_tools: list[str]) -> float:
    """Score based on whether the agent selected the right tools.

    Args:
        agent_meta: Must contain 'tool_names' list
        expected_tools: Tools the agent SHOULD have used

    Returns 1.0 if all expected tools were used (or no expectation set).
    """
    if not expected_tools or not agent_meta:
        return 1.0

    used = set(agent_meta.get("tool_names", []))
    expected = set(expected_tools)
    if not expected:
        return 1.0
    return len(used & expected) / len(expected)


# ============================================================
#  Composite scorer
# ============================================================


def compute_total_score(
    output: str,
    question: str = "",
    expected_keywords: list[str] | None = None,
    forbidden_keywords: list[str] | None = None,
    judge_prompt: str = "",
    min_length: int = 0,
    max_length: int = 99999,
    enabled_scorers: list[str] | None = None,
    agent_meta: dict | None = None,
    expected_tools: list[str] | None = None,
) -> dict:
    """Run all enabled scorers and compute a weighted total.

    If agent_meta is provided, also runs process-level scorers
    (tool_success, round_efficiency, tool_selection).
    """
    if enabled_scorers is None:
        enabled_scorers = ["contains", "llm_judge"]

    scores = {}
    meta = agent_meta or {}

    if "contains" in enabled_scorers:
        if expected_keywords is not None or forbidden_keywords is not None:
            scores["contains"] = score_contains(output, expected_keywords or [])
            scores["forbidden"] = score_forbidden(output, forbidden_keywords or [])
            scores["length"] = score_length(output, min_length, max_length)

    if "llm_judge" in enabled_scorers and judge_prompt:
        scores["llm_judge"] = score_llm_judge(question, output, judge_prompt)

    # Agent process scorers — only active when meta is present
    if meta:
        scores["tool_success"] = score_tool_success_rate(meta)
        scores["round_efficiency"] = score_round_efficiency(meta)

        if expected_tools:
            scores["tool_selection"] = score_tool_selection(meta, expected_tools)

    if scores:
        scores["total"] = sum(scores.values()) / len(scores)
    else:
        scores["total"] = 1.0

    return scores
