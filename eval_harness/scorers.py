"""Scoring methods — how to judge if an AI's answer is good.

Three scoring approaches (from simple to smart):

1. Contains check (fast, dumb):
   "Did the answer include these keywords?"
   Like: checking if a cake recipe mentions "flour" → must have flour

2. Semantic similarity (medium):
   "How close is the meaning to the expected answer?"
   Uses embedding vectors to measure "closeness of meaning"

3. LLM Judge (smart, slow):
   "Hey AI, is this a good answer to the question?"
   Uses another AI call to evaluate the output. Most reliable for open-ended
   tasks where there's no single "right answer".
"""

import re

from ai_agent_playground.llm import get_client


def score_contains(output: str, expected_keywords: list[str]) -> float:
    """Check what fraction of expected keywords appear in the output.

    Returns 1.0 if ALL keywords found, 0.0 if NONE found.
    """
    if not expected_keywords:
        return 1.0  # No keywords to check → pass

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
    """Check output length is within bounds.

    Returns 1.0 if within bounds, linearly decreases outside bounds.
    """
    actual = len(output)
    if min_len <= actual <= max_len:
        return 1.0
    if actual < min_len:
        return actual / min_len if min_len > 0 else 1.0
    return max_len / actual if actual > 0 else 1.0


def score_llm_judge(question: str, output: str, judge_prompt: str,
                    model: str = "deepseek-v4-pro[1m]") -> float:
    """Use an AI to judge the quality of another AI's output.

    This is the most reliable scoring method for open-ended tasks.
    The judge is given:
      - The original question
      - The agent's answer
      - A rubric (judge_prompt) describing what a good answer looks like

    Returns a score from 0.0 to 1.0.
    """
    if not judge_prompt:
        return 1.0  # No rubric → skip judge

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
        # Extract first number from response
        match = re.search(r'\d+', raw)
        if match:
            return min(100, max(0, int(match.group()))) / 100.0
        return 0.5  # Can't parse → neutral score
    except Exception:
        return 0.5  # API error → neutral score


def compute_total_score(
    output: str,
    question: str = "",
    expected_keywords: list[str] | None = None,
    forbidden_keywords: list[str] | None = None,
    judge_prompt: str = "",
    min_length: int = 0,
    max_length: int = 99999,
    enabled_scorers: list[str] | None = None,
) -> dict:
    """Run all enabled scorers and compute a weighted total.

    Returns a dict with individual scores and a combined total.
    """
    if enabled_scorers is None:
        enabled_scorers = ["contains", "llm_judge"]

    scores = {}

    # Fast checks (always run)
    if "contains" in enabled_scorers and expected_keywords:
        scores["contains"] = score_contains(output, expected_keywords or [])
        scores["forbidden"] = score_forbidden(output, forbidden_keywords or [])
        scores["length"] = score_length(output, min_length, max_length)

    # LLM judge (expensive, only if enabled)
    if "llm_judge" in enabled_scorers and judge_prompt:
        scores["llm_judge"] = score_llm_judge(question, output, judge_prompt)

    # Combined score: average of all scorers
    if scores:
        scores["total"] = sum(scores.values()) / len(scores)
    else:
        scores["total"] = 1.0

    return scores
