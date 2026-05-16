"""Test case definition and loading.

A test case = "given X input, the agent should produce Y kind of output".
Like unit tests, but for AI agents where the "right answer" isn't always exact.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestCase:
    """One test for an AI agent.

    Like a unit test, but for fuzzy AI outputs:
      - input: what you give the agent
      - expected_keywords: words that SHOULD appear in a good answer
      - forbidden_keywords: words that should NOT appear
      - judge_prompt: what to tell the LLM judge to look for
    """

    id: str                          # e.g. "hello_001"
    agent: str                       # which agent to test: "hello", "code-review", "rag"...
    name: str                        # human-readable: "Should answer what is AI"
    input: str                       # the user's question/input
    expected_keywords: list[str] = field(default_factory=list)  # must contain these
    forbidden_keywords: list[str] = field(default_factory=list)  # must NOT contain these
    judge_prompt: str = ""           # LLM judge: "check if the answer explains X clearly"
    min_length: int = 0              # answer must be at least this many chars
    max_length: int = 99999          # answer must not exceed this many chars


def load_test_cases(case_dir: str = "cases", agent_filter: str | None = None) -> list[TestCase]:
    """Load test cases from JSON files.

    Each JSON file = one agent's test suite. Structure:
      [
        {
          "id": "hello_001",
          "agent": "hello",
          "name": "Should define AI agent",
          "input": "What is an AI agent?",
          "expected_keywords": ["AI", "agent", "autonomous"],
          "judge_prompt": "The answer should define what an AI agent is."
        }
      ]
    """
    case_path = Path(__file__).parent / case_dir
    cases = []

    for fpath in sorted(case_path.glob("*.json")):
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for item in data:
            tc = TestCase(
                id=item["id"],
                agent=item["agent"],
                name=item["name"],
                input=item["input"],
                expected_keywords=item.get("expected_keywords", []),
                forbidden_keywords=item.get("forbidden_keywords", []),
                judge_prompt=item.get("judge_prompt", ""),
                min_length=item.get("min_length", 0),
                max_length=item.get("max_length", 99999),
            )
            if agent_filter is None or tc.agent == agent_filter:
                cases.append(tc)

    return cases
