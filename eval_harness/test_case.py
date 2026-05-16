"""
Test case definition and loading.

A test case = "given X input, the agent should produce Y kind of output".
For Agent-specific cases, also defines expected tool behavior.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestCase:
    """One test for an AI agent.

    Standard fields (all agents):
      - input: what you give the agent
      - expected_keywords: words that SHOULD appear in a good answer
      - forbidden_keywords: words that should NOT appear
      - judge_prompt: what to tell the LLM judge to look for

    Agent-specific fields (process-level):
      - expected_tools: tools the agent SHOULD use (e.g. ["calculator", "read_file"])
      - max_rounds: max acceptable ReAct rounds before scoring penalty
    """

    id: str
    agent: str
    name: str
    input: str
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    judge_prompt: str = ""
    min_length: int = 0
    max_length: int = 99999
    expected_tools: list[str] = field(default_factory=list)
    max_rounds: int = 5


def load_test_cases(
    case_dir: str = "cases", agent_filter: str | None = None
) -> list[TestCase]:
    """Load test cases from JSON files.

    Each JSON file = one agent's test suite. Example structure:

      [
        {
          "id": "mcp_001",
          "agent": "mcp-agent",
          "name": "Simple calculation",
          "input": "What is 15 * 15 + 12?",
          "expected_keywords": ["237"],
          "expected_tools": ["calculator"],
          "judge_prompt": "Answer must contain correct result."
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
                expected_tools=item.get("expected_tools", []),
                max_rounds=item.get("max_rounds", 5),
            )
            if agent_filter is None or tc.agent == agent_filter:
                cases.append(tc)

    return cases
