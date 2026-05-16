"""Test runner — runs agents against test cases and collects results.

Like pytest for AI agents: given a set of test cases, run each one,
score the output, and report pass/fail.
"""

import time
from dataclasses import dataclass, field

from ai_agent_playground.llm import get_client

from .scorers import compute_total_score
from .test_case import TestCase, load_test_cases


@dataclass
class EvalResult:
    """Result of running one test case."""
    test: TestCase
    output: str
    scores: dict           # {"contains": 1.0, "llm_judge": 0.85, "total": 0.925}
    duration_seconds: float
    passed: bool
    error: str = ""


@dataclass
class EvalReport:
    """Complete evaluation report for one agent."""
    agent_name: str
    results: list[EvalResult] = field(default_factory=list)
    total_cases: int = 0
    passed_cases: int = 0
    avg_score: float = 0.0
    avg_duration: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0


# Registry: maps agent name → runner function
# Each runner takes (test_case) → output_string
_AGENT_RUNNERS = {}


def register(agent_name: str):
    """Decorator: register an agent runner function."""
    def decorator(fn):
        _AGENT_RUNNERS[agent_name] = fn
        return fn
    return decorator


# ---- Agent runners ----

@register("hello")
def _run_hello(test_case: TestCase) -> str:
    from hello_agent.agent import HelloAgent
    return HelloAgent().ask(test_case.input)


@register("code-review")
def _run_code_review(test_case: TestCase) -> str:
    from code_review_agent.scanner import FileInfo
    from code_review_agent.reviewer import Reviewer
    from code_review_agent.config import CodeReviewConfig
    config = CodeReviewConfig()
    reviewer = Reviewer(config)
    file_info = FileInfo(
        abs_path="test", rel_path="test.py",
        language="Python", content=test_case.input,
        lines=test_case.input.count("\n") + 1,
    )
    result = reviewer._review_one(file_info)
    # Format issues as readable text
    if not result.issues:
        return "No issues found."
    return "\n".join(
        f"[{i.severity}] {i.category}: {i.title} — {i.description}"
        for i in result.issues
    )


@register("rag")
def _run_rag(test_case: TestCase) -> str:
    from rag_qa_system.agent import RAGAgent
    return RAGAgent().ask(test_case.input)


def run_evaluation(agent_filter: str | None = None,
                   scorers: list[str] | None = None,
                   pass_threshold: float = 0.6) -> dict[str, EvalReport]:
    """Run all test cases for one or all agents.

    Returns a dict: agent_name → EvalReport
    """
    cases = load_test_cases(agent_filter=agent_filter)
    if not cases:
        print("No test cases found.")
        return {}

    # Group cases by agent
    by_agent: dict[str, list[TestCase]] = {}
    for tc in cases:
        by_agent.setdefault(tc.agent, []).append(tc)

    reports = {}

    for agent_name, agent_cases in by_agent.items():
        if agent_name not in _AGENT_RUNNERS:
            print(f"  [SKIP] No runner for agent '{agent_name}'")
            continue

        runner = _AGENT_RUNNERS[agent_name]
        report = EvalReport(agent_name=agent_name, total_cases=len(agent_cases))

        print(f"\n{'='*60}")
        print(f"  Evaluating: {agent_name} ({len(agent_cases)} cases)")
        print(f"{'='*60}")

        for tc in agent_cases:
            print(f"  [{tc.id}] {tc.name} ...", end=" ", flush=True)

            # Run the agent
            start = time.time()
            try:
                output = runner(tc)
                error = ""
            except Exception as e:
                output = f"[ERROR] {e}"
                error = str(e)

            duration = time.time() - start

            # Score the output
            scores = compute_total_score(
                output=output,
                question=tc.input,
                expected_keywords=tc.expected_keywords,
                forbidden_keywords=tc.forbidden_keywords,
                judge_prompt=tc.judge_prompt,
                min_length=tc.min_length,
                max_length=tc.max_length,
                enabled_scorers=scorers,
            )

            passed = scores["total"] >= pass_threshold
            status = "PASS" if passed else "FAIL"
            print(f"{status} (score: {scores['total']:.2f}, {duration:.1f}s)")

            result = EvalResult(
                test=tc, output=output, scores=scores,
                duration_seconds=duration, passed=passed, error=error,
            )
            report.results.append(result)

        # Compute summary
        report.passed_cases = sum(1 for r in report.results if r.passed)
        report.avg_score = sum(r.scores["total"] for r in report.results) / len(report.results)
        report.avg_duration = sum(r.duration_seconds for r in report.results) / len(report.results)

        reports[agent_name] = report

        # Print summary
        print(f"\n  Summary: {report.passed_cases}/{report.total_cases} passed "
              f"({report.pass_rate:.0%}), avg score: {report.avg_score:.2f}")

    return reports
