"""
Test runner — runs agents against test cases and collects results.

Supports both plain-string agents and meta-aware agents.
Meta-aware agents return (output_text, agent_meta_dict) —
this enables process-level scoring: tool success rate, rounds, efficiency.
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
    scores: dict
    duration_seconds: float
    passed: bool
    error: str = ""
    agent_meta: dict = field(default_factory=dict)


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

    @property
    def agent_summary(self) -> dict:
        """Aggregate agent-level metrics across all cases."""
        metas = [r.agent_meta for r in self.results if r.agent_meta]
        if not metas:
            return {}
        total_tool_calls = sum(m.get("tool_calls", 0) for m in metas)
        total_rounds = sum(m.get("rounds", 0) for m in metas)
        total_errors = sum(m.get("tool_errors", 0) for m in metas)
        return {
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls": total_tool_calls / len(metas),
            "avg_rounds": total_rounds / len(metas),
            "tool_error_rate": total_errors / total_tool_calls if total_tool_calls else 0,
        }


# Registry: maps agent name → runner function
_AGENT_RUNNERS = {}


def register(agent_name: str):
    """Decorator: register an agent runner function.

    Runner can return:
      - str: just the output text
      - (str, dict): output text + agent_meta for process-level scoring
    """

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
        abs_path="test",
        rel_path="test.py",
        language="Python",
        content=test_case.input,
        lines=test_case.input.count("\n") + 1,
    )
    result = reviewer._review_one(file_info)
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


@register("mcp-agent")
def _run_mcp_agent(test_case: TestCase) -> tuple[str, dict]:
    """Run MCP agent and capture process-level metrics.

    Unlike other agents, we break into pipeline steps to capture
    tool_log, rounds, and tool errors before postprocess formats them.
    """
    from mcp_agent.agent import MCPToolAgent

    agent = MCPToolAgent()
    model_inputs = agent.preprocess(test_case.input)
    model_outputs = agent._forward(model_inputs)
    output = agent.postprocess(model_outputs)

    tool_log = model_outputs.get("tool_log", [])
    rounds = model_outputs.get("tool_rounds", 0)
    tool_errors = sum(
        1 for t in tool_log if "error" in str(t.get("result", "")).lower()
    )

    meta = {
        "tool_calls": len(tool_log),
        "rounds": rounds,
        "tool_names": [t["tool"] for t in tool_log],
        "tool_errors": tool_errors,
        "tool_success_rate": (
            (len(tool_log) - tool_errors) / len(tool_log) if tool_log else 1.0
        ),
        "max_rounds_reached": rounds >= agent.config.max_tool_rounds,
    }
    return output, meta


# ---- Evaluation engine ----


def run_evaluation(
    agent_filter: str | None = None,
    scorers: list[str] | None = None,
    pass_threshold: float = 0.6,
) -> dict[str, EvalReport]:
    """Run all test cases for one or all agents.

    Returns a dict: agent_name → EvalReport
    """
    cases = load_test_cases(agent_filter=agent_filter)
    if not cases:
        print("No test cases found.")
        return {}

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

        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {agent_name} ({len(agent_cases)} cases)")
        print(f"{'=' * 60}")

        for tc in agent_cases:
            print(f"  [{tc.id}] {tc.name} ...", end=" ", flush=True)

            start = time.time()
            agent_meta = {}
            try:
                raw = runner(tc)
                if isinstance(raw, tuple):
                    output, agent_meta = raw
                else:
                    output = raw
                error = ""
            except Exception as e:
                output = f"[ERROR] {e}"
                error = str(e)

            duration = time.time() - start

            scores = compute_total_score(
                output=output,
                question=tc.input,
                expected_keywords=tc.expected_keywords,
                forbidden_keywords=tc.forbidden_keywords,
                judge_prompt=tc.judge_prompt,
                min_length=tc.min_length,
                max_length=tc.max_length,
                enabled_scorers=scorers,
                agent_meta=agent_meta,
                expected_tools=tc.expected_tools,
            )

            passed = scores["total"] >= pass_threshold
            status = "PASS" if passed else "FAIL"

            # Agent meta summary for console
            meta_str = ""
            if agent_meta:
                tc_count = agent_meta.get("tool_calls", 0)
                tc_success = agent_meta.get("tool_success_rate", 1.0)
                meta_str = f" | tools: {tc_count}, success: {tc_success:.0%}"

            print(f"{status} (score: {scores['total']:.2f}, {duration:.1f}s{meta_str})")

            result = EvalResult(
                test=tc,
                output=output,
                scores=scores,
                duration_seconds=duration,
                passed=passed,
                error=error,
                agent_meta=agent_meta,
            )
            report.results.append(result)

        # Compute summary
        report.passed_cases = sum(1 for r in report.results if r.passed)
        report.avg_score = (
            sum(r.scores["total"] for r in report.results) / len(report.results)
        )
        report.avg_duration = (
            sum(r.duration_seconds for r in report.results) / len(report.results)
        )

        reports[agent_name] = report

        # Print summary with agent-level metrics
        summary = report.agent_summary
        print(f"\n  Summary: {report.passed_cases}/{report.total_cases} passed "
              f"({report.pass_rate:.0%}), avg score: {report.avg_score:.2f}")
        if summary:
            print(f"  Agent metrics: {summary['total_tool_calls']} tool calls, "
                  f"avg {summary['avg_tool_calls']:.1f}/case, "
                  f"avg {summary['avg_rounds']:.1f} rounds/case")

    return reports
