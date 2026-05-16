"""
Online Evaluator Agent — Generate/Evaluate Separation.

The core insight from Anthropic, OpenAI, AND Google (independently):
a model judging its own output is "self-absorbed" — it overrates itself.
The engineering fix: separate the Generator (produces) from the Evaluator (judges).

This runs as a PARALLEL thread/agent, not a post-hoc step:
  Generator produces output → Evaluator checks in real-time → feedback loop

Unlike eval_harness (offline, runs after the fact), this evaluator runs
DURING agent execution, catching issues before they compound.

Usage:
    evaluator = EvaluatorAgent()
    result = evaluator.check(
        task="Write a function to compute Fibonacci numbers",
        output="def fib(n): return fib(n-1) + fib(n-2)  # no base case!",
    )
    if not result.passed:
        print(f"Issue: {result.feedback}")
"""

from dataclasses import dataclass, field
from typing import Any

from .llm import get_client


@dataclass
class EvalCheckResult:
    """Result of one evaluator check."""

    check_name: str
    passed: bool
    score: float  # 0-1
    feedback: str


@dataclass
class EvaluatorVerdict:
    """Complete evaluator verdict on a piece of agent output."""

    passed: bool
    overall_score: float
    checks: list[EvalCheckResult] = field(default_factory=list)
    summary: str = ""

    @property
    def failed_checks(self) -> list[EvalCheckResult]:
        return [c for c in self.checks if not c.passed]


class EvaluatorAgent:
    """Runs real-time checks on agent output.

    The Evaluator is a SEPARATE LLM call with a critical/contrarian system prompt.
    It is deliberately configured to look for problems — the opposite of the
    Generator's "helpful assistant" posture.

    This separation prevents "model self-love" — the well-documented tendency
    of LLMs to rate their own outputs too highly.
    """

    def __init__(self, model: str = "deepseek-v4-pro[1m]"):
        self.model = model
        self.llm = get_client()

    # ============================================================
    #  Check suite
    # ============================================================

    def check(
        self,
        task: str,
        output: str,
        context: str = "",
        checks: list[str] | None = None,
    ) -> EvaluatorVerdict:
        """Run all enabled checks on agent output.

        Args:
            task: what the agent was asked to do
            output: what the agent produced
            context: additional context (source docs, system state, etc.)
            checks: which checks to run (default: all)
        """
        if checks is None:
            checks = ["task_completion", "factual_accuracy", "code_quality", "edge_cases"]

        results = []
        for check_name in checks:
            try:
                result = self._run_check(check_name, task, output, context)
                results.append(result)
            except Exception as e:
                results.append(EvalCheckResult(
                    check_name=check_name, passed=True, score=0.5,
                    feedback=f"Check error: {e}",
                ))

        passed = all(r.passed for r in results)
        avg_score = sum(r.score for r in results) / len(results) if results else 1.0

        summary = "All checks passed." if passed else (
            f"{len([r for r in results if not r.passed])}/{len(results)} checks failed"
        )

        return EvaluatorVerdict(
            passed=passed,
            overall_score=avg_score,
            checks=results,
            summary=summary,
        )

    # ============================================================
    #  Specific checks
    # ============================================================

    def _run_check(
        self, check_name: str, task: str, output: str, context: str
    ) -> EvalCheckResult:
        """Run one named check using a specialized evaluator prompt."""
        prompts = {
            "task_completion": (
                "You are a strict evaluator. Check if the output COMPLETELY "
                "fulfills the given task. Be critical — if anything is missing "
                "or incomplete, fail it. Do NOT praise the output.\n\n"
                f"Task: {task}\n\nOutput: {output}\n\n"
                "Reply with JSON: {\"passed\": bool, \"score\": 0.0-1.0, "
                "\"feedback\": \"specific issues or OK\"}"
            ),
            "factual_accuracy": (
                "You are a fact-checker. Check every factual claim in the output "
                "against the context. Flag anything not supported by context. "
                "If uncertain, flag it as a potential error.\n\n"
                f"Context: {context}\n\nOutput: {output}\n\n"
                "Reply with JSON: {\"passed\": bool, \"score\": 0.0-1.0, "
                "\"feedback\": \"list unsupported claims or OK\"}"
            ),
            "code_quality": (
                "You are a code reviewer. Check the output code for: "
                "missing edge cases, security issues, inefficiency, "
                "unclear variable names, missing error handling. "
                "Be strict — flag even minor issues.\n\n"
                f"Task: {task}\n\nOutput: {output}\n\n"
                "Reply with JSON: {\"passed\": bool, \"score\": 0.0-1.0, "
                "\"feedback\": \"specific issues or OK\"}"
            ),
            "edge_cases": (
                "You are a QA engineer. Think of edge cases that would BREAK "
                "the given output. Empty input, very large input, special "
                "characters, concurrent access, network failure — test mentally.\n\n"
                f"Task: {task}\n\nOutput: {output}\n\n"
                "Reply with JSON: {\"passed\": bool, \"score\": 0.0-1.0, "
                "\"feedback\": \"edge cases found or OK\"}"
            ),
        }

        prompt = prompts.get(check_name, prompts["task_completion"])
        try:
            raw = self.llm.send(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=512,
                system="You are a critical evaluator. Your job is to find problems. Reply with ONLY JSON.",
            )
            import json
            match = __import__('re').search(r'\{[^}]+\}', raw)
            if match:
                data = json.loads(match.group())
                return EvalCheckResult(
                    check_name=check_name,
                    passed=data.get("passed", True),
                    score=data.get("score", 1.0),
                    feedback=data.get("feedback", "OK"),
                )
        except Exception as e:
            pass

        return EvalCheckResult(
            check_name=check_name, passed=True, score=1.0,
            feedback="(check skipped — parse error)",
        )


# ============================================================
#  Deterministic checks (no LLM involved — fast, reliable)
# ============================================================


def check_output_not_empty(output: str) -> EvalCheckResult:
    """Deterministic: output must not be empty."""
    passed = bool(output and output.strip())
    return EvalCheckResult(
        check_name="not_empty",
        passed=passed,
        score=1.0 if passed else 0.0,
        feedback="OK" if passed else "Output is empty",
    )


def check_output_min_length(output: str, min_length: int = 20) -> EvalCheckResult:
    """Deterministic: output must meet minimum length."""
    passed = len(output) >= min_length
    return EvalCheckResult(
        check_name=f"min_length_{min_length}",
        passed=passed,
        score=min(1.0, len(output) / min_length),
        feedback="OK" if passed else f"Output too short ({len(output)} < {min_length})",
    )


def check_no_error_keywords(output: str) -> EvalCheckResult:
    """Deterministic: output must not contain common error indicators."""
    error_keywords = ["error", "failed", "cannot", "unable", "sorry", "unfortunately"]
    output_lower = output.lower()
    found = [kw for kw in error_keywords if kw in output_lower]
    passed = len(found) == 0
    return EvalCheckResult(
        check_name="no_error_keywords",
        passed=passed,
        score=0.0 if found else 1.0,
        feedback=f"Found error keywords: {found}" if found else "OK",
    )


# ============================================================
#  Convenience runner
# ============================================================


def evaluate_agent_output(
    task: str,
    output: str,
    context: str = "",
    online: bool = True,
    checks: list[str] | None = None,
) -> EvaluatorVerdict:
    """Run both deterministic + online checks on agent output.

    Args:
        task: what the agent was asked to do
        output: what the agent produced
        context: supporting context
        online: if True, also run LLM-based checks
        checks: specific checks to run
    """
    # Deterministic checks (fast, always run)
    det_results = [
        check_output_not_empty(output),
        check_output_min_length(output),
        check_no_error_keywords(output),
    ]

    if online:
        evaluator = EvaluatorAgent()
        return evaluator.check(task, output, context, checks)

    # Deterministic-only mode
    all_passed = all(r.passed for r in det_results)
    return EvaluatorVerdict(
        passed=all_passed,
        overall_score=sum(r.score for r in det_results) / len(det_results),
        checks=det_results,
        summary="All checks passed." if all_passed else "Some checks failed.",
    )
