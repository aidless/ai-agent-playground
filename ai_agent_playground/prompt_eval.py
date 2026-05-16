"""
Prompt A/B Evaluator — measure if prompt v2 is actually better than v1.

"Better" is defined quantitatively: run the same questions through both
prompt versions, score the outputs with an LLM judge, compare results.

Usage:
    from ai_agent_playground.prompt_registry import create_default_registry
    from ai_agent_playground.prompt_eval import PromptABTest

    registry = create_default_registry()
    test = PromptABTest(registry, "mcp_agent", "v1", "v2")
    report = test.run(test_questions=["What is 15*15?", "Read /etc/hosts"])
    test.print_report(report)
"""

from dataclasses import dataclass, field
from typing import Any

from .llm import get_client
from .prompt_registry import PromptRegistry


@dataclass
class ABTestResult:
    """Result of comparing two prompt versions on one question."""

    question: str
    score_v1: float
    score_v2: float
    winner: str  # "v1", "v2", "tie"
    delta: float  # positive = v2 better
    answer_v1: str = ""
    answer_v2: str = ""


@dataclass
class ABTestReport:
    """Aggregated A/B test report."""

    prompt_name: str
    version_a: str
    version_b: str
    results: list[ABTestResult] = field(default_factory=list)
    total_questions: int = 0
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    avg_delta: float = 0.0
    avg_score_a: float = 0.0
    avg_score_b: float = 0.0

    @property
    def verdict(self) -> str:
        if self.avg_delta > 0.05:
            return f"{self.version_b} is BETTER (+{self.avg_delta:.2f})"
        elif self.avg_delta < -0.05:
            return f"{self.version_a} is BETTER ({self.avg_delta:.2f})"
        return "No significant difference"


class PromptABTest:
    """Run A/B tests between two prompt versions.

    Uses an LLM judge (same as eval_harness scorers) to evaluate outputs.
    The judge doesn't know which prompt produced which answer — blind test.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        prompt_name: str,
        v1: str,
        v2: str,
        judge_model: str = "deepseek-v4-pro[1m]",
    ):
        self.prompt_a = registry.get(prompt_name, v1)
        self.prompt_b = registry.get(prompt_name, v2)
        if not self.prompt_a or not self.prompt_b:
            raise ValueError(f"Prompt versions not found: {prompt_name} ({v1} / {v2})")
        self.prompt_name = prompt_name
        self.judge_model = judge_model
        self.llm = get_client()

    def run(
        self,
        test_questions: list[str],
        judge_criteria: str = "",
    ) -> ABTestReport:
        """Run A/B test on all questions.

        Args:
            test_questions: list of test inputs
            judge_criteria: what to tell the judge to look for
        """
        if not judge_criteria:
            judge_criteria = (
                "Evaluate: accuracy, clarity, completeness, and whether "
                "the answer directly addresses the question."
            )

        report = ABTestReport(
            prompt_name=self.prompt_name,
            version_a=self.prompt_a.version,
            version_b=self.prompt_b.version,
            total_questions=len(test_questions),
        )

        for q in test_questions:
            result = self._test_one(q, judge_criteria)
            report.results.append(result)

            if result.winner == "v1":
                report.wins_a += 1
            elif result.winner == "v2":
                report.wins_b += 1
            else:
                report.ties += 1

        # Aggregate
        if report.results:
            report.avg_delta = sum(r.delta for r in report.results) / len(report.results)
            report.avg_score_a = sum(r.score_v1 for r in report.results) / len(report.results)
            report.avg_score_b = sum(r.score_v2 for r in report.results) / len(report.results)

        return report

    def _test_one(self, question: str, criteria: str) -> ABTestResult:
        """Test one question against both prompt versions."""
        answer_a = self._generate(self.prompt_a.content, question)
        answer_b = self._generate(self.prompt_b.content, question)

        # Blind judge: score both answers against the same criteria
        score_a = self._judge(question, answer_a, criteria)
        score_b = self._judge(question, answer_b, criteria)

        delta = score_b - score_a
        if delta > 0.05:
            winner = "v2"
        elif delta < -0.05:
            winner = "v1"
        else:
            winner = "tie"

        return ABTestResult(
            question=question,
            score_v1=score_a,
            score_v2=score_b,
            winner=winner,
            delta=delta,
            answer_v1=answer_a,
            answer_v2=answer_b,
        )

    def _generate(self, system_prompt: str, question: str) -> str:
        """Generate an answer using the given system prompt."""
        return self.llm.send(
            messages=[{"role": "user", "content": question}],
            model=self.judge_model,
            max_tokens=1024,
            system=system_prompt,
        )

    def _judge(self, question: str, answer: str, criteria: str) -> float:
        """LLM judge: score answer 0-100."""
        import re

        judge_prompt = (
            "You are an expert evaluator. Score the answer on a scale of 0-100 "
            "based on the criteria below. Reply with ONLY a number.\n\n"
            f"Criteria: {criteria}"
        )
        judge_msg = (
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            f"Score (0-100):"
        )
        try:
            raw = self.llm.send(
                messages=[{"role": "user", "content": judge_msg}],
                model=self.judge_model,
                max_tokens=20,
                system=judge_prompt,
            )
            match = re.search(r'\d+', raw)
            if match:
                return min(100, max(0, int(match.group()))) / 100.0
            return 0.5
        except Exception:
            return 0.5

    def print_report(self, report: ABTestReport):
        """Print a human-readable A/B test report."""
        print(f"\n{'=' * 70}")
        print(f"  PROMPT A/B TEST: {report.prompt_name}")
        print(f"  {report.version_a} vs {report.version_b}")
        print(f"{'=' * 70}")
        print()
        print(f"  Questions tested: {report.total_questions}")
        print(f"  Avg score ({report.version_a}): {report.avg_score_a:.2f}")
        print(f"  Avg score ({report.version_b}): {report.avg_score_b:.2f}")
        print(f"  Wins: {report.version_a}={report.wins_a}, "
              f"{report.version_b}={report.wins_b}, ties={report.ties}")
        print(f"  Avg delta: {report.avg_delta:+.2f}")
        print(f"\n  VERDICT: {report.verdict}")
        print()

        # Per-question breakdown
        for i, r in enumerate(report.results):
            delta_str = f"+{r.delta:.2f}" if r.delta > 0 else f"{r.delta:.2f}"
            print(f"  Q{i + 1}: [{r.winner}] {r.question[:60]}...")
            print(f"       {report.version_a}={r.score_v1:.2f}  "
                  f"{report.version_b}={r.score_v2:.2f}  "
                  f"delta={delta_str}")
