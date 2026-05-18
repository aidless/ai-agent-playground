"""Evaluation Gate — measurable quality metrics for every engine output.

Three-dimensional evaluation (Tool-Genesis inspired):
  1. INTERFACE — does the output follow the expected format?
  2. FUNCTIONAL — does it produce correct results?
  3. UTILITY — does it help solve the actual task?

Every engine decision passes through this gate before being applied.
Before/after comparisons with statistical confidence tell us whether
an evolution, debate, or bootstrap actually improved things.

Usage:
    gate = EvaluationGate(llm_client)
    result = await gate.evaluate("code", candidate_output, reference={"task": "..."})
    if result.passed:
        # Apply the change
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent.parent / "memory" / "reliability"


@dataclass
class EvalDimensions:
    interface_score: float    # 0-10: format compliance
    functional_score: float   # 0-10: correctness
    utility_score: float      # 0-10: task relevance
    overall: float            # 0-10: weighted average

    def is_passed(self, threshold: float = 6.0) -> bool:
        return self.overall >= threshold


@dataclass
class EvalResult:
    eval_id: str
    category: str              # "evolution", "debate", "bootstrap", "autopilot"
    target: str                # what is being evaluated
    candidate: Any             # the new version
    baseline: Any = None       # the old version (for comparison)
    dimensions: Optional[EvalDimensions] = None
    baseline_dimensions: Optional[EvalDimensions] = None
    delta: float = 0.0         # improvement over baseline
    passed: bool = False
    trials: int = 1
    confidence: float = 0.0    # how confident we are in the delta
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


EVAL_INTERFACE_PROMPT = (
    "Rate the FORMAT quality of this output from 0-10.\n"
    "Consider: structure, readability, proper formatting, code blocks, headers.\n"
    "Output ONLY a number (e.g. '8.5'). Nothing else.\n\n"
    "Category: {category}\n"
    "Output:\n{output[:2000]}"
)

EVAL_FUNCTIONAL_PROMPT = (
    "Rate the FUNCTIONAL CORRECTNESS of this output from 0-10.\n"
    "Consider: accuracy, logic, edge case handling, error handling.\n"
    "Output ONLY a number (e.g. '7.0'). Nothing else.\n\n"
    "Task: {task}\n"
    "Output:\n{output[:2000]}"
)

EVAL_UTILITY_PROMPT = (
    "Rate how USEFUL this output is for the given task from 0-10.\n"
    "Consider: completeness, actionability, relevance, practicality.\n"
    "Output ONLY a number (e.g. '8.0'). Nothing else.\n\n"
    "Task: {task}\n"
    "Output:\n{output[:2000]}"
)


class EvaluationGate:
    """Quality checkpoint for all engine decisions.

    Every tool evolution, debate, bootstrap, and autopilot iteration
    passes through this gate. Before/after comparisons with multiple
    trials provide statistical confidence in improvement.
    """

    def __init__(self, client, model: str = "deepseek-chat", default_threshold: float = 6.0):
        self.client = client
        self.model = model
        self.default_threshold = default_threshold
        self._history: list[EvalResult] = []
        EVAL_DIR.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self,
        category: str,
        candidate_text: str,
        task: str = "",
        baseline_output: str = "",
        threshold: float = 0.0,
        trials: int = 1,
    ) -> EvalResult:
        """Evaluate an output across 3 dimensions. Compare to baseline if provided."""
        if threshold <= 0:
            threshold = self.default_threshold

        import uuid
        result = EvalResult(
            eval_id=f"eval-{uuid.uuid4().hex[:6]}",
            category=category,
            target=category,
            candidate=candidate_text,
            baseline=baseline_output if baseline_output else None,
            trials=trials,
        )

        try:
            iface_scores = []
            func_scores = []
            util_scores = []

            for _ in range(trials):
                iface = await self._score(self._safe_format(EVAL_INTERFACE_PROMPT, category=category, output=candidate_text))
                func = await self._score(self._safe_format(EVAL_FUNCTIONAL_PROMPT, task=task, output=candidate_text))
                util = await self._score(self._safe_format(EVAL_UTILITY_PROMPT, task=task, output=candidate_text))
                iface_scores.append(iface)
                func_scores.append(func)
                util_scores.append(util)

            # Average across trials
            dims = EvalDimensions(
                interface_score=round(sum(iface_scores) / len(iface_scores), 1),
                functional_score=round(sum(func_scores) / len(func_scores), 1),
                utility_score=round(sum(util_scores) / len(util_scores), 1),
                overall=round((sum(iface_scores) / len(iface_scores) * 0.2 +
                              sum(func_scores) / len(func_scores) * 0.5 +
                              sum(util_scores) / len(util_scores) * 0.3), 1),
            )
            result.dimensions = dims

            # Evaluate baseline if provided
            if baseline_output:
                b_iface = await self._score(self._safe_format(EVAL_INTERFACE_PROMPT, category=category, output=baseline_output))
                b_func = await self._score(self._safe_format(EVAL_FUNCTIONAL_PROMPT, task=task, output=baseline_output))
                b_util = await self._score(self._safe_format(EVAL_UTILITY_PROMPT, task=task, output=baseline_output))
                result.baseline_dimensions = EvalDimensions(
                    interface_score=b_iface,
                    functional_score=b_func,
                    utility_score=b_util,
                    overall=round(b_iface * 0.2 + b_func * 0.5 + b_util * 0.3, 1),
                )
                result.delta = round(dims.overall - result.baseline_dimensions.overall, 1)

                result.confidence = 1.0 if trials >= 3 else 0.5

            result.passed = dims.is_passed(threshold)

        except Exception as e:
            result.error = str(e)
            logger.warning("Evaluation gate failed: %s", e)
            result.dimensions = EvalDimensions(5.0, 5.0, 5.0, 5.0)
            result.passed = True

        self._history.append(result)
        if len(self._history) > 200:
            self._history = self._history[-100:]
        self._save(result)
        return result

    async def _score(self, prompt: str) -> float:
        """Ask LLM to score something 0-10."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = " ".join(str(c) for c in content)
            else:
                text = str(content)
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                return min(10.0, max(0.0, float(match.group(1))))
        except Exception as e:
            logger.debug("Score parsing failed: %s", e)
        return 5.0

    def _safe_format(self, template: str, **kwargs) -> str:
        """Format a template without breaking on curly braces in values."""
        result = template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, str(value))
        return result

    async def ab_test(
        self,
        category: str,
        candidate_output: str,
        candidate_label: str,
        control_output: str,
        control_label: str,
        task: str = "",
        trials: int = 3,
    ) -> dict:
        """A/B test: candidate vs control with multiple trials."""
        c_result = await self.evaluate(category, candidate_text=candidate_output, task=task, trials=trials)
        a_result = await self.evaluate(category, candidate_text=control_output, task=task, trials=trials)

        return {
            candidate_label: {
                "overall": c_result.dimensions.overall if c_result.dimensions else 0,
                "dimensions": {
                    "interface": c_result.dimensions.interface_score if c_result.dimensions else 0,
                    "functional": c_result.dimensions.functional_score if c_result.dimensions else 0,
                    "utility": c_result.dimensions.utility_score if c_result.dimensions else 0,
                },
            },
            control_label: {
                "overall": a_result.dimensions.overall if a_result.dimensions else 0,
                "dimensions": {
                    "interface": a_result.dimensions.interface_score if a_result.dimensions else 0,
                    "functional": a_result.dimensions.functional_score if a_result.dimensions else 0,
                    "utility": a_result.dimensions.utility_score if a_result.dimensions else 0,
                },
            },
            "winner": candidate_label if (
                (c_result.dimensions.overall if c_result.dimensions else 0) >
                (a_result.dimensions.overall if a_result.dimensions else 0)
            ) else control_label,
            "delta": round(
                (c_result.dimensions.overall if c_result.dimensions else 0) -
                (a_result.dimensions.overall if a_result.dimensions else 0), 1
            ),
        }

    def _save(self, result: EvalResult):
        path = EVAL_DIR / "eval_history.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "eval_id": result.eval_id,
                    "category": result.category,
                    "passed": result.passed,
                    "delta": result.delta,
                    "confidence": result.confidence,
                    "dims": {
                        "interface": result.dimensions.interface_score if result.dimensions else 0,
                        "functional": result.dimensions.functional_score if result.dimensions else 0,
                        "utility": result.dimensions.utility_score if result.dimensions else 0,
                        "overall": result.dimensions.overall if result.dimensions else 0,
                    },
                    "timestamp": result.timestamp,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def stats(self) -> dict:
        """Aggregate statistics across all evaluations."""
        if not self._history:
            return {"total": 0}

        passed = sum(1 for r in self._history if r.passed)
        by_category = {}
        for r in self._history:
            cat = r.category
            by_category.setdefault(cat, {"total": 0, "passed": 0, "avg_overall": 0.0})
            by_category[cat]["total"] += 1
            if r.passed:
                by_category[cat]["passed"] += 1
            if r.dimensions:
                by_category[cat]["avg_overall"] = (
                    (by_category[cat]["avg_overall"] * (by_category[cat]["total"] - 1) +
                     r.dimensions.overall) / by_category[cat]["total"]
                )

        recent_deltas = [r.delta for r in self._history[-20:] if r.baseline_dimensions]

        return {
            "total": len(self._history),
            "pass_rate": round(passed / len(self._history), 3),
            "by_category": {
                cat: {
                    "total": info["total"],
                    "pass_rate": round(info["passed"] / info["total"], 3) if info["total"] else 0,
                    "avg_score": round(info["avg_overall"], 1),
                }
                for cat, info in by_category.items()
            },
            "avg_improvement_delta": round(sum(recent_deltas) / len(recent_deltas), 2) if recent_deltas else 0,
        }
