"""Self-Play Task Generator — Tool-R0 style autonomous curriculum learning.

The Generator creates tasks at the Solver's competence frontier.
The Solver attempts to solve them.
The Evaluator scores the result.
Feedback flows back to the Generator for better task selection.
Results consolidate into persistent memory for cross-session transfer.

This creates a natural curriculum: tasks get progressively harder
as the agent improves, staying just beyond current ability.

Key insight from Tool-R0 (arXiv:2602.21320):
  "The Generator is rewarded for generating challenging tasks aligned
   with the Solver's evolving capabilities, while the Solver is trained
   to solve them with outcome-based rewards."

v2 adds:
  - Generator Learning: LLM analyzes results and improves task quality
  - Memory Consolidation: self-play lessons persist as agent memories
  - Strategy Adaptation: difficulty adjusts based on score patterns
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SELFPLAY_DIR = Path(__file__).resolve().parent.parent / "memory" / "autopilot"


# ── Domain definitions ──────────────────────────

DOMAINS = {
    "python_basics": {
        "name": "Python Basics",
        "difficulty_range": (1, 4),
        "topics": ["variables", "loops", "conditionals", "functions", "lists", "dicts"],
        "example_task": "Write a function that takes a list of numbers and returns their sum.",
    },
    "python_intermediate": {
        "name": "Python Intermediate",
        "difficulty_range": (3, 6),
        "topics": ["decorators", "generators", "context managers", "OOP", "exceptions"],
        "example_task": "Write a class-based context manager for timing code blocks.",
    },
    "python_advanced": {
        "name": "Python Advanced",
        "difficulty_range": (5, 8),
        "topics": ["metaclasses", "async/await", "descriptors", "coroutines", "C extensions"],
        "example_task": "Implement a simple async task scheduler with priority queues.",
    },
    "algorithms": {
        "name": "Algorithms",
        "difficulty_range": (2, 7),
        "topics": ["sorting", "searching", "graphs", "DP", "trees", "greedy"],
        "example_task": "Implement Dijkstra's shortest path algorithm.",
    },
    "system_design": {
        "name": "System Design",
        "difficulty_range": (4, 9),
        "topics": ["caching", "load balancing", "databases", "microservices", "API design"],
        "example_task": "Design a rate limiter for a REST API with sliding window.",
    },
    "security": {
        "name": "Security",
        "difficulty_range": (3, 8),
        "topics": ["XSS", "SQL injection", "auth", "encryption", "sandboxing"],
        "example_task": "Explain how to prevent SQL injection in a Python web app.",
    },
    "code_review": {
        "name": "Code Review",
        "difficulty_range": (2, 6),
        "topics": ["bugs", "performance", "style", "architecture", "testing"],
        "example_task": "Review this code for security vulnerabilities and performance issues.",
    },
}


@dataclass
class PracticeTask:
    task_id: str
    domain: str
    difficulty: int            # 1-10
    instruction: str
    expected_keywords: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PracticeResult:
    task: PracticeTask
    agent_output: str
    score: float               # 0-10
    latency_ms: float
    success: bool
    analysis: str = ""


GEN_TASK_PROMPT = (
    "You are a curriculum designer for an AI agent. Generate a SINGLE practice "
    "task for the agent to solve. The task should be at difficulty {difficulty}/10 "
    "in the domain of {domain}.\n\n"
    "Domain topics: {topics}\n"
    "Agent's current skill level: {skill_level}/10\n"
    "Previous task scores: {recent_scores}\n\n"
    "Rules:\n"
    "1. The task must be at the agent's COMPETENCE FRONTIER — challenging but solvable\n"
    "2. Include specific instructions, not vague prompts\n"
    "3. For coding tasks, specify the function signature\n"
    "4. Output format: just the task instruction, nothing else"
)

GEN_LEARN_PROMPT = (
    "You are improving a task generator. Review the past training results "
    "and generate ONE INSIGHT about how to create better tasks.\n\n"
    "Recent results (domain, difficulty, score, task_preview):\n{results_summary}\n\n"
    "Analyze:\n"
    "1. Which domains/topics need easier tasks (prerequisites)?\n"
    "2. Which domains are ready for harder challenges?\n"
    "3. What task formats produce the best scores?\n"
    "Output: one actionable sentence starting with 'NEXT: '"
)

MEMORY_CONSOLIDATION_PROMPT = (
    "Review these self-play training results. Extract 2-3 concrete LESSONS "
    "that would help the agent perform better on real tasks.\n\n"
    "Results: {results_summary}\n\n"
    "Output format (one lesson per line):\n"
    "- Lesson 1\n"
    "- Lesson 2"
)

EVAL_PROMPT = (
    "You are scoring an AI agent's response to a practice task.\n\n"
    "Task: {task}\n"
    "Agent's response: {response}\n\n"
    "Score from 0-10 considering: correctness, completeness, clarity, efficiency.\n"
    "Also note what the agent did well and what could improve.\n"
    "Output format: SCORE: X.X\\nANALYSIS: (one sentence)"
)


class CompetenceTracker:
    """Tracks agent skill levels across domains for curriculum learning."""

    def __init__(self):
        self._domain_scores: dict[str, list[float]] = {d: [] for d in DOMAINS}
        self._load()

    def _load(self):
        path = SELFPLAY_DIR / "competence.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for domain, scores in data.items():
                    if domain in self._domain_scores:
                        self._domain_scores[domain] = scores
            except Exception:
                pass

    def _save(self):
        SELFPLAY_DIR.mkdir(parents=True, exist_ok=True)
        (SELFPLAY_DIR / "competence.json").write_text(
            json.dumps(self._domain_scores, indent=2), encoding="utf-8")

    def record(self, domain: str, score: float):
        if domain in self._domain_scores:
            self._domain_scores[domain].append(score)
            if len(self._domain_scores[domain]) > 50:
                self._domain_scores[domain] = self._domain_scores[domain][-50:]
            self._save()

    def get_skill_level(self, domain: str) -> float:
        scores = self._domain_scores.get(domain, [])
        if not scores:
            return 3.0  # default beginner
        # Weighted: recent scores matter more
        if len(scores) <= 3:
            return sum(scores) / len(scores)
        recent_weight = 0.7
        older_weight = 0.3
        recent_avg = sum(scores[-3:]) / 3
        older_avg = sum(scores[:-3]) / max(1, len(scores) - 3)
        return recent_weight * recent_avg + older_weight * older_avg

    def get_frontier_domain(self) -> tuple[str, float, int]:
        """Find the domain where the agent is ready to improve — the competence frontier."""
        candidates = []
        for domain, info in DOMAINS.items():
            skill = self.get_skill_level(domain)
            min_d, max_d = info["difficulty_range"]
            # Frontier: difficulty just above current skill
            if skill < max_d:
                target_diff = min(int(skill + 1.5), max_d)
                candidates.append((domain, skill, target_diff))

        if not candidates:
            # All domains mastered, pick random advanced
            domain = random.choice(list(DOMAINS.keys()))
            return domain, 8.0, 9

        # Prefer domains with fewer practice attempts (exploration)
        candidates.sort(key=lambda c: len(self._domain_scores.get(c[0], [])))
        return candidates[0]

    def status(self) -> dict:
        return {
            domain: {
                "skill": round(self.get_skill_level(domain), 1),
                "attempts": len(self._domain_scores.get(domain, [])),
                "recent_avg": round(sum(self._domain_scores.get(domain, [])[-3:]) / max(1, len(self._domain_scores.get(domain, [])[-3:])), 1),
            }
            for domain in DOMAINS
        }


class SelfPlayEngine:
    """Autonomous curriculum learning through self-play.

    Usage:
        engine = SelfPlayEngine(agent, client, model="deepseek-chat")
        results = await engine.train(rounds=5)
    """

    def __init__(self, agent, client, model: str = "deepseek-chat"):
        self.agent = agent
        self.client = client
        self.model = model
        self.competence = CompetenceTracker()
        self._history: list[PracticeResult] = []
        self._strategy_insights: list[str] = []

    async def train(self, rounds: int = 5, consolidate: bool = True) -> list[PracticeResult]:
        """Run N rounds of self-play training with Generator learning + memory consolidation."""
        results = []
        for i in range(rounds):
            logger.info("Self-play round %d/%d", i + 1, rounds)
            result = await self._single_round()
            results.append(result)
            self.competence.record(result.task.domain, result.score)
            self._history.append(result)

        # Generator Learning: analyze results, adjust strategy
        if len(results) >= 2:
            insight = await self._generator_learn(results)
            if insight:
                self._strategy_insights.append(insight)
                logger.info("Generator insight: %s", insight)

        # Memory Consolidation: persist lessons to agent memory
        if consolidate and len(results) >= 2:
            lessons = await self._consolidate_memory(results)
            if lessons:
                for lesson in lessons:
                    self.agent.memory.add_lesson(
                        lesson=lesson,
                        context=f"self-play training ({len(results)} rounds)",
                        success=True,
                    )
                logger.info("Consolidated %d lessons to agent memory", len(lessons))

        return results

    async def _generator_learn(self, results: list[PracticeResult]) -> str:
        """Analyze results and generate an insight for better task generation."""
        summary_lines = []
        for r in results:
            summary_lines.append(
                f"- {r.task.domain} (diff={r.task.difficulty}): score={r.score}/10 — {r.task.instruction[:80]}..."
            )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GEN_LEARN_PROMPT.format(
                        results_summary="\n".join(summary_lines),
                    )},
                ],
                max_tokens=150,
                temperature=0.5,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("NEXT:"):
                return text[5:].strip()
            return text
        except Exception as e:
            logger.warning("Generator learning failed: %s", e)
            return ""

    async def _consolidate_memory(self, results: list[PracticeResult]) -> list[str]:
        """Extract lessons from self-play and persist to agent memory."""
        summary_lines = []
        for r in results:
            analysis_preview = r.analysis[:100] if r.analysis else "no analysis"
            summary_lines.append(
                f"- {r.task.domain} (diff={r.task.difficulty}): score={r.score}/10, analysis: {analysis_preview}"
            )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MEMORY_CONSOLIDATION_PROMPT.format(
                        results_summary="\n".join(summary_lines),
                    )},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            text = response.choices[0].message.content.strip()
            lessons = [l.strip("- ").strip() for l in text.split("\n") if l.strip().startswith("-")]
            return lessons[:3]
        except Exception as e:
            logger.warning("Memory consolidation failed: %s", e)
            return []

    async def _single_round(self) -> PracticeResult:
        """One round: generate task → solve → evaluate."""
        # Step 1: Generate task at competence frontier
        task = await self._generate_task()

        # Step 2: Solve
        t0 = time.time()
        output = await self._solve(task.instruction)
        latency = (time.time() - t0) * 1000

        # Step 3: Evaluate
        score, analysis = await self._evaluate(task.instruction, output)

        return PracticeResult(
            task=task,
            agent_output=output,
            score=score,
            latency_ms=latency,
            success=score >= 6.0,
            analysis=analysis,
        )

    async def _generate_task(self) -> PracticeTask:
        """Generate a task at the agent's competence frontier."""
        domain, skill, difficulty = self.competence.get_frontier_domain()
        info = DOMAINS[domain]
        recent_scores = self.competence._domain_scores.get(domain, [])[-5:]
        scores_str = ", ".join(f"{s:.1f}" for s in recent_scores) if recent_scores else "none"

        prompt = GEN_TASK_PROMPT.format(
            difficulty=difficulty,
            domain=info["name"],
            topics=", ".join(info["topics"]),
            skill_level=f"{skill:.1f}",
            recent_scores=scores_str,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Generate one practice task."},
                ],
                max_tokens=500,
                temperature=0.8,
            )
            instruction = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Task generation failed: %s", e)
            instruction = info["example_task"]

        return PracticeTask(
            task_id=f"sp-{int(time.time())}",
            domain=domain,
            difficulty=difficulty,
            instruction=instruction,
            expected_keywords=info["topics"][:3],
        )

    async def _solve(self, task: str) -> str:
        """Ask the agent to solve the task."""
        from agent.state import AgentContext
        ctx = AgentContext(trace_id=f"selfplay_{int(time.time())}", max_steps=3)
        ctx = await self.agent.run(ctx, task)
        for msg in ctx.messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return ""

    async def _evaluate(self, task: str, output: str) -> tuple[float, str]:
        """Score the agent's output and provide analysis."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EVAL_PROMPT.format(task=task, response=output[:3000])},
                ],
                max_tokens=200,
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            # Parse SCORE: X.X
            import re
            score_match = re.search(r"SCORE:\s*(\d+\.?\d*)", text, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 5.0
            analysis_match = re.search(r"ANALYSIS:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
            analysis = analysis_match.group(1).strip() if analysis_match else ""
            return min(10.0, max(0.0, score)), analysis
        except Exception as e:
            return 5.0, f"Evaluation failed: {e}"

    def status(self) -> dict:
        return {
            "competence": self.competence.status(),
            "total_rounds": len(self._history),
            "strategy_insights": self._strategy_insights[-5:],
            "recent_scores": [
                {"domain": r.task.domain, "difficulty": r.task.difficulty, "score": r.score}
                for r in self._history[-10:]
            ],
            "improvement": self._calc_improvement(),
        }

    def _calc_improvement(self) -> dict:
        """Calculate skill improvement trends."""
        trends = {}
        for domain in DOMAINS:
            scores = self.competence._domain_scores.get(domain, [])
            if len(scores) >= 6:
                early = sum(scores[:3]) / 3
                late = sum(scores[-3:]) / 3
                trends[domain] = {
                    "early_avg": round(early, 1),
                    "recent_avg": round(late, 1),
                    "delta": round(late - early, 1),
                    "trend": "improving" if late > early else "declining" if late < early else "stable",
                }
        return trends
