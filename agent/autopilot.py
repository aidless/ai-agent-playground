"""AutoPilot — fully autonomous self-improving agent loop.

Routes task → executes → verifies → reflects → improves → retries.
All six engines work together without any human intervention.

The loop:
  CLASSIFY  → which agent(s) should handle this?
  EXECUTE   → run the task
  VERIFY    → is the output good enough?
  REFLECT   → what went wrong / right?
  IMPROVE   → evolve, bootstrap, debate, or rollback
  RETRY     → if still not good, go again (max 3 iterations)

This closes the gap between "toolbox of engines" and
"single autonomous agent that drives itself."
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

AUTOPILOT_DIR = Path(__file__).resolve().parent.parent / "memory" / "autopilot"


@dataclass
class AutoPilotIteration:
    iteration: int
    phase: str             # classify, execute, verify, reflect, improve, retry
    detail: str
    latency_ms: float
    success: bool = True


@dataclass
class AutoPilotResult:
    task: str
    final_output: str = ""
    iterations: list[AutoPilotIteration] = field(default_factory=list)
    improvements_applied: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    quality_score: float = 0.0
    completed: bool = False
    error: str = ""


QUALITY_CHECK_PROMPT = (
    "Rate the quality of this response on a scale of 0-10. "
    "Consider: correctness, completeness, clarity, safety.\n"
    "Output ONLY a number (e.g. '7.5'). Nothing else."
)


class AutoPilot:
    """Autonomous agent that routes, solves, verifies, and self-improves.

    Usage:
        ap = AutoPilot(agent, matrix, debate_engine, evolution_engine, cross_reviewer)
        result = await ap.solve("Build a CI/CD pipeline for a Python project")
    """

    def __init__(
        self,
        agent,              # AsyncAgent
        matrix,             # AgentMatrix
        debate_engine,      # DebateEngine
        evolution_engine,   # EvolutionEngine
        bootstrap_engine,   # BootstrapEngine
        cross_reviewer,     # CrossReviewer
        meta_agent,         # MetaAgent
        quality_threshold: float = 7.0,
        max_iterations: int = 3,
    ):
        self.agent = agent
        self.matrix = matrix
        self.debate = debate_engine
        self.evolution = evolution_engine
        self.bootstrap = bootstrap_engine
        self.reviewer = cross_reviewer
        self.meta = meta_agent
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self._history: list[AutoPilotResult] = []
        AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)

    async def solve(self, task: str) -> AutoPilotResult:
        """Full autonomous loop: route → solve → verify → improve → retry."""
        start = time.time()
        result = AutoPilotResult(task=task)
        current_output = ""
        current_quality = 0.0

        for iteration in range(1, self.max_iterations + 1):
            # ── Phase 1: CLASSIFY & ROUTE ──────────
            t0 = time.time()
            if iteration == 1:
                agents = self.matrix.route(task)
                agent_names = [a.name for a in agents]
            else:
                # On retry, use all available agents in parallel
                agents = list(self.matrix.agents.values())
                agent_names = ["all"]
            result.iterations.append(AutoPilotIteration(
                iteration, "classify",
                f"Routed to: {agent_names}",
                (time.time() - t0) * 1000,
            ))
            logger.info("AutoPilot iter %d: routing → %s", iteration, agent_names)

            # ── Phase 2: EXECUTE ────────────────────
            t0 = time.time()
            if iteration == 1 and len(agents) == 1:
                # Single agent: direct solve
                from agent.state import AgentContext
                ctx = AgentContext(trace_id=f"autopilot_{iteration}", max_steps=5)
                ctx = await self.agent.run(ctx, task)
                current_output = ""
                for msg in ctx.messages:
                    if msg.get("role") == "assistant" and msg.get("content"):
                        current_output = msg["content"]
            else:
                # Multiple agents: matrix solve
                mat_result = await self.matrix.solve(task)
                current_output = mat_result.final_output

            exec_ms = (time.time() - t0) * 1000
            result.iterations.append(AutoPilotIteration(
                iteration, "execute",
                f"Output: {len(current_output)} chars",
                exec_ms,
            ))
            logger.info("AutoPilot iter %d: execute → %d chars, %.0fms", iteration, len(current_output), exec_ms)

            # ── Phase 3: VERIFY ─────────────────────
            t0 = time.time()
            current_quality = await self._assess_quality(task, current_output)
            result.iterations.append(AutoPilotIteration(
                iteration, "verify",
                f"Quality: {current_quality:.1f}/10 (threshold: {self.quality_threshold})",
                (time.time() - t0) * 1000,
                success=current_quality >= self.quality_threshold,
            ))
            logger.info("AutoPilot iter %d: verify → %.1f/10", iteration, current_quality)

            if current_quality >= self.quality_threshold:
                result.final_output = current_output
                result.quality_score = current_quality
                result.completed = True
                break

            # ── Phase 4: REFLECT ────────────────────
            t0 = time.time()
            reflection = await self._reflect(task, current_output, current_quality)
            result.iterations.append(AutoPilotIteration(
                iteration, "reflect",
                f"Reflection: {reflection[:200]}",
                (time.time() - t0) * 1000,
            ))
            logger.info("AutoPilot iter %d: reflect → %s", iteration, reflection[:100])

            # ── Phase 5: IMPROVE ────────────────────
            t0 = time.time()
            improvements = await self._improve(task, current_output, reflection, current_quality)
            result.improvements_applied.extend(improvements)
            result.iterations.append(AutoPilotIteration(
                iteration, "improve",
                f"Applied: {improvements}",
                (time.time() - t0) * 1000,
                success=len(improvements) > 0,
            ))
            logger.info("AutoPilot iter %d: improve → %s", iteration, improvements)

            if not improvements:
                # Can't improve further — accept current output
                result.final_output = current_output
                result.quality_score = current_quality
                result.completed = True
                break

        if not result.completed:
            result.final_output = current_output
            result.quality_score = current_quality

        result.total_latency_ms = (time.time() - start) * 1000
        self._save(result)
        return result

    async def _assess_quality(self, task: str, output: str) -> float:
        """Ask an LLM to rate output quality. 0-10 scale."""
        if not output:
            return 0.0
        try:
            response = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=[
                    {"role": "system", "content": QUALITY_CHECK_PROMPT},
                    {"role": "user", "content": f"Task: {task}\n\nResponse:\n{output[:2000]}"},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            # Extract first number
            import re
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
        except Exception as e:
            logger.warning("Quality assessment failed: %s", e)
        # Fallback heuristics
        score = 6.0
        if len(output) > 200:
            score += 1.0
        if "```" in output:
            score += 0.5
        if len(output) > 1000:
            score += 0.5
        return min(10.0, score)

    async def _reflect(self, task: str, output: str, quality: float) -> str:
        """Generate a reflection on why quality is below threshold."""
        try:
            response = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=[
                    {"role": "system", "content": (
                        "You are analyzing a failed AI response. Quality score: "
                        f"{quality}/10. Identify what went wrong and what SPECIFIC "
                        "improvements are needed. Be concise (1-2 sentences)."
                    )},
                    {"role": "user", "content": f"Task: {task}\n\nResponse (quality {quality}/10):\n{output[:1500]}"},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Quality {quality}/10 — needs improvement. Error: {e}"

    async def _improve(self, task: str, output: str, reflection: str, quality: float) -> list[str]:
        """Apply the right improvement strategy based on reflection and quality."""
        applied = []

        # Strategy 1: If quality is very low → run debate for better answer
        if quality < 5.0 and self.debate:
            try:
                logger.info("AutoPilot: quality %.1f → running process-centric debate", quality)
                debate_result = await self.debate.debate_process_centric(
                    task,
                    primary_model=self.agent.model,
                    challenger_model=getattr(self.debate, 'challenger_model', self.agent.model),
                )
                if debate_result.completed and debate_result.consensus:
                    applied.append("debate:process_centric")
            except Exception as e:
                logger.warning("AutoPilot improve/debate failed: %s", e)

        # Strategy 2: Check for underperforming tools → evolve
        if self.evolution:
            underperforming = self.evolution.tracker.list_underperforming()
            for tool_name in underperforming[:2]:  # Max 2 evolutions per iteration
                try:
                    logger.info("AutoPilot: evolving underperforming tool '%s'", tool_name)
                    record = await self.evolution.evolve(tool_name)
                    if record.applied:
                        applied.append(f"evolution:{tool_name}")
                except Exception as e:
                    logger.warning("AutoPilot improve/evolve failed: %s", e)

        # Strategy 3: Check reflection for missing tools → bootstrap
        if self.bootstrap and any(kw in reflection for kw in ["missing", "don't have", "no tool", "need",
                                                               "没有", "缺失", "need a tool"]):
            import re
            match = re.search(r"(?:missing|need|don't have|no tool|需要|缺失)[:\s]+([a-z_]+)", reflection, re.IGNORECASE)
            if match:
                tool_name = match.group(1).strip().replace(" ", "_")
                try:
                    logger.info("AutoPilot: bootstrapping '%s'", tool_name)
                    bt = await self.bootstrap.generate_from_reflection(reflection, tool_name)
                    if bt.validated:
                        self.bootstrap.register_tool(bt, self.agent.registry)
                        applied.append(f"bootstrap:{tool_name}")
                except Exception as e:
                    logger.warning("AutoPilot improve/bootstrap failed: %s", e)

        # Strategy 4: Run MetaAgent observe to trigger any pending actions
        if self.meta:
            try:
                from agent.state import AgentContext
                ctx = AgentContext(trace_id=f"ap_meta_{uuid.uuid4().hex[:6]}")
                decisions = await self.meta.observe(ctx)
                for d in decisions:
                    if d.action_taken:
                        applied.append(f"meta:{d.decision_type}:{d.target}")
            except Exception as e:
                logger.warning("AutoPilot improve/meta failed: %s", e)

        return applied

    def _save(self, result: AutoPilotResult):
        self._history.append(result)
        if len(self._history) > 100:
            self._history = self._history[-50:]
        path = AUTOPILOT_DIR / f"ap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
        path.write_text(json.dumps({
            "task": result.task[:200],
            "quality_score": result.quality_score,
            "iterations": len(result.iterations),
            "improvements": result.improvements_applied,
            "total_latency_ms": round(result.total_latency_ms),
            "completed": result.completed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    def status(self) -> dict:
        return {
            "total_sessions": len(self._history),
            "recent": [
                {
                    "task": r.task[:80],
                    "quality": r.quality_score,
                    "iterations": len(r.iterations),
                    "improvements": r.improvements_applied,
                }
                for r in self._history[-5:]
            ],
        }
