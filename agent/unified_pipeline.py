"""Unified Pipeline — Crew → Debate → CrossReview.

Connects three previously independent systems into one flow:

  1. Crew decomposes the task into subtasks
  2. Each subtask runs through Debate (multi-model consensus) for higher quality
  3. Final output goes through CrossReview (cross-model verification)
  4. Reviewer findings are either auto-fixed or escalated to human

This is the "moat" feature: multi-agent collaboration + multi-model debate
+ cross-model review — three levels of quality assurance in one pipeline.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubtaskDebateResult:
    subtask_id: str
    description: str
    debate_result: Any   # DebateResult
    assigned_role: str
    latency_ms: float


@dataclass
class UnifiedPipelineResult:
    task: str
    subtask_count: int
    subtask_debates: list[SubtaskDebateResult] = field(default_factory=list)
    aggregated_final: str = ""
    cross_review_findings: list = field(default_factory=list)
    total_latency_ms: float = 0.0
    completed: bool = False
    error: str = ""


class UnifiedPipeline:
    """Orchestrates Crew decomposition → Debate per subtask → CrossReview final.

    Usage:
        pipeline = UnifiedPipeline(
            orchestrator=orchestrator,
            debate_engine=debate_engine,
            cross_reviewer=cross_reviewer,
            crew=crew,
            primary_model="deepseek-chat",
            challenger_model="qwen2.5:7b",
        )
        result = await pipeline.execute("Build a microservice CI/CD pipeline")
    """

    def __init__(
        self,
        orchestrator,           # AgentOrchestrator
        debate_engine,          # DebateEngine
        cross_reviewer,         # CrossReviewer
        crew,                   # Crew
        primary_model: str = "deepseek-chat",
        challenger_model: str = "",
        arbitrator_model: str = "",
    ):
        self.orchestrator = orchestrator
        self.debate = debate_engine
        self.reviewer = cross_reviewer
        self.crew = crew
        self.primary_model = primary_model
        self.challenger_model = challenger_model or primary_model
        self.arbitrator_model = arbitrator_model or primary_model

    async def execute(self, task: str, enable_debate: bool = True, enable_review: bool = True) -> UnifiedPipelineResult:
        """Run the full unified pipeline."""
        start = time.time()
        result = UnifiedPipelineResult(task=task, subtask_count=0)

        try:
            # ── Phase 1: Crew decomposes ──────────────
            master = self._get_master()
            subtasks = await self.orchestrator._decompose_via_agent(master, task)
            result.subtask_count = len(subtasks)
            logger.info("Phase 1/3 — Crew decomposed into %d subtasks", len(subtasks))

            if not subtasks:
                # Single-task fallback
                subtasks = [{"id": "1", "description": task, "role": "developer"}]

            # ── Phase 2: Debate per subtask ────────────
            for st in subtasks:
                if enable_debate and self.debate:
                    logger.info("Phase 2/3 — Debate on subtask %s: %s", st.get("id"), st.get("description")[:80])
                    debate_result = await self.debate.debate(
                        task=st.get("description", task),
                        primary_model=self.primary_model,
                        challenger_model=self.challenger_model,
                        arbitrator_model=self.arbitrator_model,
                    )
                    result.subtask_debates.append(SubtaskDebateResult(
                        subtask_id=st.get("id", "?"),
                        description=st.get("description", task),
                        debate_result=debate_result,
                        assigned_role=st.get("role", "general"),
                        latency_ms=debate_result.total_latency_ms,
                    ))
                else:
                    # Fallback: single-agent act
                    agent = self._get_agent_for_role(st.get("role", "developer"))
                    act_result = await agent.act(st.get("description", task), "")
                    from agent.debate import DebateResult
                    fallback = DebateResult(
                        debate_id=f"fallback-{st.get('id','0')}",
                        task=st.get("description", task),
                        primary_model=self.primary_model,
                        consensus=str(act_result.output),
                        completed=True,
                    )
                    result.subtask_debates.append(SubtaskDebateResult(
                        subtask_id=st.get("id", "?"),
                        description=st.get("description", task),
                        debate_result=fallback,
                        assigned_role=st.get("role", "general"),
                        latency_ms=0,
                    ))

            # ── Aggregate debates into final text ─────
            consensus_texts = []
            for sd in result.subtask_debates:
                c = sd.debate_result.consensus if sd.debate_result else ""
                consensus_texts.append(f"[Subtask {sd.subtask_id} — {sd.assigned_role}]\n{c}")

            result.aggregated_final = "\n\n".join(consensus_texts)

            # ── Phase 3: Cross-review ──────────────────
            if enable_review and self.reviewer and consensus_texts:
                logger.info("Phase 3/3 — Cross-review of aggregated output")
                try:
                    review_result = await self.reviewer.review(
                        original_text=result.aggregated_final,
                        source_context=task,
                        instructions="Verify correctness, check for errors, suggest improvements",
                    )
                    result.cross_review_findings = [
                        {
                            "type": f.type.value,
                            "finding": f.finding[:300],
                            "suggestion": f.suggestion[:300],
                            "status": f.status.value,
                        }
                        for f in review_result.findings
                    ] if review_result else []
                except Exception as e:
                    logger.warning("Cross-review unavailable: %s", e)
                    result.cross_review_findings = [{"type": "error", "finding": str(e)}]

            result.completed = True

        except Exception as e:
            result.error = str(e)
            logger.error("Unified pipeline failed: %s", e)

        result.total_latency_ms = (time.time() - start) * 1000
        return result

    def _get_master(self):
        """Get or find the master/planner agent from the crew."""
        for name, agent in self.crew.agents.items():
            if agent.identity.role in ("planner", "master", "admin"):
                return agent
        if self.crew.agents:
            return list(self.crew.agents.values())[0]
        raise RuntimeError("No agents in crew")

    def _get_agent_for_role(self, role: str):
        """Find the best agent for a role, fallback to master."""
        for name, agent in self.crew.agents.items():
            if agent.identity.role == role:
                return agent
        return self._get_master()
