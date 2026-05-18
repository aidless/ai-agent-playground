"""Multi-Agent Matrix — specialized agents, different models, intelligent routing.

Each agent in the matrix:
  - Has its own model (DeepSeek V4, Qwen2.5, local model)
  - Has its own role + specialty
  - Has its own subset of tools
  - Returns results with confidence scores

The matrix router:
  - Analyzes incoming tasks → classifies by type
  - Routes to the best agent(s)
  - Runs in parallel when multiple agents are assigned
  - Aggregates results with confidence-weighted voting

This is the "一群Agent各司其职" (TipKay-style) architecture —
specialized > general for production reliability.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Agent Profiles ──────────────────────────────

@dataclass
class MatrixAgentProfile:
    agent_id: str
    name: str
    role: str               # "reasoner", "coder", "reviewer", "searcher", "writer"
    model: str              # "deepseek-chat", "qwen2.5:7b", "qwen2.5-coder:7b", etc.
    client: Any             # AsyncOpenAI client for this agent's model
    tools: list[str] = field(default_factory=list)  # tool names this agent can use
    system_prompt: str = ""
    confidence_weight: float = 1.0  # historical reliability weight

    @property
    def is_available(self) -> bool:
        return self.client is not None


@dataclass
class MatrixResult:
    agent_id: str
    agent_name: str
    role: str
    model: str
    output: str
    confidence: float      # self-reported confidence (derived from response)
    latency_ms: float
    success: bool
    error: str = ""


@dataclass
class MatrixAggregation:
    task: str
    results: list[MatrixResult] = field(default_factory=list)
    final_output: str = ""
    consensus: dict[str, int] = field(default_factory=dict)
    routing_decision: str = ""     # "single" | "parallel" | "sequential"
    total_latency_ms: float = 0.0
    completed: bool = False


# ── Task Classification ─────────────────────────

TASK_PATTERNS = [
    # (regex, role, weight)
    (r"(写|编写|实现|代码|code|program|编程|开发|function|class|def |import |bug|fix|debug|refactor)", "coder", 0.9),
    (r"(推理|分析|思考|reason|analyze|why|explain|逻辑|论证|评估|建议)", "reasoner", 0.9),
    (r"(审查|review|检查|check|审计|audit|安全|漏洞|security|scan)", "reviewer", 0.85),
    (r"(搜索|search|查找|查询|find|信息|资料|research|最新)", "searcher", 0.85),
    (r"(写|文档|报告|文章|blog|readme|documentation|总结|summary)", "writer", 0.8),
]

ROLE_PROMPTS = {
    "reasoner": (
        "You are a senior reasoning specialist. Think step by step. "
        "Analyze deeply. Consider edge cases. Be precise and logical."
    ),
    "coder": (
        "You are a senior software engineer. Write clean, correct, efficient code. "
        "Include error handling. Follow best practices. Output code with brief explanation."
    ),
    "reviewer": (
        "You are a senior code reviewer and security auditor. Find bugs, "
        "security issues, performance problems, and style violations. Be thorough."
    ),
    "searcher": (
        "You are a research specialist. Find and synthesize information. "
        "Provide accurate, well-sourced answers. Be comprehensive."
    ),
    "writer": (
        "You are a technical writer. Produce clear, well-structured documentation. "
        "Use proper formatting. Be concise yet complete."
    ),
}


class AgentMatrix:
    """Multi-agent matrix with intelligent routing and aggregation.

    Usage:
        matrix = AgentMatrix()
        matrix.add_agent(MatrixAgentProfile(
            agent_id="deepseek-reasoner",
            name="DeepSeek Reasoner",
            role="reasoner",
            model="deepseek-chat",
            client=deepseek_client,
        ))
        matrix.add_agent(MatrixAgentProfile(
            agent_id="qwen-coder",
            name="Qwen Coder",
            role="coder",
            model="qwen2.5-coder:7b",
            client=ollama_client,
        ))
        result = await matrix.solve("Write a FastAPI rate limiter")
    """

    def __init__(self):
        self.agents: dict[str, MatrixAgentProfile] = {}
        self._history: list[MatrixAggregation] = []

    def add_agent(self, profile: MatrixAgentProfile):
        if not profile.system_prompt:
            profile.system_prompt = ROLE_PROMPTS.get(profile.role, "")
        self.agents[profile.agent_id] = profile
        logger.info("Matrix agent registered: %s (role=%s, model=%s)",
                    profile.name, profile.role, profile.model)

    def remove_agent(self, agent_id: str):
        self.agents.pop(agent_id, None)

    def classify_task(self, task: str) -> list[tuple[str, float]]:
        """Classify task → [(role, confidence), ...]

        Priority rules prevent misrouting (e.g. 'review this code' should
        route to reviewer, not coder).
        """
        scores: dict[str, float] = {}
        for pattern, role, weight in TASK_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                scores[role] = max(scores.get(role, 0), weight)

        # Rule: if reviewer is highest, suppress coder (review != write code)
        if scores.get("reviewer", 0) >= scores.get("coder", 0) + 0.1:
            scores.pop("coder", None)

        # Rule: if searcher is highest, don't also route to reasoner
        if scores.get("searcher", 0) >= scores.get("reasoner", 0) + 0.1:
            scores.pop("reasoner", None)

        if not scores:
            scores["reasoner"] = 0.5

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def route(self, task: str) -> list[MatrixAgentProfile]:
        """Determine which agents should handle this task."""
        classifications = self.classify_task(task)
        selected = []

        # Get top role
        top_role, top_score = classifications[0]
        # Find all agents matching top role
        matched = [a for a in self.agents.values()
                   if a.role == top_role and a.is_available]

        if matched:
            # If high confidence, use just the best agent for this role
            if top_score >= 0.85:
                # Best agent = highest confidence_weight
                best = max(matched, key=lambda a: a.confidence_weight)
                selected.append(best)
            else:
                selected.extend(matched)

        # If task has secondary classification and confidence is moderate
        if len(classifications) > 1 and classifications[1][1] >= 0.6:
            second_role = classifications[1][0]
            if second_role != top_role:
                second_matched = [a for a in self.agents.values()
                                  if a.role == second_role and a.is_available]
                if second_matched:
                    selected.append(second_matched[0])

        # Fallback: if no agent selected, use first available
        if not selected and self.agents:
            selected.append(list(self.agents.values())[0])

        return selected

    async def solve(self, task: str) -> MatrixAggregation:
        """Route task → run agents → aggregate results."""
        start = time.time()
        agents = self.route(task)
        agg = MatrixAggregation(task=task)

        if not agents:
            agg.error = "No agents available"
            agg.total_latency_ms = (time.time() - start) * 1000
            return agg

        routing_type = "single" if len(agents) == 1 else "parallel"
        agg.routing_decision = routing_type

        # Run agents in parallel
        futures = []
        for agent in agents:
            futures.append(self._run_agent(agent, task))

        results = await asyncio.gather(*futures)
        agg.results = [r for r in results if r is not None]

        # Aggregate
        if len(agg.results) == 1:
            agg.final_output = agg.results[0].output
        else:
            agg.final_output = self._aggregate(task, agg.results)

        agg.total_latency_ms = (time.time() - start) * 1000
        agg.completed = any(r.success for r in agg.results)
        self._history.append(agg)
        if len(self._history) > 100:
            self._history = self._history[-50:]
        return agg

    async def _run_agent(self, agent: MatrixAgentProfile, task: str) -> MatrixResult:
        """Execute one agent on a task."""
        t0 = time.time()
        try:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": task},
            ]
            response = await agent.client.chat.completions.create(
                model=agent.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.5,
            )
            output = response.choices[0].message.content or ""
            confidence = self._estimate_confidence(output)
            return MatrixResult(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                role=agent.role,
                model=agent.model,
                output=output,
                confidence=confidence,
                latency_ms=(time.time() - t0) * 1000,
                success=True,
            )
        except Exception as e:
            return MatrixResult(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                role=agent.role,
                model=agent.model,
                output="",
                confidence=0,
                latency_ms=(time.time() - t0) * 1000,
                success=False,
                error=str(e),
            )

    def _estimate_confidence(self, output: str) -> float:
        """Estimate model's self-confidence from response characteristics."""
        score = 0.6  # base
        if len(output) > 100:
            score += 0.1
        if len(output) > 500:
            score += 0.1
        # Presence of uncertainty markers reduces confidence
        uncertainty = ["不确定", "可能", "maybe", "might", "不确定", "不太清楚"]
        if any(u in output for u in uncertainty):
            score -= 0.15
        # Presence of structured output increases confidence
        if "```" in output or "## " in output:
            score += 0.1
        return max(0.1, min(1.0, score))

    def _aggregate(self, task: str, results: list[MatrixResult]) -> str:
        """Aggregate multiple agent results into one output."""
        parts = []
        for r in results:
            if not r.success or not r.output:
                continue
            parts.append(
                f"## [{r.agent_name}] ({r.role}/{r.model}, confidence={r.confidence:.2f})\n\n"
                f"{r.output}\n"
            )
        return "\n\n---\n\n".join(parts)

    def status(self) -> dict:
        return {
            "agents": [
                {
                    "id": a.agent_id, "name": a.name, "role": a.role,
                    "model": a.model, "tools": a.tools,
                    "available": a.is_available, "weight": a.confidence_weight,
                }
                for a in self.agents.values()
            ],
            "recent_aggregations": [
                {
                    "task": agg.task[:100],
                    "agents_used": len(agg.results),
                    "latency_ms": round(agg.total_latency_ms),
                    "completed": agg.completed,
                }
                for agg in self._history[-5:]
            ],
        }
