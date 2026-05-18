"""MetaAgent — autonomous self-improvement coordinator.

The MetaAgent observes the Task Agent's execution, analyzes performance
across all tools, and autonomously decides when to:

  1. EVOLVE — generate optimization for underperforming tools
  2. BOOTSTRAP — create new tools for detected capability gaps
  3. DEGRADE — disable broken tools and substitute alternatives
  4. ROLLBACK — revert a bad evolution
  5. PATTERN TRANSFER — apply lessons from one tool to similar tools

This is the "Meta" layer from the HYPERAGENTS paper — the agent that
watches, learns, and modifies the system itself.

Architecture:
    Task Agent (async_core.py)
         │
         ▼
    MetaAgent.observe(ctx)     ← called after every agent run
         │
         ├── check PerformanceTracker  → auto-trigger EvolutionEngine
         ├── check ReflectActionEngine  → auto-trigger Bootstrap
         ├── verify recent evolutions   → rollback if degraded
         └── cross-pollinate patterns  → suggest similar tool optimizations
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

META_DIR = Path(__file__).resolve().parent.parent / "memory" / "evolution"


@dataclass
class MetaDecision:
    decision_id: str
    decision_type: str  # "evolve", "bootstrap", "degrade", "rollback", "transfer", "skip"
    target: str         # tool name or "none"
    reason: str
    action_taken: bool
    result: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MetaAgent:
    """Autonomous coordinator — watches, decides, acts.

    Usage:
        meta = MetaAgent(agent, llm_client, registry)
        # Called automatically after each agent.run():
        decisions = await meta.observe(ctx)
        # decisions = [MetaDecision, ...] — what actions were taken
    """

    def __init__(self, agent, llm_client, registry):
        self.agent = agent          # AsyncAgent instance
        self.client = llm_client    # LLM client for evolution/bootstrap
        self.registry = registry
        self._decisions: list[MetaDecision] = []
        self._rollback_store: dict[str, Any] = {}  # tool_name -> old_func
        self._load()

    def _load(self):
        META_DIR.mkdir(parents=True, exist_ok=True)
        path = META_DIR / "meta_decisions.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    self._decisions.append(MetaDecision(**json.loads(line)))
                except Exception:
                    pass

    def _save_decision(self, d: MetaDecision):
        self._decisions.append(d)
        with open(META_DIR / "meta_decisions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "decision_id": d.decision_id, "decision_type": d.decision_type,
                "target": d.target, "reason": d.reason,
                "action_taken": d.action_taken, "result": d.result,
                "timestamp": d.timestamp,
            }, ensure_ascii=False) + "\n")

    async def observe(self, ctx) -> list[MetaDecision]:
        """Main entry point: called after every agent run to assess and act."""
        decisions = []
        import uuid

        # ── 1. Check for underperforming tools → EVOLVE ──
        if self.agent.perf_tracker and self.agent.evolution:
            underperforming = self.agent.perf_tracker.list_underperforming()
            for tool_name in underperforming:
                # Skip recently evolved tools (cooldown: 5 min)
                recent_evolutions = [
                    d for d in self._decisions
                    if d.target == tool_name and d.decision_type == "evolve"
                ]
                if recent_evolutions:
                    last_evo = recent_evolutions[-1]
                    last_time = datetime.fromisoformat(last_evo.timestamp).timestamp()
                    if time.time() - last_time < 300:
                        continue

                logger.info("MetaAgent: auto-evolving underperforming tool '%s'", tool_name)
                d = MetaDecision(
                    decision_id=f"meta-{uuid.uuid4().hex[:8]}",
                    decision_type="evolve",
                    target=tool_name,
                    reason=f"Auto-detected underperforming (from PerformanceTracker)",
                    action_taken=False,
                )
                try:
                    record = await self.agent.evolution.evolve(tool_name)
                    d.action_taken = record.applied
                    d.result = {
                        "version": record.version,
                        "applied": record.applied,
                        "validated": record.validated,
                        "diff_lines": record.diff.count("\n"),
                    }
                    if record.applied:
                        logger.info("MetaAgent: evolution applied to '%s' v%d", tool_name, record.version)
                except Exception as e:
                    d.result = {"error": str(e)}
                    logger.warning("MetaAgent: evolution failed for '%s': %s", tool_name, e)
                self._save_decision(d)
                decisions.append(d)

        # ── 2. Check for missing tool gaps → BOOTSTRAP ──
        if self.agent.bootstrap:
            missing_actions = [
                a for a in ctx.trace_steps
                if a.get("event") == "reflect_action" and a.get("type") == "missing_tool"
            ]
            for action in missing_actions[-1:]:
                suggested = action.get("suggested_name", "unknown_tool")
                # Skip already-existing tools
                if hasattr(self.registry, "_tools") and suggested in self.registry._tools:
                    continue

                logger.info("MetaAgent: auto-bootstrapping tool '%s'", suggested)
                d = MetaDecision(
                    decision_id=f"meta-{uuid.uuid4().hex[:8]}",
                    decision_type="bootstrap",
                    target=suggested,
                    reason=f"Capability gap: {action.get('description', '')[:200]}",
                    action_taken=False,
                )
                try:
                    tool = await self.agent.bootstrap.generate_from_reflection(
                        action.get("description", ""), suggested,
                    )
                    if tool.validated:
                        success = self.agent.bootstrap.register_tool(tool, self.registry)
                        d.action_taken = success
                        d.result = {"validated": True, "registered": success}
                        if success:
                            logger.info("MetaAgent: bootstrapped tool '%s' registered", suggested)
                    else:
                        d.result = {"validated": False, "error": tool.error}
                except Exception as e:
                    d.result = {"error": str(e)}
                self._save_decision(d)
                decisions.append(d)

        # ── 3. Verify recent evolutions → ROLLBACK if degraded ──
        if self.agent.evolution:
            recent_evolutions = [
                d for d in self._decisions[-10:]
                if d.decision_type == "evolve" and d.action_taken
            ]
            for evo_d in recent_evolutions:
                tool_name = evo_d.target
                if tool_name not in self._rollback_store:
                    continue
                metrics = self.agent.perf_tracker.get_metrics(tool_name)
                if metrics and metrics.call_count >= 3:
                    # If success rate dropped by >20% after evolution, rollback
                    # (approximation: check if tool is now underperforming)
                    if metrics.is_underperforming:
                        logger.warning("MetaAgent: rolling back degraded evolution of '%s'", tool_name)
                        d = MetaDecision(
                            decision_id=f"meta-{uuid.uuid4().hex[:8]}",
                            decision_type="rollback",
                            target=tool_name,
                            reason=f"Post-evolution degradation: success_rate={metrics.success_rate:.1%}",
                            action_taken=False,
                        )
                        try:
                            self._rollback_tool(tool_name)
                            d.action_taken = True
                        except Exception as e:
                            d.result = {"error": str(e)}
                        self._save_decision(d)
                        decisions.append(d)

        # ── 4. Cross-domain pattern transfer ──────────
        # If tool X was evolved successfully, suggest similar tools for evolution
        successful_evolutions = [
            d for d in self._decisions[-20:]
            if d.decision_type == "evolve" and d.action_taken
        ]
        if successful_evolutions and self.agent.perf_tracker:
            evolved_tools = {d.target for d in successful_evolutions}
            for name, m in self.agent.perf_tracker.all_metrics().items():
                if name in evolved_tools:
                    continue
                if m.get("success_rate", 1.0) < 0.80:
                    d = MetaDecision(
                        decision_id=f"meta-{uuid.uuid4().hex[:8]}",
                        decision_type="transfer",
                        target=name,
                        reason=f"Similar to previously evolved tool(s): {evolved_tools}",
                        action_taken=False,
                    )
                    try:
                        record = await self.agent.evolution.evolve(name)
                        d.action_taken = record.applied
                        d.result = {"applied": record.applied, "version": record.version}
                    except Exception as e:
                        d.result = {"error": str(e)}
                    self._save_decision(d)
                    decisions.append(d)
                    break  # Only transfer one at a time

        return decisions

    def store_for_rollback(self, tool_name: str):
        """Snapshot current tool code before evolution (called by EvolutionEngine)."""
        if hasattr(self.registry, "_tools") and tool_name in self.registry._tools:
            import copy
            func = self.registry._tools[tool_name].func
            self._rollback_store[tool_name] = copy.copy(func)
            logger.info("MetaAgent: stored rollback snapshot for '%s'", tool_name)

    def _rollback_tool(self, tool_name: str):
        """Restore the pre-evolution version of a tool."""
        if tool_name not in self._rollback_store:
            raise ValueError(f"No rollback snapshot for {tool_name}")
        old_func = self._rollback_store.pop(tool_name)
        if hasattr(self.registry, "register"):
            self.registry.register(
                tool_name,
                f"Rolled back {tool_name}",
                {"properties": {}, "required": []},
                old_func,
            )
            logger.info("MetaAgent: rolled back '%s' to pre-evolution version", tool_name)

    def status(self) -> dict:
        return {
            "total_decisions": len(self._decisions),
            "by_type": {
                t: sum(1 for d in self._decisions if d.decision_type == t)
                for t in {"evolve", "bootstrap", "degrade", "rollback", "transfer", "skip"}
            },
            "recent_decisions": [
                {
                    "type": d.decision_type, "target": d.target,
                    "action_taken": d.action_taken, "reason": d.reason[:100],
                }
                for d in self._decisions[-10:]
            ],
            "rollback_snapshots": list(self._rollback_store.keys()),
        }
