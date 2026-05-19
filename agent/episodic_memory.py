"""Episodic Reflection Memory — Reflexion paper's verbal reinforcement learning.

Stores agent reflections indexed by task type. On new tasks, retrieves
relevant past reflections and injects them as context, enabling the agent
to learn from past failures without weight updates.

Key insight from Reflexion (Shinn et al., 2023):
  "The agent maintains a heuristic-based 'reflective memory' which it
   updates after each action. When it receives a negative signal, it
   reflects on the failure and stores the insight. In the next trial,
   the agent is prompted with these reflections, enabling it to learn
   from past mistakes without requiring any gradient updates."
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory" / "episodic"


@dataclass
class EpisodicMemory:
    reflection: str
    task_type: str = "general"
    success: bool = False
    tool_name: str = ""
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reuse_count: int = 0
    last_reused: str = ""


class EpisodicMemoryStore:
    """Stores and retrieves agent reflections indexed by task type.

    Usage:
        store = EpisodicMemoryStore()
        store.add("I should check the API docs before calling endpoints",
                  task_type="code_generation", tool_name="web_search", success=False)

        # On next similar task:
        relevant = store.retrieve("code_generation", k=3)
        # Injects into agent context: "Past experiences: [reflections]"
    """

    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._memories: list[EpisodicMemory] = []
        self._load()

    def _load(self):
        path = MEMORY_DIR / "reflections.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    data = json.loads(line)
                    self._memories.append(EpisodicMemory(**data))
                except Exception:
                    pass

    def _save(self, entry: EpisodicMemory):
        self._memories.append(entry)
        with open(MEMORY_DIR / "reflections.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "reflection": entry.reflection,
                "task_type": entry.task_type,
                "success": entry.success,
                "tool_name": entry.tool_name,
                "error": entry.error,
                "timestamp": entry.timestamp,
                "reuse_count": entry.reuse_count,
                "last_reused": entry.last_reused,
            }, ensure_ascii=False) + "\n")

    def add(self, reflection: str, task_type: str = "general",
            tool_name: str = "", success: bool = True, error: str = ""):
        entry = EpisodicMemory(
            reflection=reflection,
            task_type=task_type,
            success=success,
            tool_name=tool_name,
            error=error,
        )
        self._save(entry)
        logger.debug("Episodic memory stored: %s (type=%s)", reflection[:60], task_type)

    def retrieve(self, task_type: str = "", k: int = 3, include_failures: bool = True,
                 tier: str = "hot") -> list[EpisodicMemory]:
        """Retrieve relevant past reflections. MemGPT-inspired tiered retrieval.

        hot tier: recent (<1 day), high reuse, task-matched (always included)
        warm tier: 1-7 days, moderate reuse (included if hot < k)
        cold tier: >7 days, low reuse (archive access only)
        """
        candidates = []
        now = datetime.now(timezone.utc)
        for m in self._memories:
            score = 0
            tier_weight = 0
            try:
                age_days = (now - datetime.fromisoformat(m.timestamp)).days
            except Exception:
                age_days = 999

            # Tier classification (MemGPT pattern)
            if age_days < 1:
                tier_weight = 10   # hot
            elif age_days < 7:
                tier_weight = 5    # warm
            else:
                tier_weight = 1    # cold

            if task_type and m.task_type == task_type:
                score += tier_weight + 3
            else:
                score += tier_weight

            if not m.success:
                score += 2
            elif not include_failures:
                continue

            if m.reuse_count > 0:
                score += min(m.reuse_count * 0.5, 5)

            candidates.append((score, m))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in candidates[:k]]

    def build_context(self, task_type: str = "", k: int = 3) -> str:
        """Build a context string for injection into the agent prompt."""
        relevant = self.retrieve(task_type, k=k)
        if not relevant:
            return ""

        lines = ["## Past Experiences (learn from these):"]
        for i, m in enumerate(relevant, 1):
            icon = "[FAILURE]" if not m.success else "[SUCCESS]"
            lines.append(f"{i}. {icon} [{m.task_type}] {m.reflection}")
            # Mark reused
            m.reuse_count += 1
            m.last_reused = datetime.now(timezone.utc).isoformat()

        return "\n".join(lines)

    def classify_task(self, user_input: str) -> str:
        """Simple keyword-based task classification."""
        patterns = {
            "code_generation": [r"write|code|implement|function|bug|fix|debug|refactor|编程|实现"],
            "reasoning": [r"explain|why|how|analyze|think|reason|分析|解释"],
            "search": [r"search|find|lookup|查询|搜索|查找"],
            "security": [r"security|safety|vulnerability|attack|defend|安全|漏洞"],
            "design": [r"design|architecture|system|design|架构|设计"],
        }
        for task_type, pat_list in patterns.items():
            if any(re.search(p, user_input, re.IGNORECASE) for p in pat_list):
                return task_type
        return "general"

    def synthesize(self, task_type: str = "") -> str:
        """Synthesize multiple reflections into a higher-level insight.

        Generative Agents (Park et al., 2023): "synthesize memories over time
        into higher-level reflections."
        """
        candidates = self.retrieve(task_type, k=10, include_failures=True)
        if len(candidates) < 5:
            return ""

        failures = [m for m in candidates if not m.success]
        successes = [m for m in candidates if m.success]

        parts = []
        if failures:
            common_errors = set(m.reflection[:80] for m in failures[:3])
            parts.append(f"Common failures in '{task_type}': {'; '.join(common_errors)}")
        if successes:
            patterns = set(m.reflection[:80] for m in successes[:3])
            parts.append(f"Successful patterns in '{task_type}': {'; '.join(patterns)}")

        return " | ".join(parts) if parts else ""

    def status(self) -> dict:
        by_type = {}
        for m in self._memories:
            by_type.setdefault(m.task_type, {"total": 0, "failures": 0})
            by_type[m.task_type]["total"] += 1
            if not m.success:
                by_type[m.task_type]["failures"] += 1
        return {
            "total_memories": len(self._memories),
            "by_type": by_type,
            "recent": [m.reflection[:80] for m in self._memories[-5:]],
        }
