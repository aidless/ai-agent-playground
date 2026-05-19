"""Test-Time Reinforcement Learning — adaptive policy without weight updates.

David Silver's TTRL direction: the agent learns during inference through
reward signals, improving action selection in real-time without fine-tuning.

Our approach (no gradient updates available for the LLM):
  1. Track (context, action, reward) triples at test time
  2. Build an ActionPreferenceModel that adjusts tool/strategy weights
  3. Use exponential moving average of rewards per context-action pair
  4. Guide future action selection based on learned preferences

This is a lightweight policy gradient over the agent's meta-actions
(tool selection, strategy choice) rather than over model parameters.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TTRL_DIR = Path(__file__).resolve().parent.parent / "memory" / "ttrl"


@dataclass
class Experience:
    context_key: str     # hash of task type + first few words
    action: str          # tool name / strategy
    reward: float        # 0.0-1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ActionPreferenceModel:
    """Learns which actions work best in which contexts from test-time rewards.

    Uses exponential moving average: Q(c,a) = (1-α)·Q(c,a) + α·r
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._q: dict[str, dict[str, float]] = {}    # context -> {action: q_value}
        self._counts: dict[str, dict[str, int]] = {}  # context -> {action: count}
        TTRL_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        path = TTRL_DIR / "preferences.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._q = data.get("q", {})
            self._counts = data.get("counts", {})

    def _save(self):
        (TTRL_DIR / "preferences.json").write_text(json.dumps({
            "q": self._q, "counts": self._counts,
        }, indent=2), encoding="utf-8")

    def update(self, context_key: str, action: str, reward: float):
        if context_key not in self._q:
            self._q[context_key] = {}
            self._counts[context_key] = {}
        if action not in self._q[context_key]:
            self._q[context_key][action] = 0.5  # neutral prior
            self._counts[context_key][action] = 0

        old = self._q[context_key][action]
        self._q[context_key][action] = (1 - self.alpha) * old + self.alpha * reward
        self._counts[context_key][action] += 1

    def get_preference(self, context_key: str, action: str) -> float:
        return self._q.get(context_key, {}).get(action, 0.5)

    def best_action(self, context_key: str, candidates: list[str]) -> Optional[str]:
        if context_key not in self._q:
            return None
        best = None
        best_q = 0.0
        for action in candidates:
            q = self._q[context_key].get(action, 0.5)
            # Add exploration bonus for under-explored actions
            count = self._counts.get(context_key, {}).get(action, 0)
            exploration_bonus = 0.1 / (1 + math.log(1 + count))
            score = q + exploration_bonus
            if score > best_q:
                best_q = score
                best = action
        return best


class TTRLEngine:
    """Test-Time RL — learns from agent execution outcomes in real-time.

    Usage:
        engine = TTRLEngine()
        # Before action:
        ctx_key = engine.make_context(task_type, user_input)
        preferred = engine.model.best_action(ctx_key, candidates)
        # After result:
        engine.record(ctx_key, action, reward=1.0 if success else 0.0)
    """

    def __init__(self):
        self.model = ActionPreferenceModel()
        self._experiences: list[Experience] = []

    def make_context(self, task_type: str, user_input: str) -> str:
        return f"{task_type}:{user_input[:40]}"

    def record(self, context_key: str, action: str, reward: float):
        self.model.update(context_key, action, reward)
        self._experiences.append(Experience(
            context_key=context_key, action=action, reward=reward,
        ))
        if len(self._experiences) > 1000:
            self._experiences = self._experiences[-500:]
        self.model._save()

    def get_action_reward(self, context_key: str, action: str) -> float:
        return self.model.get_preference(context_key, action)

    def select_tool(self, context_key: str, candidates: list[str]) -> Optional[str]:
        return self.model.best_action(context_key, candidates)

    def reward_from_result(self, success: bool, latency_ms: float = 0) -> float:
        """Convert tool execution result to a reward signal."""
        if success:
            # Bonus for speed
            if latency_ms < 1000:
                return 1.0
            elif latency_ms < 5000:
                return 0.8
            return 0.6
        else:
            return 0.0

    def status(self) -> dict:
        contexts = list(self._q.keys())[-5:] if hasattr(self, '_q') else []
        return {
            "total_experiences": len(self._experiences),
            "unique_contexts": len(self._q),
            "recent_contexts": contexts,
            "top_actions": self._top_actions(5),
        }

    def _top_actions(self, n: int) -> list[dict]:
        results = []
        for ctx, actions in self._q.items():
            for action, q_val in actions.items():
                count = self._counts.get(ctx, {}).get(action, 0)
                results.append({"context": ctx[:40], "action": action, "q": round(q_val, 3), "count": count})
        results.sort(key=lambda x: x["q"], reverse=True)
        return results[:n]

    @property
    def _q(self):
        return self.model._q

    @property
    def _counts(self):
        return self.model._counts
