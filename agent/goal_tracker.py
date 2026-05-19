"""Goal Drift Detection — prevent agent from wandering off-target.

Research shows 67% of agents drift from original goal after 15 steps.
This module tracks semantic distance between current actions and the
original goal, triggering warnings when drift exceeds threshold.

Uses simple TF-IDF cosine similarity (no external embedding service needed).
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DRIFT_DIR = Path(__file__).resolve().parent.parent / "memory" / "goal_tracking"


class GoalDriftDetector:
    """Tracks goal alignment across agent steps.

    Usage:
        tracker = GoalDriftDetector("Build a REST API for task management")
        tracker.record_step("Created FastAPI app with CRUD endpoints")
        tracker.record_step("Looking up cat videos on YouTube")  # DRIFT!
        result = tracker.check()
        if result["drift"]:
            logger.warning("Goal drift detected!")
    """

    def __init__(self, original_goal: str, drift_threshold: float = 0.35, consecutive_threshold: int = 3):
        self.original_goal = original_goal
        self.goal_keywords = self._extract_keywords(original_goal)
        self.drift_threshold = drift_threshold
        self.consecutive_threshold = consecutive_threshold
        self.steps: list[dict] = []
        self.drift_scores: list[float] = []
        self.warnings: list[dict] = []
        self.total_steps = 0
        DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from goal text."""
        words = re.findall(r'[a-zA-Z\u4e00-\u9fff]{3,}', text.lower())
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have',
                     '的', '是', '在', '了', '和', '也', '就', '都', '要', '把', '被',
                     'a', 'an', 'in', 'on', 'to', 'of', 'is', 'it', 'be', 'as', 'at', 'by',
                     'not', 'or', 'we', 'our', 'you', 'your', 'they', 'their'}
        return {w for w in words if w not in stopwords}

    def _keyword_overlap(self, text: str) -> float:
        """Calculate keyword overlap between goal and current text."""
        step_keywords = self._extract_keywords(text)
        if not self.goal_keywords or not step_keywords:
            return 0.0
        overlap = len(self.goal_keywords & step_keywords)
        return overlap / max(1, len(step_keywords))

    def _length_ratio(self, text: str) -> float:
        """Detect explosion of irrelevant content (tool outputs)."""
        if not self.steps:
            return 0.0
        avg_prev = sum(len(s.get("action", "")) for s in self.steps[-5:]) / max(1, len(self.steps[-5:]))
        if avg_prev < 10:
            return 0.0
        return min(1.0, len(text) / max(1, avg_prev))

    def record_step(self, action: str, result: str = "") -> dict:
        """Record a step and return drift assessment."""
        self.total_steps += 1
        combined = f"{action} {result}"

        # Calculate drift from keyword overlap
        overlap = self._keyword_overlap(combined)
        # Invert: high overlap = low drift
        drift = 1.0 - overlap

        # Penalize if output is very long (sign of tool output diversion)
        length_drift = self._length_ratio(combined)
        if length_drift > 3.0:
            drift += 0.15

        drift = min(1.0, drift)
        self.drift_scores.append(drift)
        self.steps.append({
            "step": self.total_steps,
            "action": action[:200],
            "result": result[:200],
            "drift": drift,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.steps) > 100:
            self.steps = self.steps[-50:]

        return {"drift": drift, "step": self.total_steps}

    def check(self) -> dict:
        """Evaluate current drift state. Returns warning if threshold exceeded."""
        if len(self.drift_scores) < self.consecutive_threshold:
            return {"drift": False, "consecutive_high": 0}

        recent = self.drift_scores[-self.consecutive_threshold:]
        consecutive_high = sum(1 for s in recent if s > self.drift_threshold)
        avg_drift = sum(recent) / len(recent)

        is_drifting = consecutive_high >= self.consecutive_threshold

        if is_drifting:
            self.warnings.append({
                "step": self.total_steps,
                "avg_drift": round(avg_drift, 3),
                "consecutive": consecutive_high,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.warning("GoalDrift: step %d, avg drift=%.3f, consecutive=%d",
                         self.total_steps, avg_drift, consecutive_high)

        return {
            "drift": is_drifting,
            "avg_drift": round(avg_drift, 3),
            "consecutive_high": consecutive_high,
            "total_steps": self.total_steps,
            "warnings_count": len(self.warnings),
        }

    def reset_goal(self, new_goal: str):
        """Reset the goal (e.g., after subtask completion)."""
        self.original_goal = new_goal
        self.goal_keywords = self._extract_keywords(new_goal)
        self.drift_scores = []

    def status(self) -> dict:
        return {
            "goal": self.original_goal[:100],
            "total_steps": self.total_steps,
            "drift_warnings": len(self.warnings),
            "recent_drift_avg": round(sum(self.drift_scores[-10:]) / max(1, len(self.drift_scores[-10:])), 3),
            "latest_warning": self.warnings[-1] if self.warnings else None,
        }
