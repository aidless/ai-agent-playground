"""Reflect→Action Closed Loop — reflection drives behavioral change.

When REFLECT detects failures or inefficiencies, this module translates
reflection text into concrete actions:

  1. Tool Degradation — if a tool fails N consecutive times, auto-degrade
  2. Tool Replacement — suggest alternative tools for the same intent
  3. Strategy Pivot — change approach when stuck in a loop
  4. Capability Gap Detection — identify missing tools for bootstrapping

This is the "Meta's Hyperagents" style self-improvement loop applied to
an enterprise agent framework.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DegradationEntry:
    tool_name: str
    consecutive_failures: int
    alternatives: list[str]
    degraded_at: str
    reason: str


class ReflectActionEngine:
    """Translates reflection insights into concrete tool/strategy changes.

    Usage:
        engine = ReflectActionEngine()
        engine.record_tool_result("web_search", success=False, error="timeout")
        actions = engine.evaluate(reflection_text, tool_results)
        # actions = [{"type": "degrade", "tool": "web_search", ...}, ...]
    """

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self._tool_failures: dict[str, int] = {}
        self._degraded: dict[str, DegradationEntry] = {}
        self._action_log: list[dict] = []

        # Tool alternatives map (tool_name -> [fallback tools])
        self.alternatives: dict[str, list[str]] = {
            "web_search": ["web_fetch"],
            "web_fetch": ["web_search"],
            "run_python": ["calculator"],
            "code_exec": ["run_python"],
            "run_command": ["run_python", "code_exec"],
            "write_file": ["edit_file"],
            "edit_file": ["write_file"],
        }

    def record_tool_result(self, tool_name: str, success: bool, error: str = ""):
        if success:
            self._tool_failures[tool_name] = 0
        else:
            self._tool_failures[tool_name] = self._tool_failures.get(tool_name, 0) + 1
            if self._tool_failures[tool_name] >= self.failure_threshold:
                self._degrade_tool(tool_name, error)

    def _degrade_tool(self, tool_name: str, reason: str):
        if tool_name in self._degraded:
            return
        failures = self._tool_failures.get(tool_name, self.failure_threshold)
        alternatives = self.alternatives.get(tool_name, [])
        from datetime import datetime, timezone
        entry = DegradationEntry(
            tool_name=tool_name,
            consecutive_failures=failures,
            alternatives=alternatives,
            degraded_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        self._degraded[tool_name] = entry
        self._action_log.append({
            "type": "degrade",
            "tool": tool_name,
            "alternatives": alternatives,
            "reason": reason,
        })
        logger.warning("Tool degraded: %s → alternatives: %s (reason: %s)",
                       tool_name, alternatives, reason)

    def is_degraded(self, tool_name: str) -> bool:
        return tool_name in self._degraded

    def get_alternatives(self, tool_name: str) -> list[str]:
        return self.alternatives.get(tool_name, [])

    def evaluate(self, reflection: str, tool_results: list[dict]) -> list[dict]:
        """Analyze reflection + tool results, return recommended actions.

        Returns list of action dicts:
          {"type": "degrade", "tool": "X", "alternatives": ["Y"], "reason": "..."}
          {"type": "pivot", "message": "..."}
          {"type": "missing_tool", "description": "...", "suggested_name": "..."}
        """
        actions = []

        # Record results from this cycle
        for tr in tool_results[-10:]:
            self.record_tool_result(
                tr.get("tool", "unknown"),
                tr.get("status") == "ok",
                str(tr.get("result", "")),
            )

        # Detect loops: same tool called repeatedly with same error
        recent_errors = [tr for tr in tool_results[-6:] if tr.get("status") == "error"]
        if len(recent_errors) >= 3:
            error_tools = [tr.get("tool") for tr in recent_errors]
            if len(set(error_tools)) == 1:
                actions.append({
                    "type": "pivot",
                    "message": f"Tool {error_tools[0]} failing repeatedly. Consider alternative approach.",
                    "stuck_tool": error_tools[0],
                })

        # Detect capability gap from reflection
        gap_keywords = ["don't have", "missing", "not available", "no tool", "can't find",
                        "没有", "缺失", "不存在", "找不到"]
        if any(kw in reflection for kw in gap_keywords):
            import re
            # Extract what's missing
            missing_match = re.search(
                r"(?:missing|need|don't have|no tool|没有|需要|缺失)[:\s]+([a-z_]+(?:\s*[a-z_]+)*)",
                reflection, re.IGNORECASE
            )
            suggested = missing_match.group(1).strip().replace(" ", "_") if missing_match else "unknown_tool"
            actions.append({
                "type": "missing_tool",
                "description": reflection,
                "suggested_name": suggested,
            })

        return actions

    def filter_degraded_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Remove degraded tools from tool calls, substituting alternatives."""
        filtered = []
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            if self.is_degraded(tool_name):
                alts = self.get_alternatives(tool_name)
                if alts:
                    tc_copy = dict(tc)
                    tc_copy["function"]["name"] = alts[0]
                    filtered.append(tc_copy)
                    logger.info("Substituted degraded tool %s → %s", tool_name, alts[0])
                else:
                    logger.warning("Dropped degraded tool %s (no alternatives)", tool_name)
            else:
                filtered.append(tc)
        return filtered

    def status(self) -> dict:
        return {
            "degraded_tools": list(self._degraded.keys()),
            "degradations": [
                {"tool": d.tool_name, "reason": d.reason, "alternatives": d.alternatives}
                for d in self._degraded.values()
            ],
            "recent_actions": self._action_log[-10:],
            "failure_counts": dict(self._tool_failures),
        }
