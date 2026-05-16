"""
Human-in-the-Loop — safety brakes for autonomous agents.

Agents are powerful but amoral — a tool-calling agent will happily:
  - Delete your production database
  - Send an email to your entire customer list
  - Run `rm -rf /` in a shared environment

The solution: risk-graded approval gates. Before executing any tool call,
the harness checks its risk level and pauses for human confirmation if needed.

Risk levels (per Anthropic's "human in the loop" engineering guidance):
  - LOW:     read-only, no side effects (read_file, web_search, calculator)
  - MEDIUM:  write to local filesystem (write_file, sandbox_execute)
  - HIGH:    execute commands, network calls (run_command)
  - CRITICAL: destructive operations (delete files, drop tables, send emails)

Usage:
    from ai_agent_playground.human_loop import ApprovalGate

    gate = ApprovalGate()
    gate.set_policy("run_command", Policy.ALWAYS_ASK)

    # Before executing a tool
    if not gate.approve("run_command", {"command": "rm -rf ./tmp"}):
        print("Approval denied!")
        return

    # After execution
    gate.post_check("run_command", result="Deleted 5 files", duration_ms=120)
"""

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# ============================================================
#  Policy
# ============================================================


class Policy(str, enum.Enum):
    AUTO_APPROVE = "auto_approve"  # Always allow (for safe tools)
    ALWAYS_ASK = "always_ask"  # Always ask human (for dangerous tools)
    NEVER = "never"  # Block entirely (for forbidden operations)


@dataclass
class ApprovalRequest:
    """A request for tool execution approval."""

    id: str
    tool_name: str
    arguments: dict
    risk_level: str  # low / medium / high / critical
    timestamp: str
    reason: str = ""  # Why this needs approval


@dataclass
class ApprovalDecision:
    approved: bool
    reason: str
    approved_by: str = "auto"  # "auto" | "human" | "policy"
    timestamp: str = ""


# ============================================================
#  Tool risk registry
# ============================================================


# Default risk levels for common agent tools
DEFAULT_RISK_LEVELS: dict[str, str] = {
    # LOW: read-only, no side effects
    "read_file": "low",
    "web_search": "low",
    "calculator": "low",

    # MEDIUM: local writes, sandboxed execution
    "write_file": "medium",
    "sandbox_execute": "medium",

    # HIGH: shell commands, network
    "run_command": "high",

    # CRITICAL: destructive operations
    "delete_file": "critical",
    "run_sql": "critical",
    "send_email": "critical",
    "http_post": "critical",
}

# Words in command strings that escalate risk
DANGEROUS_COMMAND_PATTERNS = [
    "rm", "rmdir", "del", "format",
    "DROP", "DELETE", "TRUNCATE", "ALTER",
    "shutdown", "reboot", "init 0", "init 6",
    "chmod 777", "chown",
    ">", "/dev/null",
    "wget", "curl", "nc ",
    "sudo", "su ",
]


def assess_risk(tool_name: str, arguments: dict) -> str:
    """Assess the risk level of a tool call.

    For commands, scans argument strings for dangerous patterns.
    """
    base_risk = DEFAULT_RISK_LEVELS.get(tool_name, "medium")

    # For run_command, scan the command string for danger patterns
    if tool_name == "run_command" and "command" in arguments:
        cmd = arguments["command"].lower()
        for pattern in DANGEROUS_COMMAND_PATTERNS:
            if pattern.lower() in cmd:
                return "critical"

    # For write_file, check if path is sensitive
    if tool_name == "write_file" and "path" in arguments:
        path = arguments["path"].lower()
        for sensitive in ["/etc/", "/boot/", "c:/windows", ".env", ".ssh", "id_rsa"]:
            if sensitive.lower() in path:
                return "critical"

    return base_risk


# ============================================================
#  Approval Gate
# ============================================================


@dataclass
class ApprovalLogEntry:
    tool_name: str
    risk_level: str
    arguments: dict
    approved: bool
    result_summary: str = ""
    duration_ms: float = 0
    timestamp: str = ""


class ApprovalGate:
    """Risk-graded approval system for agent tool calls.

    The harness sits between the agent's decision and the tool's execution.
    Every tool call passes through this gate:

        Agent decides → ApprovalGate.check() → Execute tool → ApprovalGate.log()
    """

    def __init__(self):
        self._policies: dict[str, Policy] = {}
        self._logs: list[ApprovalLogEntry] = []
        self._on_approve: list[Callable] = []
        self._on_deny: list[Callable] = []
        self._approval_counter: int = 0

    # ============================================================
    #  Policy configuration
    # ============================================================

    def set_policy(self, tool_name: str, policy: Policy):
        """Set approval policy for a specific tool."""
        self._policies[tool_name] = policy

    def set_policies(self, policies: dict[str, Policy]):
        """Batch-set policies for multiple tools."""
        self._policies.update(policies)

    def get_policy(self, tool_name: str) -> Policy:
        """Get the effective policy for a tool."""
        if tool_name in self._policies:
            return self._policies[tool_name]

        # Default policies based on risk level
        risk = DEFAULT_RISK_LEVELS.get(tool_name, "medium")
        if risk == "low":
            return Policy.AUTO_APPROVE
        elif risk in ("medium", "high"):
            return Policy.ALWAYS_ASK
        else:
            return Policy.NEVER  # critical → blocked by default

    # ============================================================
    #  Approval check
    # ============================================================

    def approve(
        self,
        tool_name: str,
        arguments: dict,
        input_fn: Callable | None = None,
    ) -> ApprovalDecision:
        """Check if a tool call is approved.

        Args:
            tool_name: name of the tool being called
            arguments: arguments to the tool
            input_fn: optional custom input function (for testing / custom UI).
                     Defaults to built-in input().

        Returns ApprovalDecision.
        """
        import uuid

        self._approval_counter += 1
        risk = assess_risk(tool_name, arguments)
        policy = self.get_policy(tool_name)

        req = ApprovalRequest(
            id=f"req_{self._approval_counter}",
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=f"Risk: {risk}, Policy: {policy.value}",
        )

        if policy == Policy.AUTO_APPROVE:
            return ApprovalDecision(
                approved=True,
                reason=f"Auto-approved (risk={risk})",
                approved_by="auto",
                timestamp=req.timestamp,
            )

        if policy == Policy.NEVER:
            self._handle_denied(tool_name, arguments, risk)
            return ApprovalDecision(
                approved=False,
                reason=f"Blocked by policy (risk={risk})",
                approved_by="policy",
                timestamp=req.timestamp,
            )

        # Policy.ALWAYS_ASK — prompt human
        if policy == Policy.ALWAYS_ASK:
            approved = self._prompt_human(req, input_fn)
            if approved:
                self._handle_approved(tool_name, arguments, risk)
                return ApprovalDecision(
                    approved=True,
                    reason="Human approved",
                    approved_by="human",
                    timestamp=req.timestamp,
                )
            else:
                self._handle_denied(tool_name, arguments, risk)
                return ApprovalDecision(
                    approved=False,
                    reason="Human denied",
                    approved_by="human",
                    timestamp=req.timestamp,
                )

        # Fallback
        return ApprovalDecision(approved=False, reason="Unknown policy", approved_by="policy")

    def _prompt_human(
        self, req: ApprovalRequest, input_fn: Callable | None = None
    ) -> bool:
        """Ask the human for approval via console (or custom input_fn)."""
        print(f"\n{'─' * 50}")
        print(f"  ⚠️  APPROVAL REQUIRED [Risk: {req.risk_level.upper()}]")
        print(f"  Tool:   {req.tool_name}")
        for k, v in req.arguments.items():
            preview = str(v)[:120].replace("\n", " ")
            print(f"  Arg:    {k} = {preview}")
        print(f"{'─' * 50}")

        fn = input_fn or input
        response = fn("  Approve? [y/N]: ").strip().lower()
        return response in ("y", "yes", "approve", "ok")

    # ============================================================
    #  Post-execution logging
    # ============================================================

    def log(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        approved: bool,
        duration_ms: float = 0,
    ):
        """Log a tool execution result."""
        entry = ApprovalLogEntry(
            tool_name=tool_name,
            risk_level=assess_risk(tool_name, arguments),
            arguments=arguments,
            approved=approved,
            result_summary=str(result)[:300],
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._logs.append(entry)

    # ============================================================
    #  Lifecycle hooks
    # ============================================================

    def on_approve(self, callback: Callable):
        """Register a callback invoked after approval."""
        self._on_approve.append(callback)

    def on_deny(self, callback: Callable):
        """Register a callback invoked after denial."""
        self._on_deny.append(callback)

    def _handle_approved(self, tool_name: str, args: dict, risk: str):
        for cb in self._on_approve:
            cb(tool_name, args, risk)

    def _handle_denied(self, tool_name: str, args: dict, risk: str):
        for cb in self._on_deny:
            cb(tool_name, args, risk)

    # ============================================================
    #  Reporting
    # ============================================================

    def report(self) -> dict:
        """Generate an approval audit report."""
        by_risk = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        by_tool: dict[str, int] = {}
        denied = 0

        for entry in self._logs:
            by_risk[entry.risk_level] = by_risk.get(entry.risk_level, 0) + 1
            by_tool[entry.tool_name] = by_tool.get(entry.tool_name, 0) + 1
            if not entry.approved:
                denied += 1

        return {
            "total_approvals": len(self._logs),
            "denied": denied,
            "approval_rate": (len(self._logs) - denied) / len(self._logs) if self._logs else 1.0,
            "by_risk": by_risk,
            "by_tool": by_tool,
        }

    def print_report(self):
        """Print a human-readable approval audit."""
        r = self.report()
        print(f"\n  Approval Gate Report: {r['total_approvals']} checks, "
              f"{r['denied']} denied ({r['approval_rate']:.0%} approved)")
        print("  By risk level:")
        for level, count in r["by_risk"].items():
            if count:
                print(f"    {level}: {count}")
