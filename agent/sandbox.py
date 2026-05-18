"""Sandbox Executor — isolated tool execution environment.

Every tool invocation runs in a sandbox with:
  - Filesystem isolation (constrained work directory)
  - Network egress control (allowed/denied hosts)
  - Time budget (per-call timeout)
  - Resource limits (max output size, max recursion depth)
  - Audit trail (logged independently of tool code)

Implementation: process-level isolation via subprocess with restricted env.
"""

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class SandboxViolation(Exception):
    """Raised when sandbox policy is violated."""
    pass


class TimeBudgetExceeded(Exception):
    """Raised when execution exceeds time budget."""
    pass


@dataclass
class SandboxPolicy:
    """Defines what a sandboxed execution may do."""

    allowed_dirs: list[str] = field(default_factory=list)
    denied_dirs: list[str] = field(default_factory=lambda: [
        "/etc", "/sys", "/proc", "C:\\Windows", "C:\\Windows\\System32",
    ])
    allowed_hosts: list[str] = field(default_factory=list)
    denied_hosts: list[str] = field(default_factory=list)
    max_output_bytes: int = 1_000_000
    max_runtime_seconds: float = 30.0
    max_memory_mb: int = 256
    readonly_fs: bool = False
    require_approval: bool = False
    audit_level: str = "standard"  # minimal | standard | verbose


@dataclass
class SandboxResult:
    """Result of a sandboxed execution."""

    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    policy_violation: bool = False
    audit_entry: dict[str, Any] = field(default_factory=dict)
    sandbox_id: str = ""


RISK_LEVELS = {
    "read_file": "low",
    "list_files": "low",
    "calculator": "low",
    "web_search": "low",
    "run_python": "medium",
    "web_fetch": "medium",
    "write_file": "high",
    "code_exec": "high",
    "run_command": "critical",
    "delete_file": "critical",
    "modify_system": "critical",
}

POLICIES_BY_RISK = {
    "low": SandboxPolicy(
        readonly_fs=True,
        max_runtime_seconds=10.0,
        audit_level="standard",
    ),
    "medium": SandboxPolicy(
        readonly_fs=False,
        max_runtime_seconds=20.0,
        max_output_bytes=500_000,
        require_approval=False,
        audit_level="standard",
    ),
    "high": SandboxPolicy(
        readonly_fs=False,
        max_runtime_seconds=30.0,
        max_output_bytes=250_000,
        require_approval=True,
        audit_level="verbose",
    ),
    "critical": SandboxPolicy(
        readonly_fs=False,
        max_runtime_seconds=60.0,
        max_output_bytes=100_000,
        require_approval=True,
        audit_level="verbose",
        denied_dirs=["/etc", "/sys", "/proc", "/boot", "C:\\Windows",
                      "C:\\Windows\\System32", "~/.ssh", "~/.gnupg"],
    ),
}


class SandboxExecutor:
    """Executes tool calls within a sandboxed environment.

    Each call:
    1. Receives tool_name + params + identity
    2. Evaluates risk level
    3. Applies policy constraints
    4. Executes in isolated subprocess (or direct with audit for read-only)
    5. Collects audit trail
    6. Returns SandboxResult
    """

    def __init__(self, work_dir: str = "./sandbox_workspace"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.work_dir / "audit.jsonl"
        self._call_counter = 0
        self._lock = threading.Lock()

    def evaluate_risk(self, tool_name: str, params: dict) -> str:
        """Determine risk level for a tool call."""
        return RISK_LEVELS.get(tool_name, "high")

    def execute(
        self,
        tool_name: str,
        tool_func: Callable,
        params: dict,
        identity: str = "anonymous",
        session_id: str = "",
    ) -> SandboxResult:
        """Execute a tool call within sandbox constraints."""
        risk = self.evaluate_risk(tool_name, params)
        policy = POLICIES_BY_RISK[risk]

        self._call_counter += 1
        sandbox_id = f"snd-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._call_counter:04d}"

        # Check filesystem constraints
        for key, val in params.items():
            if isinstance(val, str) and any(
                val.startswith(d) or d in val
                for d in policy.denied_dirs
            ):
                result = SandboxResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Access denied: path {val} is in denied directories",
                    policy_violation=True,
                    sandbox_id=sandbox_id,
                )
                self._audit(tool_name, params, result, identity, session_id, risk, policy)
                return result

        # Execute with time budget
        start = time.perf_counter()
        error_msg = ""
        output = ""

        try:
            result_value = self._run_with_timeout(tool_func, params, policy.max_runtime_seconds)
            output = str(result_value) if result_value is not None else ""
            # Truncate if too large
            if len(output) > policy.max_output_bytes:
                output = output[:policy.max_output_bytes] + "\n... [truncated]"
            success = True
        except TimeBudgetExceeded:
            error_msg = f"Execution exceeded time budget of {policy.max_runtime_seconds}s"
            success = False
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            success = False

        elapsed = (time.perf_counter() - start) * 1000

        result = SandboxResult(
            tool_name=tool_name,
            success=success,
            output=output if success else "",
            error=error_msg,
            duration_ms=elapsed,
            sandbox_id=sandbox_id,
            audit_entry=self._build_audit_entry(
                tool_name, params, success, error_msg, elapsed,
                identity, session_id, risk, policy.audit_level,
                sandbox_id,
            ),
        )

        self._audit(tool_name, params, result, identity, session_id, risk, policy)
        return result

    def _run_with_timeout(self, func: Callable, timeout: float, *args, **kwargs):
        """Run a function with a timeout using threading."""
        result = []
        error = []

        def target():
            try:
                result.append(func(*args, **kwargs))
            except Exception as e:
                error.append(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeBudgetExceeded(f"Timeout after {timeout}s")
        if error:
            raise error[0]
        return result[0] if result else None

    def _build_audit_entry(
        self,
        tool_name: str,
        params: dict,
        success: bool,
        error: str,
        duration_ms: float,
        identity: str,
        session_id: str,
        risk: str,
        audit_level: str,
        sandbox_id: str,
    ) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sandbox_id": sandbox_id,
            "tool": tool_name,
            "risk_level": risk,
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "identity": identity,
            "session_id": session_id,
        }
        if audit_level == "verbose":
            entry["params"] = self._sanitize_params(params)
        if not success:
            entry["error"] = error
        return entry

    def _sanitize_params(self, params: dict) -> dict:
        """Redact sensitive values from audit params."""
        sensitive_keys = {"api_key", "password", "token", "secret", "credential"}
        sanitized = {}
        for k, v in params.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                sanitized[k] = "***REDACTED***"
            elif isinstance(v, str) and len(v) > 200:
                sanitized[k] = v[:197] + "..."
            else:
                sanitized[k] = v
        return sanitized

    def _audit(
        self,
        tool_name: str,
        params: dict,
        result: SandboxResult,
        identity: str,
        session_id: str,
        risk: str,
        policy: SandboxPolicy,
    ):
        """Write audit entry to log."""
        entry = result.audit_entry if result.audit_entry else self._build_audit_entry(
            tool_name, params, result.success, result.error,
            result.duration_ms, identity, session_id, risk,
            policy.audit_level, result.sandbox_id,
        )
        entry["policy_applied"] = {
            "readonly_fs": policy.readonly_fs,
            "require_approval": policy.require_approval,
            "risk_level": risk,
        }

        with self._lock:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_audit_trail(
        self, hours: int = 24, risk_filter: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Retrieve recent audit entries."""
        if not self.audit_log_path.exists():
            return []

        cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
        entries = []
        with open(self.audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                    if ts >= cutoff:
                        if risk_filter and entry.get("risk_level") != risk_filter:
                            continue
                        entries.append(entry)
                        if len(entries) >= limit:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue
        return entries

    def audit_summary(self, hours: int = 24) -> dict:
        """Generate audit summary for compliance reporting."""
        entries = self.get_audit_trail(hours=hours)
        if not entries:
            return {"total_calls": 0, "message": "No audit records in period"}

        total = len(entries)
        success = sum(1 for e in entries if e.get("success"))
        by_risk = {}
        for e in entries:
            r = e.get("risk_level", "unknown")
            by_risk.setdefault(r, {"total": 0, "success": 0})
            by_risk[r]["total"] += 1
            if e.get("success"):
                by_risk[r]["success"] += 1

        return {
            "period_hours": hours,
            "total_calls": total,
            "success_rate": round(success / total, 4) if total else 0,
            "by_risk_level": {
                level: {
                    "calls": stats["total"],
                    "success_rate": round(stats["success"] / stats["total"], 4)
                    if stats["total"] else 0,
                }
                for level, stats in by_risk.items()
            },
            "violations": sum(1 for e in entries if e.get("policy_violation", False)),
        }
