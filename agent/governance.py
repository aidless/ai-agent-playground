"""Agent 安全治理模块（方向三：安全治理）

三层治理体系：
    1. 执行审计 (AuditLog) — 记录每一次操作，可追溯
    2. 权限分级 (PermissionManager) — 工具按风险分级，高危需确认
    3. 熔断机制 (CircuitBreaker) — 连续失败自动熔断，防止级联故障
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

AUDIT_DIR = Path(__file__).resolve().parent.parent / "memory" / "audit_trails"


class PermissionLevel(str, Enum):
    READONLY = "readonly"
    RESTRICTED = "restricted"
    EXECUTE = "execute"
    ADMIN = "admin"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ── 1. 执行审计 ──────────────────────────────────

@dataclass
class AuditEntry:
    """单条审计记录"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    operation: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    result_summary: str = ""
    duration_ms: float = 0
    success: bool = True
    error: Optional[str] = None
    permission_level: str = "readonly"
    trace_id: str = ""


class AuditLogger:
    """执行审计日志 — 全量记录每次工具调用，自动清理>90天旧日志"""

    def __init__(self, retention_days: int = 90):
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

    def log(self, entry: AuditEntry):
        """记录一条审计日志"""
        today = datetime.now().strftime("%Y-%m-%d")
        path = AUDIT_DIR / f"audit-{today}.jsonl"

        record = {
            "ts": entry.timestamp,
            "operation": entry.operation,
            "tool": entry.tool,
            "args": {k: str(v)[:200] for k, v in entry.args.items()},
            "result": entry.result_summary[:200],
            "duration_ms": entry.duration_ms,
            "success": entry.success,
            "error": entry.error,
            "level": entry.permission_level,
            "trace_id": entry.trace_id,
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def query(self, date: Optional[str] = None, success_only: bool = False, limit: int = 50) -> list[dict]:
        """查询审计日志"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        path = AUDIT_DIR / f"audit-{date}.jsonl"
        if not path.exists():
            return []

        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if success_only and not record.get("success"):
                        continue
                    results.append(record)
                    if len(results) >= limit:
                        break
                except json.JSONDecodeError:
                    continue
        return results

    def purge_old(self):
        """删除超过 retention_days 天的审计日志文件"""
        cutoff = datetime.now().timestamp() - self.retention_days * 86400
        for f in AUDIT_DIR.glob("audit-*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.info("清理过期审计: %s", f.name)
            except OSError:
                pass

    def stats(self, date: Optional[str] = None) -> dict:
        """审计统计摘要"""
        self.purge_old()
        records = self.query(date=date, limit=10000)
        if not records:
            return {"total": 0}

        success = sum(1 for r in records if r.get("success"))
        by_level = {}
        for r in records:
            lvl = r.get("level", "unknown")
            by_level[lvl] = by_level.get(lvl, 0) + 1

        return {
            "total": len(records),
            "success_rate": success / len(records) if records else 0,
            "by_level": by_level,
            "avg_duration_ms": sum(r.get("duration_ms", 0) for r in records) / len(records) if records else 0,
        }


# ── 2. 权限分级 ──────────────────────────────────

class PermissionManager:
    """工具权限管理器"""

    DEFAULT_POLICY: dict[str, PermissionLevel] = {
        "read_file": PermissionLevel.READONLY,
        "list_files": PermissionLevel.READONLY,
        "web_search": PermissionLevel.READONLY,
        "web_fetch": PermissionLevel.READONLY,
        "calculator": PermissionLevel.READONLY,
        "write_file": PermissionLevel.RESTRICTED,
        "run_python": PermissionLevel.EXECUTE,
        "run_command": PermissionLevel.EXECUTE,
        "edit_file": PermissionLevel.RESTRICTED,
        "delete_file": PermissionLevel.ADMIN,
    }

    def __init__(self, policy: Optional[dict] = None):
        self.policy = policy or dict(self.DEFAULT_POLICY)

    def get_level(self, tool_name: str) -> PermissionLevel:
        return self.policy.get(tool_name, PermissionLevel.READONLY)

    def allow(self, tool_name: str, required_level: PermissionLevel) -> bool:
        order = {
            PermissionLevel.READONLY: 0,
            PermissionLevel.RESTRICTED: 1,
            PermissionLevel.EXECUTE: 2,
            PermissionLevel.ADMIN: 3,
        }
        return order.get(self.get_level(tool_name), 0) >= order.get(required_level, 0)

    def requires_confirmation(self, tool_name: str) -> bool:
        return self.get_level(tool_name) in (PermissionLevel.EXECUTE, PermissionLevel.ADMIN)

    def grant(self, tool_name: str, level: PermissionLevel):
        self.policy[tool_name] = level
        logger.warning("权限变更: %s → %s", tool_name, level.value)

    def revoke(self, tool_name: str):
        if tool_name in self.DEFAULT_POLICY:
            self.policy[tool_name] = self.DEFAULT_POLICY[tool_name]
        else:
            self.policy.pop(tool_name, None)

    def report(self) -> str:
        lines = ["权限分级报告:"]
        for tool, level in sorted(self.policy.items()):
            risk = "R" if level in (PermissionLevel.EXECUTE, PermissionLevel.ADMIN) else "Y" if level == PermissionLevel.RESTRICTED else "G"
            lines.append(f"  [{risk}] {tool}: {level.value}")
        return "\n".join(lines)


# ── 3. 熔断机制 ──────────────────────────────────

class CircuitBreaker:
    """工具调用熔断器

    状态机: CLOSED → (N次失败) → OPEN → (timeout) → HALF_OPEN → (成功) → CLOSED
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: float = 60.0, half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_max = half_open_max_calls
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0
        self.half_open_calls = 0
        self.last_error: Optional[str] = None

    def before_call(self) -> bool:
        """调用前检查"""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("熔断器进入 HALF_OPEN（试探恢复）")
                return True
            logger.warning("熔断器 OPEN，拒绝调用（距上次失败 %.1fs）", elapsed)
            return False
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max:
                self.half_open_calls += 1
                return True
            return False
        return True

    def on_success(self):
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("熔断器恢复 → CLOSED")

    def on_failure(self, error: str = ""):
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.last_error = error
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.error("熔断器 HALF_OPEN 试探失败 → OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.error("熔断器触发 OPEN: %d 次连续失败 (threshold=%d)", self.failure_count, self.failure_threshold)

    def status(self) -> dict:
        return {
            "state": self.state.value,
            "failures": self.failure_count,
            "successes": self.success_count,
            "threshold": self.failure_threshold,
            "last_error": self.last_error,
        }


# ── 治理门面 ─────────────────────────────────────

# ── 4. CISO 审批门 ──────────────────────────────────

class CISOApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class CISOApprovalRequest:
    request_id: str
    tool_name: str
    risk_level: str
    requester: str
    justification: str
    state: str = "pending"
    approver: str = ""
    approved_at: str = ""
    expires_at: str = ""
    denial_reason: str = ""


class CISOApprovalGate:
    """CISO 审批门 - 高风险操作需正式安全审批"""

    def __init__(self):
        self.store = AUDIT_DIR / "ciso_approvals.jsonl"
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self._pending: dict[str, CISOApprovalRequest] = {}
        import uuid

    def request_approval(self, tool_name: str, risk_level: str, requester: str, justification: str) -> CISOApprovalRequest:
        import uuid
        req = CISOApprovalRequest(
            request_id=f"ciso-{uuid.uuid4().hex[:8]}",
            tool_name=tool_name, risk_level=risk_level,
            requester=requester, justification=justification,
        )
        self._pending[req.request_id] = req
        self._save(req)
        return req

    def approve(self, request_id: str, approver: str, ttl_hours: int = 24) -> bool:
        req = self._pending.get(request_id)
        if not req:
            return False
        req.state = "approved"
        req.approver = approver
        req.approved_at = datetime.now().isoformat()
        req.expires_at = (datetime.now() + (ttl_hours * 3600)).isoformat()
        self._save(req)
        del self._pending[request_id]
        return True

    def deny(self, request_id: str, approver: str, reason: str) -> bool:
        req = self._pending.get(request_id)
        if not req:
            return False
        req.state = "denied"
        req.approver = approver
        req.denial_reason = reason
        self._save(req)
        del self._pending[request_id]
        return True

    def pending_requests(self) -> list:
        return list(self._pending.values())

    def _save(self, req: CISOApprovalRequest):
        with open(self.store, "a", encoding="utf-8") as f:
            f.write(json.dumps(req.__dict__, ensure_ascii=False) + "\n")


# ── 5. SLA/SLO 监控 ──────────────────────────────────

DEFAULT_SLO = {
    "latency_p95_ms": 2000,
    "latency_p99_ms": 5000,
    "success_rate": 0.95,
    "error_budget_pct": 0.05,
}


class SLOMonitor:
    """SLA/SLO 合规监控器 — 追踪延迟、成功率、错误预算"""

    def __init__(self, slo: dict | None = None):
        self.slo = slo or DEFAULT_SLO
        self._records: list[dict] = []
        self._breach_history: list[dict] = []

    def record(self, tool_name: str, latency_ms: float, success: bool):
        self._records.append({
            "ts": time.time(), "tool": tool_name,
            "latency_ms": latency_ms, "success": success,
        })
        if len(self._records) > 10000:
            self._records = self._records[-5000:]

        if len(self._records) >= 10:
            recent = [r for r in self._records[-100:] if r["tool"] == tool_name]
            if recent:
                sr = sum(1 for r in recent if r["success"]) / len(recent)
                if sr < self.slo["success_rate"]:
                    self._breach_history.append({
                        "ts": time.time(), "tool": tool_name,
                        "metric": "success_rate", "actual": sr,
                        "threshold": self.slo["success_rate"],
                    })

    def get_compliance_report(self) -> dict:
        if not self._records:
            return {"total_calls": 0, "success_rate": 0, "avg_latency_ms": 0, "p95_latency_ms": 0, "p99_latency_ms": 0, "slo_compliant": True, "error_budget_consumed_pct": 0, "total_breaches": 0}

        total = len(self._records)
        success = sum(1 for r in self._records if r["success"])
        latencies = sorted(r["latency_ms"] for r in self._records)
        n = len(latencies)

        return {
            "total_calls": total,
            "success_rate": round(success / total, 4) if total else 0,
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "p95_latency_ms": latencies[int(n * 0.95)],
            "p99_latency_ms": latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
            "slo_compliant": (success / total) >= self.slo["success_rate"],
            "error_budget_consumed_pct": round(max(0, (1 - success / total) * 100), 2) if total else 0,
            "total_breaches": len(self._breach_history),
        }

    def error_budget_status(self) -> dict:
        report = self.get_compliance_report()
        return {
            "success_rate": report["success_rate"],
            "target": self.slo["success_rate"],
            "budget_consumed_pct": report["error_budget_consumed_pct"],
            "status": "healthy" if report["slo_compliant"] else "breach",
        }


class GovernancePanel:
    """治理总控面板 — 审计 + 权限 + 熔断 + CISO + SLO"""

    def __init__(self):
        self.audit = AuditLogger()
        self.permission = PermissionManager()
        self.breakers: dict[str, CircuitBreaker] = {}
        self.ciso = CISOApprovalGate()
        self.slo = SLOMonitor()

    def get_breaker(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self.breakers:
            self.breakers[tool_name] = CircuitBreaker()
        return self.breakers[tool_name]

    async def wrap_tool_call(self, tool_name: str, args: dict, func, trace_id: str = "") -> Any:
        if self.permission.requires_confirmation(tool_name):
            logger.info("高危操作 [%s] 需要确认: %s", tool_name, args)

        breaker = self.get_breaker(tool_name)
        if not breaker.before_call():
            raise RuntimeError(f"熔断器已断开 [{tool_name}]: {breaker.last_error}")

        start = time.time()
        try:
            result = func()
            duration = (time.time() - start) * 1000
            breaker.on_success()
            self.audit.log(AuditEntry(
                operation="tool_call", tool=tool_name, args=args,
                result_summary=str(result)[:200], duration_ms=duration,
                success=True, permission_level=self.permission.get_level(tool_name).value,
                trace_id=trace_id,
            ))
            self.slo.record(tool_name, duration, True)
            return result
        except Exception as e:
            duration = (time.time() - start) * 1000
            breaker.on_failure(str(e))
            self.audit.log(AuditEntry(
                operation="tool_call", tool=tool_name, args=args,
                duration_ms=duration, success=False, error=str(e),
                permission_level=self.permission.get_level(tool_name).value,
                trace_id=trace_id,
            ))
            self.slo.record(tool_name, duration, False)
            raise

    def report(self) -> str:
        lines = ["=" * 40, "安全治理报告", "=" * 40, "", self.permission.report(), "", "熔断器状态:"]
        for tool, breaker in self.breakers.items():
            s = breaker.status()
            lines.append(f"  {tool}: {s['state']} (fail={s['failures']}, ok={s['successes']})")
        lines.append("")
        stats = self.audit.stats()
        if stats["total"] > 0:
            lines.append(f"今日审计: {stats['total']} 条, 成功率 {stats['success_rate']:.1%}")
        lines.append("")
        slo_report = self.slo.get_compliance_report()
        lines.append(f"SLO: 成功率 {slo_report.get('success_rate', 0):.2%} | P95 {slo_report.get('p95_latency_ms', 0):.0f}ms | 违规 {slo_report.get('total_breaches', 0)}")
        lines.append(f"CISO待审批: {len(self.ciso.pending_requests())}")
        return "\n".join(lines)
