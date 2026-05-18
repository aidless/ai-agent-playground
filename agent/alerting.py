"""Alerting & Health Monitoring — threshold-based alert rules.

Alert rules evaluate metrics and trigger notifications when thresholds are breached.
Supports multiple channels: console, file, webhook (Slack/WeChat/Feishu compatible).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertState(str, Enum):
    OK = "ok"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    name: str
    metric: str
    operator: str  # ">", "<", ">=", "<=", "=="
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    cooldown_seconds: float = 300  # Don't re-fire within this window
    enabled: bool = True

    def evaluate(self, value: float) -> bool:
        ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
        }
        return ops.get(self.operator, lambda a, b: False)(value, self.threshold)


@dataclass
class Alert:
    rule: AlertRule
    state: AlertState = AlertState.OK
    current_value: float = 0.0
    last_fired_at: float = 0.0
    resolved_at: float = 0.0
    fire_count: int = 0
    message: str = ""


DEFAULT_RULES = [
    AlertRule("error_rate_high", "error_rate", ">", 0.10, AlertSeverity.CRITICAL,
              "Error rate exceeds 10%", cooldown_seconds=120),
    AlertRule("latency_p95_high", "p95_latency_ms", ">", 2000, AlertSeverity.WARNING,
              "P95 latency exceeds 2000ms", cooldown_seconds=300),
    AlertRule("latency_p99_high", "p99_latency_ms", ">", 5000, AlertSeverity.CRITICAL,
              "P99 latency exceeds 5000ms", cooldown_seconds=120),
    AlertRule("per_task_cost_high", "cost_per_task", ">", 0.20, AlertSeverity.WARNING,
              "Per-task cost exceeds $0.20", cooldown_seconds=600),
    AlertRule("success_rate_low", "success_rate", "<", 0.90, AlertSeverity.CRITICAL,
              "Success rate below 90%", cooldown_seconds=60),
    AlertRule("cost_anomaly", "cost_per_hour", ">", 5.0, AlertSeverity.WARNING,
              "Hourly cost exceeds $5.00", cooldown_seconds=3600),
    AlertRule("circuit_open", "circuit_breaker_open", ">", 0, AlertSeverity.CRITICAL,
              "Circuit breaker is OPEN", cooldown_seconds=30),
    AlertRule("audit_failure_rate", "audit_failure_rate", ">", 0.05, AlertSeverity.WARNING,
              "Audit trail failure rate > 5%", cooldown_seconds=600),
    # Security intrusion rules
    AlertRule("intrusion_auth_brute_force", "intrusion_auth_failures", ">", 10, AlertSeverity.CRITICAL,
              "Brute force attack detected: >10 auth failures/min", cooldown_seconds=60),
    AlertRule("intrusion_path_traversal", "intrusion_path_denials", ">", 5, AlertSeverity.HIGH,
              "Path traversal attack detected: >5 denied paths/min", cooldown_seconds=60),
    AlertRule("intrusion_injection_surge", "intrusion_injections", ">", 5, AlertSeverity.HIGH,
              "Prompt injection surge: >5 blocked attempts/min", cooldown_seconds=60),
    AlertRule("intrusion_active_threats_high", "intrusion_active_threats", ">", 0, AlertSeverity.CRITICAL,
              "Active security threats detected", cooldown_seconds=30),
]


class AlertManager:
    """Evaluates alert rules and manages alert lifecycle."""

    def __init__(self, rules: list[AlertRule] | None = None):
        self.rules = rules or DEFAULT_RULES
        self._alerts: dict[str, Alert] = {r.name: Alert(rule=r) for r in self.rules}
        self._history: list[dict] = []
        self._notifiers: list[Callable] = []

    def add_notifier(self, notifier: Callable[[Alert], None]):
        """Add a notification channel. notifier(alert) -> None."""
        self._notifiers.append(notifier)

    def evaluate(self, metrics: dict[str, float]) -> list[Alert]:
        """Evaluate all rules against current metrics. Returns firing alerts."""
        firing = []
        now = time.time()

        for rule in self.rules:
            if not rule.enabled:
                continue
            if rule.metric not in metrics:
                continue

            alert = self._alerts[rule.name]
            value = metrics[rule.metric]

            if rule.evaluate(value):
                alert.current_value = value
                if alert.state != AlertState.FIRING:
                    if now - alert.last_fired_at > rule.cooldown_seconds:
                        alert.state = AlertState.FIRING
                        alert.last_fired_at = now
                        alert.fire_count += 1
                        alert.message = (
                            f"[{rule.severity.value.upper()}] {rule.name}: "
                            f"{rule.metric}={value} {rule.operator} {rule.threshold}"
                        )
                        self._history.append({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "rule": rule.name,
                            "state": "firing",
                            "value": value,
                            "threshold": rule.threshold,
                        })
                        for notifier in self._notifiers:
                            notifier(alert)
                        firing.append(alert)
            else:
                if alert.state == AlertState.FIRING:
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = now
                    self._history.append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "rule": rule.name,
                        "state": "resolved",
                        "value": value,
                    })

        return firing

    def status(self) -> dict:
        """Return current alert status for all rules."""
        return {
            name: {
                "state": alert.state.value,
                "current_value": alert.current_value,
                "threshold": alert.rule.threshold,
                "severity": alert.rule.severity.value,
                "fire_count": alert.fire_count,
            }
            for name, alert in self._alerts.items()
        }

    def recent_history(self, limit: int = 50) -> list[dict]:
        return self._history[-limit:]

    def get_firing(self) -> list[Alert]:
        return [a for a in self._alerts.values() if a.state == AlertState.FIRING]


class HealthChecker:
    """Aggregated health check across all subsystems."""

    def __init__(self):
        self._checks: dict[str, Callable[[], dict]] = {}
        self.register("basic", self._check_basic)

    def register(self, name: str, check_fn: Callable[[], dict]):
        self._checks[name] = check_fn

    def _check_basic(self) -> dict:
        return {
            "status": "ok",
            "uptime_seconds": time.time(),
            "python_version": __import__("sys").version,
        }

    def run_all(self) -> dict:
        """Run all health checks and return aggregated status."""
        results = {}
        all_ok = True

        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                results[name] = result
                if result.get("status") != "ok":
                    all_ok = False
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                all_ok = False

        return {
            "status": "ok" if all_ok else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results,
            "total": len(results),
            "healthy": sum(1 for r in results.values() if r.get("status") == "ok"),
        }
