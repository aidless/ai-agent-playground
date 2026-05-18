"""Intrusion Detection System — anomaly-based threat monitoring.

Detects:
  - Auth brute force (failed validations exceeding baseline)
  - Unusual tool usage patterns (tool calls outside normal distribution)
  - Path traversal attempts (access to denied directories)
  - Prompt injection surges
  - Tenant hopping (rapid tenant switching suggesting header forgery)

Integrates with AlertManager for real-time notifications.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class IntrusionEvent:
    timestamp: str
    event_type: str      # "auth_brute_force", "unusual_tool", "path_traversal", "tenant_hop", "prompt_injection"
    severity: str        # "low", "medium", "high", "critical"
    source_ip: str
    details: dict
    score: float         # anomaly score 0.0-1.0


class IntrusionDetector:
    """Tracks patterns and raises alerts on anomalies."""

    def __init__(self):
        # Auth tracking: ip -> [timestamp, ...]
        self._auth_failures: dict[str, list[float]] = defaultdict(list)
        self._auth_successes: dict[str, list[float]] = defaultdict(list)

        # Tool usage: tool_name -> count per window
        self._tool_usage: dict[str, list[float]] = defaultdict(list)
        self._tool_baseline: dict[str, float] = {}  # tool -> expected calls per minute

        # Path denial: ip -> [timestamp, ...]
        self._path_denials: dict[str, list[float]] = defaultdict(list)

        # Tenant tracking: ip -> [(tenant_id, timestamp), ...]
        self._tenant_access: dict[str, list[tuple[str, float]]] = defaultdict(list)

        # Prompt injection: ip -> [timestamp, ...]
        self._injection_attempts: dict[str, list[float]] = defaultdict(list)

        # Event log
        self._events: list[IntrusionEvent] = []
        self._alert_callbacks: list = []

        # Thresholds
        self.auth_brute_threshold = 10     # failed auth per minute
        self.auth_spike_threshold = 20     # total auth attempts per minute
        self.path_denial_threshold = 5     # denied paths per minute
        self.tenant_hop_threshold = 3      # different tenants per minute
        self.injection_threshold = 5       # injection attempts per minute
        self.tool_anomaly_threshold = 3.0  # stddev multiplier for tool anomaly
        self.score_critical = 0.8
        self.score_high = 0.6

    def on_notify(self, callback):
        """Register callback(IntrusionEvent) for alerting."""
        self._alert_callbacks.append(callback)

    def _prune(self, ts_list: list[float], window: float = 60.0):
        """Remove timestamps older than window seconds."""
        now = time.time()
        ts_list[:] = [t for t in ts_list if now - t < window]

    def record_auth_failure(self, client_ip: str):
        now = time.time()
        self._auth_failures[client_ip].append(now)
        self._prune(self._auth_failures[client_ip])
        self._check_auth_brute_force(client_ip)

    def record_auth_success(self, client_ip: str):
        now = time.time()
        self._auth_successes[client_ip].append(now)
        self._prune(self._auth_successes[client_ip])

    def record_tool_call(self, tool_name: str):
        now = time.time()
        self._tool_usage[tool_name].append(now)
        self._prune(self._tool_usage[tool_name], window=300)
        self._check_tool_anomaly(tool_name)

    def record_path_denial(self, client_ip: str, path: str):
        now = time.time()
        self._path_denials[client_ip].append(now)
        self._prune(self._path_denials[client_ip])
        count = len(self._path_denials[client_ip])
        if count >= self.path_denial_threshold:
            self._emit("path_traversal", "high", client_ip, {
                "denial_count": count,
                "window_seconds": 60,
                "latest_path": path,
            }, min(1.0, count / (self.path_denial_threshold * 2)))

    def record_tenant_access(self, client_ip: str, tenant_id: str):
        now = time.time()
        self._tenant_access[client_ip].append((tenant_id, now))
        self._prune_tenant_entries(client_ip)
        self._check_tenant_hop(client_ip)

    def record_injection_attempt(self, client_ip: str, pattern: str):
        now = time.time()
        self._injection_attempts[client_ip].append(now)
        self._prune(self._injection_attempts[client_ip])
        count = len(self._injection_attempts[client_ip])
        if count >= self.injection_threshold:
            self._emit("prompt_injection", "high", client_ip, {
                "attempt_count": count,
                "window_seconds": 60,
                "latest_pattern": pattern,
            }, min(1.0, count / (self.injection_threshold * 2)))

    def _prune_tenant_entries(self, client_ip: str):
        now = time.time()
        self._tenant_access[client_ip] = [
            (tid, ts) for tid, ts in self._tenant_access[client_ip]
            if now - ts < 60
        ]

    def _check_auth_brute_force(self, client_ip: str):
        failures = len(self._auth_failures[client_ip])
        successes = len(self._auth_successes[client_ip])
        ratio = failures / max(1, successes + failures)

        if failures >= self.auth_brute_threshold:
            score = 0.6 + 0.4 * ratio
            self._emit("auth_brute_force", "critical", client_ip, {
                "failed_attempts": failures,
                "successful_attempts": successes,
                "failure_ratio": round(ratio, 3),
                "window_seconds": 60,
            }, score)
        elif failures + successes >= self.auth_spike_threshold and ratio > 0.7:
            self._emit("auth_brute_force", "high", client_ip, {
                "total_attempts": failures + successes,
                "failure_ratio": round(ratio, 3),
            }, 0.65)

    def _check_tool_anomaly(self, tool_name: str):
        """Detect if a tool is called much more frequently than baseline."""
        count = len(self._tool_usage[tool_name])
        baseline = self._tool_baseline.get(tool_name, 5)  # default 5 calls per 5 min
        if baseline == 0:
            self._tool_baseline[tool_name] = count
            return
        ratio = count / max(1, baseline)
        if ratio > self.tool_anomaly_threshold:
            self._emit("unusual_tool", "medium", "internal", {
                "tool": tool_name,
                "recent_calls": count,
                "window_seconds": 300,
                "baseline": baseline,
                "anomaly_ratio": round(ratio, 2),
            }, min(1.0, ratio / 10))

        # Update baseline with exponential moving average
        self._tool_baseline[tool_name] = 0.9 * baseline + 0.1 * count

    def _check_tenant_hop(self, client_ip: str):
        entries = self._tenant_access[client_ip]
        unique_tenants = set(tid for tid, _ in entries)
        if len(unique_tenants) >= self.tenant_hop_threshold:
            self._emit("tenant_hop", "medium", client_ip, {
                "unique_tenants": len(unique_tenants),
                "tenants": list(unique_tenants),
                "window_seconds": 60,
            }, min(1.0, len(unique_tenants) / (self.tenant_hop_threshold * 2)))

    def _emit(self, event_type: str, severity: str, source_ip: str, details: dict, score: float):
        event = IntrusionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            details=details,
            score=round(score, 3),
        )
        self._events.append(event)
        if len(self._events) > 1000:
            self._events = self._events[-500:]

        for cb in self._alert_callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def status(self) -> dict:
        """Current intrusion detection status."""
        now = time.time()
        active_threats = {}
        for ip, failures in self._auth_failures.items():
            recent = sum(1 for t in failures if now - t < 60)
            if recent >= self.auth_brute_threshold:
                active_threats[ip] = {"type": "auth_brute_force", "recent_failures": recent}

        for ip, denials in self._path_denials.items():
            recent = sum(1 for t in denials if now - t < 60)
            if recent >= self.path_denial_threshold:
                active_threats[ip] = active_threats.get(ip, {})
                active_threats[ip].update({"type": "path_traversal", "denial_count": recent})

        return {
            "active_threats": len(active_threats),
            "threat_detail": active_threats,
            "events_24h": len(self._events),
            "recent_events": [
                {"type": e.event_type, "severity": e.severity, "source": e.source_ip, "score": e.score}
                for e in self._events[-10:]
            ],
        }

    def recent_events(self, limit: int = 50) -> list[IntrusionEvent]:
        return self._events[-limit:]

    def get_metrics(self) -> dict[str, float]:
        """Export metrics for alert rule evaluation."""
        now = time.time()
        total_auth_failures = sum(
            sum(1 for t in failures if now - t < 60)
            for failures in self._auth_failures.values()
        )
        total_path_denials = sum(
            sum(1 for t in denials if now - t < 60)
            for denials in self._path_denials.values()
        )
        total_injections = sum(
            sum(1 for t in attempts if now - t < 60)
            for attempts in self._injection_attempts.values()
        )
        return {
            "intrusion_auth_failures": float(total_auth_failures),
            "intrusion_path_denials": float(total_path_denials),
            "intrusion_injections": float(total_injections),
            "intrusion_active_threats": float(len([
                ip for ip, f in self._auth_failures.items()
                if sum(1 for t in f if now - t < 60) >= self.auth_brute_threshold
            ])),
        }
