"""Uptime & MTTR Tracker — server availability and recovery time metrics.

Tracks:
  - Service uptime: % of time the service is reachable
  - MTTR (Mean Time To Recovery): average time between failure and recovery
  - Consecutive failure tracking for auto-rollback triggers
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Incident:
    start: float
    end: float = 0.0
    reason: str = ""
    resolved: bool = False

    @property
    def duration_seconds(self) -> float:
        end = self.end if self.end > 0 else time.time()
        return end - self.start

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60


class UptimeTracker:
    """Tracks service uptime and MTTR.

    Usage:
        tracker = UptimeTracker()
        tracker.mark_healthy()    # Service is up
        tracker.mark_unhealthy("Connection refused")  # Service is down
        ...
        tracker.mark_healthy()    # Recovered — incident auto-closed
        print(tracker.status())
    """

    def __init__(self):
        self.start_time = time.time()
        self._healthy = True
        self._lock = threading.Lock()
        self._incidents: list[Incident] = []
        self._current_incident: Incident | None = None
        self._total_healthy_seconds = 0.0
        self._last_check = time.time()
        self._consecutive_failures = 0
        self._total_checks = 0
        self._failed_checks = 0

    def mark_healthy(self):
        with self._lock:
            now = time.time()
            if not self._healthy:
                self._healthy = True
                self._consecutive_failures = 0
                if self._current_incident:
                    self._current_incident.end = now
                    self._current_incident.resolved = True
                    self._incidents.append(self._current_incident)
                    self._current_incident = None
            self._total_healthy_seconds += now - self._last_check
            self._last_check = now
            self._total_checks += 1

    def mark_unhealthy(self, reason: str = ""):
        with self._lock:
            now = time.time()
            if self._healthy:
                self._total_healthy_seconds += now - self._last_check
            self._healthy = False
            self._consecutive_failures += 1
            if not self._current_incident:
                self._current_incident = Incident(start=now, reason=reason)
            self._last_check = now
            self._total_checks += 1
            self._failed_checks += 1

    @property
    def healthy(self) -> bool:
        return self._healthy

    @property
    def uptime_pct(self) -> float:
        with self._lock:
            total = time.time() - self.start_time
            if total <= 0:
                return 1.0
            healthy = self._total_healthy_seconds
            if self._healthy:
                healthy += time.time() - self._last_check
            return min(1.0, healthy / total)

    @property
    def mttr_minutes(self) -> float:
        """Mean Time To Recovery in minutes."""
        resolved = [i for i in self._incidents if i.resolved]
        if not resolved:
            return 0.0
        return sum(i.duration_minutes for i in resolved) / len(resolved)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def sla_compliant(self) -> bool:
        """Check if uptime meets 99.9% SLO."""
        return self.uptime_pct >= 0.999

    def status(self) -> dict:
        return {
            "uptime_pct": round(self.uptime_pct * 100, 4),
            "sla_compliant": self.sla_compliant,
            "healthy": self._healthy,
            "mttr_minutes": round(self.mttr_minutes, 1),
            "total_incidents": len(self._incidents),
            "consecutive_failures": self._consecutive_failures,
            "total_checks": self._total_checks,
            "failed_checks": self._failed_checks,
            "since": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
        }


_tracker: UptimeTracker | None = None


def get_uptime() -> UptimeTracker:
    global _tracker
    if _tracker is None:
        _tracker = UptimeTracker()
    return _tracker
