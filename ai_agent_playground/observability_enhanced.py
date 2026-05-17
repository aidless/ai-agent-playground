"""Enhanced Observability — 跨Agent链路追踪 + 实时告警。"""

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentSpan:
    span_id: str
    parent_id: str | None
    agent_name: str
    operation: str
    start_time: float
    end_time: float = 0.0
    status: str = "ok"
    error_message: str = ""
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class EnhancedTracer:
    def __init__(self, log_dir: str | None = None):
        self.log_dir = log_dir
        self._traces: dict[str, list[AgentSpan]] = defaultdict(list)
        self._lock = threading.RLock()
        self._current_trace_id: threading.local = threading.local()
        self._on_span_end: list[Callable] = []

    def start_trace(self, name: str, trace_id: str | None = None) -> str:
        import uuid
        trace_id = trace_id or str(uuid.uuid4())[:8]
        self._current_trace_id.trace_id = trace_id
        return trace_id

    @property
    def current_trace_id(self) -> str | None:
        return getattr(self._current_trace_id, "trace_id", None)

    def start_span(self, agent_name: str, operation: str = "run", parent_id: str | None = None, **attrs) -> "AgentSpanContext":
        import uuid
        parent_id = parent_id or self.current_trace_id
        span_id = str(uuid.uuid4())[:8]
        span = AgentSpan(span_id=span_id, parent_id=parent_id, agent_name=agent_name, operation=operation, start_time=time.time(), attributes=attrs)
        return AgentSpanContext(self, span)

    def record_span(self, span: AgentSpan):
        with self._lock:
            if span.parent_id:
                self._traces[span.parent_id].append(span)
            for cb in self._on_span_end:
                try:
                    cb(span)
                except Exception as e:
                    print(f"[EnhancedTracer] Callback error: {e}")

    def on_span_end(self, callback: Callable):
        self._on_span_end.append(callback)

    def get_trace(self, trace_id: str) -> list[AgentSpan]:
        with self._lock:
            return self._traces.get(trace_id, []).copy()

    def get_stats(self) -> dict:
        with self._lock:
            all_spans = [s for spans in self._traces.values() for s in spans]
            if not all_spans:
                return {"total_traces": 0, "total_spans": 0}
            durations = [s.duration_ms for s in all_spans]
            errors = sum(1 for s in all_spans if s.status == "error")
            return {"total_traces": len(self._traces), "total_spans": len(all_spans), "avg_duration_ms": sum(durations) / len(durations), "max_duration_ms": max(durations), "error_count": errors, "error_rate": errors / len(all_spans)}

    def print_dashboard(self):
        stats = self.get_stats()
        print("\n=== Enhanced Observability ===")
        print(f"Traces: {stats['total_traces']} | Spans: {stats['total_spans']}")
        print(f"Avg duration: {stats.get('avg_duration_ms', 0):.1f}ms")
        print(f"Error rate: {stats.get('error_rate', 0)*100:.1f}%")


class AgentSpanContext:
    def __init__(self, tracer: EnhancedTracer, span: AgentSpan):
        self._tracer = tracer
        self._span = span

    def __enter__(self):
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._span.end_time = time.time()
        if exc_type:
            self._span.status = "error"
            self._span.error_message = str(exc_val)
        self._tracer.record_span(self._span)


class AlertManager:
    def __init__(self, error_rate_threshold: float = 0.1, latency_p95_threshold_ms: float = 5000, check_interval_sec: float = 60):
        self.error_rate_threshold = error_rate_threshold
        self.latency_p95_threshold_ms = latency_p95_threshold_ms
        self.check_interval_sec = check_interval_sec
        self._recent_traces: list[dict] = []
        self._lock = threading.RLock()
        self._alerts: list[dict] = []
        self._callbacks: list[Callable] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def record_trace(self, trace_data: dict):
        with self._lock:
            self._recent_traces.append(trace_data)
            if len(self._recent_traces) > 1000:
                self._recent_traces = self._recent_traces[-500:]

    def add_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def check(self) -> list[dict]:
        with self._lock:
            if len(self._recent_traces) < 10:
                return []
            alerts = []
            errors = sum(1 for t in self._recent_traces if t.get("status") == "error")
            error_rate = errors / len(self._recent_traces)
            if error_rate > self.error_rate_threshold:
                alerts.append({"type": "error_rate", "message": f"错误率 {error_rate*100:.1f}% 超过阈值 {self.error_rate_threshold*100:.1f}%", "severity": "critical" if error_rate > 0.3 else "warning", "value": error_rate})
            latencies = [t.get("duration_ms", 0) for t in self._recent_traces]
            latencies.sort()
            p95 = latencies[int(len(latencies) * 0.95)]
            if p95 > self.latency_p95_threshold_ms:
                alerts.append({"type": "latency", "message": f"P95延迟 {p95:.0f}ms 超过阈值 {self.latency_p95_threshold_ms}ms", "severity": "critical" if p95 > self.latency_p95_threshold_ms * 2 else "warning", "value": p95})
            self._alerts = alerts
            return alerts

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _check_loop(self):
        while self._running:
            time.sleep(self.check_interval_sec)
            alerts = self.check()
            for alert in alerts:
                for cb in self._callbacks:
                    try:
                        cb(alert)
                    except Exception as e:
                        print(f"[AlertManager] Callback error: {e}")

    def print_alerts(self):
        alerts = self._alerts or self.check()
        if not alerts:
            print("✅ 无告警")
            return
        print("\n🚨 Alerts:")
        for alert in alerts:
            icon = "🔴" if alert["severity"] == "critical" else "🟡"
            print(f"  {icon} [{alert['type']}] {alert['message']}")


def trace_agent_call(tracer: EnhancedTracer, agent_name: str):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("trace_id", tracer.current_trace_id)
            with tracer.start_span(agent_name, parent_id=trace_id) as span:
                try:
                    result = func(*args, **kwargs)
                    span.status = "ok"
                    return result
                except Exception as e:
                    span.status = "error"
                    span.error_message = str(e)
                    raise
        return wrapper
    return decorator


_enhanced_tracer: EnhancedTracer | None = None
_alert_manager: AlertManager | None = None


def get_enhanced_tracer(log_dir: str | None = None) -> EnhancedTracer:
    global _enhanced_tracer
    if _enhanced_tracer is None:
        _enhanced_tracer = EnhancedTracer(log_dir=log_dir or "logs/traces_enhanced")
    return _enhanced_tracer


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager