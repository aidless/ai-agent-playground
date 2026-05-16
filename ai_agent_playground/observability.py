"""
LLM Observability — tracing, metrics, structured logging.

The 70% of AI engineering that happens AFTER the demo works.

Every request through the system is traced:
  - Latency breakdown (LLM inference / tool execution / retrieval / total)
  - Token counts (input / output / total)
  - Tool call stats (which tools, success/failure, durations)
  - Error tracking (where in the pipeline did it fail?)

Output formats:
  - Structured JSON lines (for ELK / Splunk / file analysis)
  - Prometheus text format (for scraping into Grafana)
  - Console summary (for development debugging)

Architecture:
  Trace → Spans → Events
  MetricsCollector → aggregates across traces → exportable

Usage:
    from ai_agent_playground.observability import get_tracer

    tracer = get_tracer()
    with tracer.trace("user_question", user_id="demo") as trace:
        with trace.span("llm_call", model="deepseek-v4"):
            response = llm.send(...)
        with trace.span("tool_call", tool="calculator"):
            result = tools["calculator"](...)
    # Trace auto-logged on exit
"""

import json
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ============================================================
#  Data structures
# ============================================================


@dataclass
class SpanEvent:
    """One timed unit of work in a trace."""

    name: str
    start_time: float
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # "ok" | "error"
    error_message: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "status": self.status,
        }


@dataclass
class Trace:
    """One complete user interaction from start to finish."""

    trace_id: str
    root_name: str
    start_time: float
    end_time: float = 0.0
    spans: list[SpanEvent] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def llm_spans(self) -> list[SpanEvent]:
        return [s for s in self.spans if s.name == "llm_call"]

    @property
    def tool_spans(self) -> list[SpanEvent]:
        return [s for s in self.spans if s.name == "tool_call"]

    @property
    def total_llm_tokens(self) -> int:
        return sum(s.attributes.get("output_tokens", 0) for s in self.llm_spans)

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_spans)

    @property
    def tool_success_rate(self) -> float:
        if not self.tool_spans:
            return 1.0
        ok = sum(1 for s in self.tool_spans if s.status == "ok")
        return ok / len(self.tool_spans)

    @contextmanager
    def span(self, name: str, **attrs):
        """Create a timed span within this trace."""
        span = SpanEvent(name=name, start_time=time.time(), attributes=attrs)
        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.error_message = str(e)
            raise
        finally:
            span.end_time = time.time()
            self.spans.append(span)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "root": self.root_name,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "llm_calls": len(self.llm_spans),
            "total_llm_tokens": self.total_llm_tokens,
            "tool_calls": self.total_tool_calls,
            "tool_success_rate": round(self.tool_success_rate, 2),
            "spans": [s.to_dict() for s in self.spans],
        }


# ============================================================
#  Metrics Collector
# ============================================================


@dataclass
class MetricsSnapshot:
    """Aggregated metrics over a time window."""

    total_traces: int = 0
    total_spans: int = 0
    error_count: int = 0
    total_llm_tokens: int = 0
    total_tool_calls: int = 0
    tool_error_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics across traces."""

    def __init__(self):
        self.traces: list[Trace] = []
        self._latencies: list[float] = []

    def record(self, trace: Trace):
        self.traces.append(trace)
        self._latencies.append(trace.duration_ms)
        # Keep bounded
        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-5000:]

    def snapshot(self) -> MetricsSnapshot:
        """Compute aggregated metrics from recorded traces."""
        if not self.traces:
            return MetricsSnapshot()

        latencies = sorted(self._latencies)
        n = len(latencies)

        return MetricsSnapshot(
            total_traces=n,
            total_spans=sum(len(t.spans) for t in self.traces),
            error_count=sum(1 for t in self.traces if any(
                s.status == "error" for s in t.spans
            )),
            total_llm_tokens=sum(t.total_llm_tokens for t in self.traces),
            total_tool_calls=sum(t.total_tool_calls for t in self.traces),
            tool_error_count=sum(
                sum(1 for s in t.tool_spans if s.status == "error")
                for t in self.traces
            ),
            avg_latency_ms=sum(latencies) / n,
            p50_latency_ms=latencies[int(n * 0.5)],
            p95_latency_ms=latencies[int(n * 0.95)],
            p99_latency_ms=latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
        )

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Compatible with Prometheus textfile collector:
          https://prometheus.io/docs/instrumenting/exposition_formats/
        """
        snap = self.snapshot()
        lines = [
            "# HELP agent_traces_total Total number of agent traces",
            f"agent_traces_total {snap.total_traces}",
            "# HELP agent_spans_total Total number of spans",
            f"agent_spans_total {snap.total_spans}",
            "# HELP agent_errors_total Total number of traces with errors",
            f"agent_errors_total {snap.error_count}",
            "# HELP agent_llm_tokens_total Total LLM tokens consumed",
            f"agent_llm_tokens_total {snap.total_llm_tokens}",
            "# HELP agent_tool_calls_total Total tool calls",
            f"agent_tool_calls_total {snap.total_tool_calls}",
            "# HELP agent_tool_errors_total Total tool call errors",
            f"agent_tool_errors_total {snap.tool_error_count}",
            "# HELP agent_latency_ms_avg Average end-to-end latency in ms",
            f"agent_latency_ms_avg {snap.avg_latency_ms:.1f}",
            "# HELP agent_latency_ms_p50 P50 latency in ms",
            f"agent_latency_ms_p50 {snap.p50_latency_ms:.1f}",
            "# HELP agent_latency_ms_p95 P95 latency in ms",
            f"agent_latency_ms_p95 {snap.p95_latency_ms:.1f}",
            "# HELP agent_latency_ms_p99 P99 latency in ms",
            f"agent_latency_ms_p99 {snap.p99_latency_ms:.1f}",
        ]
        return "\n".join(lines) + "\n"

    def clear(self):
        self.traces.clear()
        self._latencies.clear()


# ============================================================
#  Tracer — the main entry point
# ============================================================


class LLMTracer:
    """Main tracer that creates traces and collects metrics.

    Each trace = one user interaction.
    Each span = one step in the pipeline (LLM call, tool call, retrieval).
    """

    def __init__(self, log_dir: str | None = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics = MetricsCollector()
        self._on_trace_end: list[Callable] = []

    def on_trace_end(self, callback: Callable):
        """Register a callback invoked after each trace completes."""
        self._on_trace_end.append(callback)

    @contextmanager
    def trace(self, name: str, **attrs):
        """Create a trace. Auto-logs on exit.

        Usage:
            with tracer.trace("agent_query", user_id="demo") as trace:
                with trace.span("llm_call", model="deepseek-v4"):
                    ...
        """
        trace = Trace(
            trace_id=str(uuid.uuid4())[:8],
            root_name=name,
            start_time=time.time(),
            attributes=attrs,
        )
        try:
            yield trace
            self.metrics.record(trace)
        finally:
            trace.end_time = time.time()
            self._log_trace(trace)
            for cb in self._on_trace_end:
                cb(trace)

    def _log_trace(self, trace: Trace):
        """Write trace to structured log."""
        entry = trace.to_dict()
        entry["timestamp"] = time.time()

        # Console output (dev mode)
        snap = self.metrics.snapshot()
        tool_str = ""
        if trace.tool_spans:
            tools = [s.attributes.get("tool", "?") for s in trace.tool_spans]
            tool_str = f" | tools: {','.join(tools)} ({trace.tool_success_rate:.0%})"

        llm_str = ""
        if trace.llm_spans:
            llm_str = f" | tokens: {trace.total_llm_tokens}"

        print(
            f"[trace {trace.trace_id}] {trace.root_name} "
            f"{trace.duration_ms:.0f}ms{llm_str}{tool_str}"
        )

        # File output (structured JSON lines)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / "traces.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def snapshot(self) -> MetricsSnapshot:
        return self.metrics.snapshot()

    def export_prometheus(self, path: str):
        """Write Prometheus metrics to a file for node_exporter textfile collector."""
        Path(path).write_text(self.metrics.to_prometheus(), encoding="utf-8")

    def print_dashboard(self):
        """Print a quick console dashboard."""
        snap = self.snapshot()
        if snap.total_traces == 0:
            print("No traces recorded yet.")
            return

        error_rate = snap.error_count / snap.total_traces * 100
        tool_error_rate = (
            snap.tool_error_count / snap.total_tool_calls * 100
            if snap.total_tool_calls else 0
        )

        print("\n" + "=" * 50)
        print("  LLM OBSERVABILITY DASHBOARD")
        print("=" * 50)
        print(f"  Traces:     {snap.total_traces}")
        print(f"  Error rate: {error_rate:.1f}% ({snap.error_count} errors)")
        print(f"  Latency:    avg={snap.avg_latency_ms:.0f}ms "
              f"p50={snap.p50_latency_ms:.0f}ms "
              f"p95={snap.p95_latency_ms:.0f}ms "
              f"p99={snap.p99_latency_ms:.0f}ms")
        print(f"  Tokens:     {snap.total_llm_tokens} total")
        print(f"  Tool calls: {snap.total_tool_calls} "
              f"(error rate: {tool_error_rate:.1f}%)")
        print("=" * 50 + "\n")


# ============================================================
#  Global singleton
# ============================================================

_tracer: LLMTracer | None = None


def get_tracer(log_dir: str | None = None) -> LLMTracer:
    global _tracer
    if _tracer is None:
        _tracer = LLMTracer(log_dir=log_dir or "logs/traces")
    return _tracer
