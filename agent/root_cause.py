"""Auto Root Cause Analysis — traces failures to specific step or module.

P3 requirement: 80% of failed tasks can auto-locate the failing step/module.

Strategy:
  1. Instrument every step with latency + status
  2. On failure, collect the trace graph
  3. Walk the graph to find the first failing node
  4. Classify: tool_error / llm_timeout / data_quality / config / network
  5. Suggest remediation action

This reduces MTTR from manual investigation (~30min) to automated (~5s).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RCA_DIR = Path(__file__).resolve().parent.parent / "memory" / "rca"


class FailureCategory(str, Enum):
    TOOL_ERROR = "tool_error"          # Tool execution failed
    LLM_TIMEOUT = "llm_timeout"        # LLM call timed out
    LLM_ERROR = "llm_error"            # LLM returned error
    DATA_QUALITY = "data_quality"      # Input data malformed
    CONFIG_ERROR = "config_error"      # Misconfiguration
    NETWORK_ERROR = "network_error"    # Network unreachable
    PERMISSION = "permission"          # Access denied
    CIRCUIT_OPEN = "circuit_open"      # Circuit breaker tripped
    BUDGET_EXCEED = "budget_exceed"    # Cost budget exceeded
    UNKNOWN = "unknown"


REMEDIATION = {
    FailureCategory.TOOL_ERROR: "Check tool implementation and parameters. Retry with corrected input.",
    FailureCategory.LLM_TIMEOUT: "Increase timeout or reduce prompt size. Consider model fallback.",
    FailureCategory.LLM_ERROR: "Check API key, rate limits, and model availability. Retry with backoff.",
    FailureCategory.DATA_QUALITY: "Validate input data schema. Check for missing fields or type errors.",
    FailureCategory.CONFIG_ERROR: "Verify configuration values. Roll back to last known-good config.",
    FailureCategory.NETWORK_ERROR: "Check network connectivity. Retry with exponential backoff.",
    FailureCategory.PERMISSION: "Verify identity and role. Request elevated permissions if needed.",
    FailureCategory.CIRCUIT_OPEN: "Wait for circuit breaker recovery. Check upstream dependency.",
    FailureCategory.BUDGET_EXCEED: "Review cost allocation. Increase budget or optimize prompts.",
    FailureCategory.UNKNOWN: "Collect full trace and escalate to human operator.",
}


@dataclass
class StepTrace:
    """One step in a task execution trace."""
    step_id: str
    name: str                        # e.g., "llm_call", "tool:read_file"
    status: str = "pending"          # pending / running / ok / error / timeout
    latency_ms: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)
    children: list["StepTrace"] = field(default_factory=list)
    started_at: float = 0.0
    category: FailureCategory | None = None

    @property
    def is_failing(self) -> bool:
        return self.status in ("error", "timeout")

    @property
    def first_failing_child(self) -> "StepTrace | None":
        for child in self.children:
            if child.is_failing:
                return child
            if failing := child.first_failing_child:
                return failing
        return None


@dataclass
class RCAReport:
    """Root cause analysis report."""
    trace_id: str
    task: str
    failure_category: FailureCategory
    failing_step: str = ""
    root_cause: str = ""
    evidence: list[str] = field(default_factory=list)
    remediation: str = ""
    confidence: float = 0.0
    total_latency_ms: float = 0.0
    step_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RootCauseAnalyzer:
    """Analyzes task execution traces to find root causes.

    Usage:
        analyzer = RootCauseAnalyzer()
        with analyzer.trace("build_api", "Build a REST API") as trace:
            with trace.step("llm_call"):
                response = llm.call(...)
            with trace.step("tool_call", tool="read_file"):
                result = registry.execute(...)
        # On failure, analyzer automatically generates RCA report
    """

    def __init__(self):
        RCA_DIR.mkdir(parents=True, exist_ok=True)
        self._current_trace: StepTrace | None = None
        self._reports: list[RCAReport] = []

    def trace(self, name: str, task: str):
        """Context manager for a task trace."""
        return _TraceContext(self, name, task)

    def analyze(self, trace: StepTrace, task: str) -> RCAReport:
        """Analyze a completed trace and generate RCA report."""
        trace_id = trace.step_id

        # Find the failing step
        if trace.is_failing:
            failing = trace
        else:
            failing = trace.first_failing_child

        if not failing:
            # All steps succeeded — but maybe called on failure anyway
            return RCAReport(
                trace_id=trace_id, task=task,
                failure_category=FailureCategory.UNKNOWN,
                root_cause="No failing step found in trace",
                confidence=0.0,
            )

        # Classify the failure
        category = self._classify(failing)
        root_cause = self._build_root_cause(trace, failing, category)
        evidence = self._collect_evidence(trace, failing)
        remediation = REMEDIATION.get(category, REMEDIATION[FailureCategory.UNKNOWN])

        # Confidence: how certain are we about this classification?
        confidence = self._compute_confidence(failing, category)

        report = RCAReport(
            trace_id=trace_id,
            task=task,
            failure_category=category,
            failing_step=failing.name,
            root_cause=root_cause,
            evidence=evidence,
            remediation=remediation,
            confidence=confidence,
            total_latency_ms=trace.latency_ms,
            step_count=self._count_steps(trace),
        )

        self._reports.append(report)
        self._save(report)
        return report

    def _classify(self, step: StepTrace) -> FailureCategory:
        """Classify a failing step into a failure category."""
        error_lower = step.error.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return FailureCategory.LLM_TIMEOUT
        if "connection" in error_lower or "network" in error_lower or "connect" in error_lower:
            return FailureCategory.NETWORK_ERROR
        if "permission" in error_lower or "access denied" in error_lower or "unauthorized" in error_lower:
            return FailureCategory.PERMISSION
        if "circuit" in error_lower or "breaker" in error_lower:
            return FailureCategory.CIRCUIT_OPEN
        if "budget" in error_lower or "cost" in error_lower:
            return FailureCategory.BUDGET_EXCEED
        if "key" in error_lower and "api" in error_lower:
            return FailureCategory.LLM_ERROR
        if "json" in error_lower or "parse" in error_lower or "schema" in error_lower:
            return FailureCategory.DATA_QUALITY
        if "config" in error_lower:
            return FailureCategory.CONFIG_ERROR
        if step.name.startswith("tool:"):
            return FailureCategory.TOOL_ERROR
        if "llm" in step.name:
            return FailureCategory.LLM_ERROR

        return FailureCategory.UNKNOWN

    def _build_root_cause(self, trace: StepTrace, failing: StepTrace, category: FailureCategory) -> str:
        """Build human-readable root cause description."""
        parts = [f"Failure in step '{failing.name}': {failing.error}"]

        # Walk upward to find context
        parent_context = []
        current = trace
        while current and current != failing:
            if current.metadata:
                parent_context.append(f"  Context [{current.name}]: {json.dumps(current.metadata, default=str)[:200]}")
            for child in current.children:
                if child == failing or self._contains(child, failing):
                    current = child
                    break
            else:
                break

        if parent_context:
            parts.extend(parent_context[-3:])

        return " | ".join(parts)

    def _contains(self, parent: StepTrace, target: StepTrace) -> bool:
        if parent == target:
            return True
        return any(self._contains(c, target) for c in parent.children)

    def _collect_evidence(self, trace: StepTrace, failing: StepTrace) -> list[str]:
        """Collect evidence chain leading to the failure."""
        evidence = [f"Step: {failing.name} | Status: {failing.status} | Latency: {failing.latency_ms:.0f}ms"]
        if failing.error:
            evidence.append(f"Error: {failing.error[:300]}")
        if failing.metadata:
            evidence.append(f"Metadata: {json.dumps(failing.metadata, default=str)[:300]}")
        return evidence

    def _compute_confidence(self, step: StepTrace, category: FailureCategory) -> float:
        """Compute confidence score for the classification."""
        base = 0.5

        # Strong signals
        if step.error:
            base += 0.2
        if step.metadata:
            base += 0.1
        if category != FailureCategory.UNKNOWN:
            base += 0.15
        if step.latency_ms > 0:
            base += 0.05

        return min(1.0, base)

    def _count_steps(self, trace: StepTrace) -> int:
        count = 1
        for child in trace.children:
            count += self._count_steps(child)
        return count

    def _save(self, report: RCAReport):
        """Persist RCA report for pattern analysis."""
        report_file = RCA_DIR / f"rca-{report.trace_id}.json"
        report_file.write_text(
            json.dumps(report.__dict__, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def get_recurring_failures(self, limit: int = 10) -> list[dict]:
        """Find patterns across multiple RCA reports."""
        by_category = {}
        for f in RCA_DIR.glob("rca-*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                cat = data.get("failure_category", "unknown")
                by_category.setdefault(cat, []).append(data)
            except Exception:
                continue

        recurring = sorted(by_category.items(), key=lambda x: -len(x[1]))
        return [
            {
                "category": cat,
                "count": len(items),
                "latest": max(items, key=lambda x: x.get("created_at", "")),
                "avg_confidence": sum(i.get("confidence", 0) for i in items) / len(items),
            }
            for cat, items in recurring[:limit]
        ]

    def stats(self) -> dict:
        """RCA statistics."""
        reports = list(RCA_DIR.glob("rca-*.json"))
        if not reports:
            return {"total_analyses": 0, "auto_diagnosis_rate": 0.0}

        total = len(reports)
        classified = 0
        for f in reports:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("failure_category") != "unknown":
                    classified += 1
            except Exception:
                continue

        return {
            "total_analyses": total,
            "classified": classified,
            "auto_diagnosis_rate": round(classified / total, 3) if total else 0,
            "target_p3_80pct": classified / total >= 0.8 if total else False,
        }


class _TraceContext:
    """Context manager for creating execution traces."""

    def __init__(self, analyzer: RootCauseAnalyzer, name: str, task: str):
        self.analyzer = analyzer
        self.name = name
        self.task = task
        self.trace: StepTrace | None = None

    def __enter__(self):
        import uuid
        self.trace = StepTrace(
            step_id=str(uuid.uuid4())[:8],
            name=self.name,
            started_at=time.perf_counter(),
            status="running",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace:
            self.trace.latency_ms = (time.perf_counter() - self.trace.started_at) * 1000
            if exc_type:
                self.trace.status = "error"
                self.trace.error = str(exc_val) if exc_val else "Unknown error"
            else:
                self.trace.status = "ok"
            self.analyzer.analyze(self.trace, self.task)
        return False

    def step(self, name: str, **metadata):
        """Create a child step within this trace."""
        return _StepContext(self.trace, name, metadata)


class _StepContext:
    def __init__(self, parent: StepTrace | None, name: str, metadata: dict):
        import uuid
        self.step = StepTrace(
            step_id=str(uuid.uuid4())[:8],
            name=name,
            started_at=time.perf_counter(),
            status="running",
            metadata=metadata,
        )
        self.parent = parent

    def __enter__(self):
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.latency_ms = (time.perf_counter() - self.step.started_at) * 1000
        if exc_type:
            self.step.status = "error"
            self.step.error = str(exc_val) if exc_val else "Unknown"
        else:
            self.step.status = "ok"
        if self.parent:
            self.parent.children.append(self.step)
        return False
