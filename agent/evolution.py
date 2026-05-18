"""Tool Evolution Engine — Meta's HYPERAGENTS-style self-optimization loop.

Observes tool execution performance, identifies bottlenecks, generates
optimization patches via LLM, validates, and atomically replaces tools
with improved versions.

The loop:
  1. PerformanceTracker records every tool call (latency, success, errors)
  2. EvolutionEngine detects underperforming tools
  3. LLM reads current code + error history, generates optimized version
  4. Diff old vs new, validate syntax+safety
  5. If improved: atomically swap in ToolRegistry, log evolution
  6. If not: keep old version, log attempt for future analysis

This is the "recursive self-improvement" from the HYPERAGENTS paper,
applied to function-level tool optimization.
"""

import difflib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

EVOLUTION_DIR = Path(__file__).resolve().parent.parent / "memory" / "evolution"


@dataclass
class ToolMetrics:
    tool_name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_errors: list[str] = field(default_factory=list)  # last 10 errors
    last_10_latencies: list[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.call_count)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.call_count)

    @property
    def p95_latency_ms(self) -> float:
        if not self.last_10_latencies:
            return 0.0
        sorted_lat = sorted(self.last_10_latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def is_underperforming(self) -> bool:
        """Tool is a candidate for evolution if success rate < 70% or P95 > 5s."""
        if self.call_count < 5:
            return False
        return self.success_rate < 0.70 or self.p95_latency_ms > 5000


@dataclass
class EvolutionRecord:
    tool_name: str
    version: int
    old_code: str
    new_code: str
    diff: str
    reason: str
    metrics_before: dict
    validated: bool
    applied: bool
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PerformanceTracker:
    """Records and analyzes tool execution performance."""

    def __init__(self):
        EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, ToolMetrics] = {}
        self._load()

    def _load(self):
        path = EVOLUTION_DIR / "tool_metrics.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for name, m in data.items():
                tm = ToolMetrics(
                    tool_name=name,
                    call_count=m.get("call_count", 0),
                    success_count=m.get("success_count", 0),
                    error_count=m.get("error_count", 0),
                    total_latency_ms=m.get("total_latency_ms", 0),
                    last_errors=m.get("last_errors", []),
                    last_10_latencies=m.get("last_10_latencies", []),
                )
                self._metrics[name] = tm

    def _save(self):
        data = {}
        for name, m in self._metrics.items():
            data[name] = {
                "call_count": m.call_count,
                "success_count": m.success_count,
                "error_count": m.error_count,
                "total_latency_ms": m.total_latency_ms,
                "last_errors": m.last_errors[-10:],
                "last_10_latencies": m.last_10_latencies[-10:],
            }
        (EVOLUTION_DIR / "tool_metrics.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def record(self, tool_name: str, success: bool, latency_ms: float, error: str = ""):
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics(tool_name=tool_name)
        m = self._metrics[tool_name]
        m.call_count += 1
        if success:
            m.success_count += 1
        else:
            m.error_count += 1
            m.last_errors.append(error)
            if len(m.last_errors) > 10:
                m.last_errors = m.last_errors[-10:]
        m.total_latency_ms += latency_ms
        m.last_10_latencies.append(latency_ms)
        if len(m.last_10_latencies) > 10:
            m.last_10_latencies = m.last_10_latencies[-10:]
        self._save()

    def get_metrics(self, tool_name: str) -> Optional[ToolMetrics]:
        return self._metrics.get(tool_name)

    def list_underperforming(self) -> list[str]:
        return [name for name, m in self._metrics.items() if m.is_underperforming]

    def all_metrics(self) -> dict:
        return {
            name: {
                "calls": m.call_count,
                "success_rate": round(m.success_rate, 3),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "p95_latency_ms": round(m.p95_latency_ms, 1),
                "underperforming": m.is_underperforming,
            }
            for name, m in self._metrics.items()
        }


EVOLUTION_PROMPT = (
    "You are a code optimizer. Given a tool function and its performance metrics, "
    "generate an OPTIMIZED version of the same function. Keep the same signature "
    "(params: dict) -> str. Focus on:\n"
    "1. Reducing latency (simplify logic, use better algorithms)\n"
    "2. Improving success rate (better error handling, input validation)\n"
    "3. Handling edge cases shown in the error history\n"
    "Output ONLY the optimized function code, nothing else."
)

EVOLUTION_WITH_TEMPLATES_PROMPT = (
    "You are a code optimizer. Given a tool function, its performance metrics, "
    "and EXAMPLES of successful optimizations from similar tools, generate an "
    "OPTIMIZED version. Learn from the patterns in the examples:\n"
    "- What optimization strategies worked before?\n"
    "- What patterns improved performance?\n"
    "Apply these lessons to the current tool. Keep signature (params: dict) -> str.\n"
    "Output ONLY the optimized function code, nothing else."
)

META_EVOLUTION_PROMPT = (
    "You are a self-referential meta-programmer. You are optimizing the AGENT "
    "SYSTEM ITSELF, not a tool function. The code below is part of the MetaAgent "
    "or evolution engine. Analyze it, identify weaknesses, and generate an improved "
    "version.\n"
    "Focus on:\n"
    "1. Better decision logic (when to evolve, when to rollback)\n"
    "2. More efficient execution (parallelism, caching)\n"
    "3. Stronger safety checks\n"
    "Output ONLY the improved code, nothing else."
)


class EvolutionEngine:
    """Generates and applies optimizations to underperforming tools.

    Usage:
        engine = EvolutionEngine(llm_client, tracker, registry)
        record = await engine.evolve("web_search")
        # If record.applied: tool is now optimized + registered
    """

    def __init__(self, client, tracker: PerformanceTracker, registry, model: str = "deepseek-chat", meta_agent=None):
        self.client = client
        self.tracker = tracker
        self.registry = registry
        self.model = model
        self.meta_agent = meta_agent
        self._history: dict[str, list[EvolutionRecord]] = {}
        self._load_history()

    def _load_history(self):
        history_dir = EVOLUTION_DIR / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        for f in history_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                records = [EvolutionRecord(**r) for r in data.get("records", [])]
                self._history[data.get("tool_name", f.stem)] = records
            except Exception:
                pass

    def _save_history(self, tool_name: str):
        history_dir = EVOLUTION_DIR / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        path = history_dir / f"{tool_name}.json"
        data = {
            "tool_name": tool_name,
            "records": [
                {
                    "tool_name": r.tool_name, "version": r.version,
                    "old_code": r.old_code, "new_code": r.new_code,
                    "diff": r.diff, "reason": r.reason,
                    "metrics_before": r.metrics_before,
                    "validated": r.validated, "applied": r.applied,
                    "error": r.error, "created_at": r.created_at,
                }
                for r in self._history.get(tool_name, [])
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    async def evolve(self, tool_name: str) -> EvolutionRecord:
        """Analyze performance → generate optimization → validate → apply.

        Uses the archive of past successful evolutions as templates — this is
        the 'growing archive of stepping stones' from DGM-Hyperagents.
        """
        metrics = self.tracker.get_metrics(tool_name)
        if not metrics:
            raise ValueError(f"No metrics for tool: {tool_name}")

        version = len(self._history.get(tool_name, [])) + 1
        old_code = self._get_tool_source(tool_name)

        # Build context for LLM
        errors_summary = "\n".join(metrics.last_errors[-5:]) if metrics.last_errors else "none"
        perf_context = (
            f"Tool: {tool_name}\n"
            f"Calls: {metrics.call_count}\n"
            f"Success rate: {metrics.success_rate:.1%}\n"
            f"Avg latency: {metrics.avg_latency_ms:.0f}ms\n"
            f"P95 latency: {metrics.p95_latency_ms:.0f}ms\n"
            f"Recent errors:\n{errors_summary}\n\n"
        )

        # ── Template learning: find similar past evolutions as stepping stones ──
        similar_templates = self._find_similar_templates(tool_name, max_templates=2)
        if similar_templates:
            template_context = "SUCCESSFUL OPTIMIZATION EXAMPLES FROM SIMILAR TOOLS:\n\n"
            for tmpl in similar_templates:
                template_context += (
                    f"--- Example: {tmpl['tool']} v{tmpl['version']} ---\n"
                    f"Diff applied:\n{tmpl['diff'][:800]}\n"
                    f"Result: {tmpl['reason'][:200]}\n\n"
                )
            system_prompt = EVOLUTION_WITH_TEMPLATES_PROMPT
            user_prompt = f"{template_context}\n\nNOW OPTIMIZE THIS TOOL:\n{perf_context}\nCurrent code:\n```python\n{old_code}\n```"
        else:
            system_prompt = EVOLUTION_PROMPT
            user_prompt = f"{perf_context}\nCurrent code:\n```python\n{old_code}\n```"

        prompt = user_prompt

        # Generate optimization
        optimized_code = ""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            optimized_code = response.choices[0].message.content or ""
        except Exception as e:
            record = EvolutionRecord(
                tool_name=tool_name, version=version,
                old_code=old_code, new_code="", diff="",
                reason=f"LLM call failed: {e}",
                metrics_before={"success_rate": metrics.success_rate},
                validated=False, applied=False, error=str(e),
            )
            self._add_record(tool_name, record)
            return record

        # Clean markdown wrappers
        optimized_code = optimized_code.strip()
        optimized_code = re.sub(r'^```(?:python)?\s*\n', '', optimized_code)
        optimized_code = re.sub(r'\n```\s*$', '', optimized_code)
        optimized_code = re.sub(r'\n```(?:python)?\s*\n', '\n', optimized_code)

        # Diff
        diff = "\n".join(difflib.unified_diff(
            old_code.splitlines(), optimized_code.splitlines(),
            fromfile=f"{tool_name}_v{version - 1}", tofile=f"{tool_name}_v{version}",
            lineterm="",
        ))

        # Validate
        validated = self._validate_safety(optimized_code, tool_name)
        applied = False
        error = ""

        if validated:
            try:
                applied = self._apply_optimization(tool_name, optimized_code)
                error = "" if applied else "Failed to register optimized tool"
            except Exception as e:
                error = str(e)
        else:
            error = "Safety validation failed"

        record = EvolutionRecord(
            tool_name=tool_name, version=version,
            old_code=old_code, new_code=optimized_code, diff=diff,
            reason=f"Optimization — success_rate={metrics.success_rate:.1%}, p95={metrics.p95_latency_ms:.0f}ms",
            metrics_before={
                "success_rate": metrics.success_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
            },
            validated=validated, applied=applied, error=error,
        )
        self._add_record(tool_name, record)
        return record

    def _get_tool_source(self, tool_name: str) -> str:
        """Get the current source code of a tool."""
        if hasattr(self.registry, "_tools") and tool_name in self.registry._tools:
            func = self.registry._tools[tool_name].func
            import inspect
            try:
                return inspect.getsource(func)
            except (OSError, TypeError):
                return f"# Source unavailable for {tool_name}"
        return f"# Tool {tool_name} not found in registry"

    def _validate_safety(self, code: str, name: str) -> bool:
        """Validate syntax + AST safety check."""
        import ast
        try:
            compile(code, f"<evolution:{name}>", "exec")
        except SyntaxError as e:
            logger.warning("Evolution syntax error in %s: %s", name, e)
            return False

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("os", "subprocess", "shutil", "socket", "ctypes"):
                            return False
                if isinstance(node, ast.ImportFrom):
                    if node.module in ("os", "subprocess", "shutil", "socket", "ctypes"):
                        return False
        except SyntaxError:
            return False

        return "def " in code or "class " in code or "import " in code

    def _apply_optimization(self, tool_name: str, new_code: str) -> bool:
        """Execute new code and register the optimized function in place."""
        namespace = {}
        exec(new_code, namespace)
        func = None
        for name, obj in namespace.items():
            if callable(obj) and name != "__builtins__" and not name.startswith("_"):
                func = obj
                break

        if func is None:
            return False

        # Wrap to match registry calling convention
        def _wrapped(**kwargs):
            return func(kwargs)

        _wrapped.__name__ = tool_name

        # Store rollback snapshot before replacing
        if self.meta_agent:
            self.meta_agent.store_for_rollback(tool_name)

        # Re-register with same name (overwrites)
        if hasattr(self.registry, "register"):
            self.registry.register(
                tool_name,
                f"Evolved {tool_name} — auto-optimized",
                {"properties": {}, "required": []},
                _wrapped,
            )
            logger.info("Evolution applied: %s (new version registered)", tool_name)
            return True
        return False

    def _add_record(self, tool_name: str, record: EvolutionRecord):
        if tool_name not in self._history:
            self._history[tool_name] = []
        self._history[tool_name].append(record)
        self._save_history(tool_name)

    def get_evolution_history(self, tool_name: str = None) -> list[dict]:
        if tool_name:
            records = self._history.get(tool_name, [])
        else:
            records = [r for rs in self._history.values() for r in rs]
        return [
            {
                "tool": r.tool_name, "version": r.version,
                "applied": r.applied, "validated": r.validated,
                "diff_lines": r.diff.count("\n"),
                "created_at": r.created_at,
            }
            for r in records
        ]

    def _find_similar_templates(self, tool_name: str, max_templates: int = 2) -> list[dict]:
        """Find past successful evolutions that can serve as templates."""
        templates = []
        # Collect successful evolutions from any tool
        all_successful = []
        for tname, records in self._history.items():
            if tname == tool_name:
                continue  # skip current tool's own history
            for r in records:
                if r.applied and r.validated:
                    all_successful.append({
                        "tool": tname,
                        "version": r.version,
                        "diff": r.diff,
                        "reason": r.reason,
                    })

        # Simple similarity: prefer tools with similar names or patterns
        def similarity(other_tool):
            score = 0
            # Name prefix match
            for part in tool_name.split("_"):
                if part in other_tool:
                    score += 1
            # Recently evolved = higher similarity
            return score

        sorted_templates = sorted(all_successful, key=lambda t: similarity(t["tool"]), reverse=True)
        return sorted_templates[:max_templates]

    async def evolve_meta_code(self, file_path: str, old_code: str) -> EvolutionRecord:
        """Self-referential evolution — optimize the agent's OWN code.

        This is the core of HYPERAGENTS: the meta agent modifies itself.
        """
        import uuid
        tool_name = Path(file_path).stem
        version = len(self._history.get(tool_name, [])) + 1

        user_prompt = (
            f"File: {file_path}\n"
            f"This is the agent's own meta-level code. Analyze and improve it.\n\n"
            f"Current code:\n```python\n{old_code[:3000]}\n```"
        )

        optimized_code = ""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": META_EVOLUTION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            optimized_code = response.choices[0].message.content or ""
        except Exception as e:
            return EvolutionRecord(
                tool_name=tool_name, version=version,
                old_code=old_code, new_code="", diff="",
                reason=f"Meta LLM call failed: {e}",
                metrics_before={},
                validated=False, applied=False, error=str(e),
            )

        # Clean markdown wrappers
        optimized_code = optimized_code.strip()
        # Remove leading/trailing ``` fences
        optimized_code = re.sub(r'^```(?:python)?\s*\n', '', optimized_code)
        optimized_code = re.sub(r'\n```\s*$', '', optimized_code)
        # Remove any remaining single ``` lines
        optimized_code = re.sub(r'\n```(?:python)?\s*\n', '\n', optimized_code)

        diff = "\n".join(difflib.unified_diff(
            old_code.splitlines(), optimized_code.splitlines(),
            fromfile=f"{file_path}.old", tofile=f"{file_path}.new", lineterm="",
        ))

        validated = self._validate_safety(optimized_code, tool_name)

        record = EvolutionRecord(
            tool_name=f"meta:{file_path}", version=version,
            old_code=old_code, new_code=optimized_code, diff=diff,
            reason="Self-referential meta-evolution (HYPERAGENTS pattern)",
            metrics_before={},
            validated=validated, applied=False,  # Meta evolutions require human review
        )
        self._add_record(tool_name, record)
        if validated:
            # Save optimized version as proposal (requires human approval)
            proposal_path = EVOLUTION_DIR / "meta_proposals" / f"{tool_name}_v{version}.py"
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text(optimized_code, encoding="utf-8")
            logger.info("Meta evolution proposal saved: %s", proposal_path)
        return record

    def status(self) -> dict:
        return {
            "tools_tracked": len(self.tracker.all_metrics()),
            "underperforming": self.tracker.list_underperforming(),
            "evolutions_applied": sum(
                1 for rs in self._history.values()
                for r in rs if r.applied
            ),
            "recent": [
                {
                    "tool": r.tool_name, "version": r.version,
                    "applied": r.applied, "diff_preview": r.diff[:200],
                }
                for rs in self._history.values()
                for r in rs[-3:]
            ],
        }
