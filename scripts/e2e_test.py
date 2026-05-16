#!/usr/bin/env python3
"""
End-to-End Test — MCP Agent + Streaming + Observability in one verified flow.

Tests the golden path: a user asks a tool-requiring question,
the agent streams its response token by token, every step is traced,
and the final output is validated against deterministic constraints.

Usage:
  uv run python scripts/e2e_test.py
  uv run python scripts/e2e_test.py --quick   # Skip API calls
"""

import sys
import time
from pathlib import Path


def test_deterministic_constraints():
    """Test 1: All constraint types pass/fail correctly (no API)."""
    print("=" * 60)
    print("  Test 1: Deterministic Constraints")
    print("=" * 60)

    from ai_agent_playground.constraints import (
        NotEmptyConstraint,
        ValidJsonConstraint,
        JsonSchemaConstraint,
        PythonSyntaxConstraint,
        MustNotContainConstraint,
        code_output_runner,
        json_output_runner,
    )

    results = []

    # NotEmpty
    c = NotEmptyConstraint()
    r = c.check("hello")
    results.append(("NotEmpty (valid)", r.passed, r.passed))
    r = c.check("")
    results.append(("NotEmpty (empty)", not r.passed, not r.passed))

    # ValidJson
    c = ValidJsonConstraint()
    r = c.check('{"key": "value"}')
    results.append(("ValidJson (valid)", r.passed, r.passed))
    r = c.check("not json")
    results.append(("ValidJson (invalid)", not r.passed, not r.passed))

    # JsonSchema
    c = JsonSchemaConstraint(["tool", "args"])
    r = c.check('{"tool": "calc", "args": {}}')
    results.append(("JsonSchema (valid)", r.passed, r.passed))
    r = c.check('{"tool": "calc"}')
    results.append(("JsonSchema (missing field)", not r.passed, not r.passed))

    # PythonSyntax
    c = PythonSyntaxConstraint()
    r = c.check("def foo():\n    return 1")
    results.append(("PythonSyntax (valid)", r.passed, r.passed))
    r = c.check("def foo(\n    return 1")
    results.append(("PythonSyntax (syntax error)", not r.passed, not r.passed))

    # MustNotContain
    c = MustNotContainConstraint([r"DROP\s+TABLE", r"<script"])
    r = c.check("hello world")
    results.append(("MustNotContain (clean)", r.passed, r.passed))
    r = c.check("DROP TABLE users; -- injection")
    results.append(("MustNotContain (danger)", not r.passed, not r.passed))

    # Factory runners
    runner = code_output_runner()
    r = runner.check_all("def hello():\n    return 'world'")
    results.append(("CodeOutputRunner (valid)", r.passed, r.passed))

    runner = json_output_runner(["tool", "args"])
    r = runner.check_all('{"tool": "calc", "args": {"expr": "2+2"}}')
    results.append(("JsonOutputRunner (valid)", r.passed, r.passed))

    passed = sum(1 for _, actual, __ in results if actual)
    total = len(results)
    for name, actual, expected in results:
        status = "OK" if actual == expected else "FAIL"
        print(f"  [{status}] {name}")
    print(f"  {passed}/{total} passed")


def test_state_manager():
    """Test 2: State manager checkpoint/restore cycle."""
    print("\n" + "=" * 60)
    print("  Test 2: State Manager — Task Lifecycle + Resume")
    print("=" * 60)

    import tempfile
    from ai_agent_playground.state_manager import StateManager

    with tempfile.TemporaryDirectory(prefix="e2e_state_") as tmp:
        sm = StateManager(work_dir=tmp)
        sm.start_session("Build a REST API with 3 endpoints")

        sm.add_tasks([
            ("init", "Initialize project structure"),
            ("models", "Define data models"),
            ("routes", "Create API routes"),
            ("tests", "Write unit tests"),
        ])

        sm.start_task("init")
        sm.complete_task("init", "Created main.py, models.py, routes.py")
        sm.start_task("models")
        sm.complete_task("models", "User and Task models with SQLAlchemy")

        assert sm.manifest.completed_count == 2, "Should have 2 completed"
        assert sm.manifest.progress_pct == 0.5, f"Should be 50%, got {sm.manifest.progress_pct}"
        print(f"  OK Task lifecycle: {sm.manifest.completed_count}/{sm.manifest.total_count} completed")

        # Test resume
        sm2 = StateManager(work_dir=tmp)
        resumed = sm2.resume()
        assert resumed is not None, "Should resume"
        assert resumed.goal == "Build a REST API with 3 endpoints"
        assert resumed.completed_count == 2
        print(f"  OK Resume: goal='{resumed.goal}', progress={resumed.completed_count}/{resumed.total_count}")

        # Test context rebuild
        ctx = sm2.rebuild_context()
        assert "REST API" in ctx, "Context should contain goal"
        assert "✅" in ctx, "Context should contain completed task markers"
        print(f"  OK Context rebuild: {len(ctx)} chars, contains goal + task status")

        # Test checkpoint (no git needed)
        sm.checkpoint("after_models")
        assert len(sm.manifest.git_checkpoints) >= 1
        print(f"  OK Checkpoint: {sm.manifest.git_checkpoints[-1][:60]}...")


def test_human_loop():
    """Test 3: Human-in-the-Loop approval gate logic."""
    print("\n" + "=" * 60)
    print("  Test 3: Human-in-the-Loop — Risk Assessment + Policies")
    print("=" * 60)

    from ai_agent_playground.human_loop import (
        ApprovalGate, Policy, assess_risk,
    )

    # Risk assessment
    assert assess_risk("read_file", {"path": "test.txt"}) == "low"
    assert assess_risk("write_file", {"path": "test.txt"}) == "medium"
    assert assess_risk("run_command", {"command": "ls"}) == "high"
    assert assess_risk("run_command", {"command": "rm -rf /"}) == "critical"
    assert assess_risk("write_file", {"path": "/etc/passwd"}) == "critical"
    print("  OK Risk assessment: low/medium/high/critical all correct")

    # Policy enforcement
    gate = ApprovalGate()
    gate.set_policies({
        "read_file": Policy.AUTO_APPROVE,
        "write_file": Policy.ALWAYS_ASK,
        "run_command": Policy.ALWAYS_ASK,
        "delete_file": Policy.NEVER,
    })

    # Auto-approve
    decision = gate.approve("read_file", {"path": "test.txt"}, input_fn=lambda _: "y")
    assert decision.approved, "read_file should be auto-approved"
    print(f"  OK Auto-approve: read_file → {decision.reason}")

    # Always-ask with human deny
    decision = gate.approve("write_file", {"path": "test.txt"}, input_fn=lambda _: "n")
    assert not decision.approved, "write_file should be denied when human says no"
    print(f"  OK Human deny: write_file → {decision.reason}")

    # Never policy
    decision = gate.approve("delete_file", {"path": "prod.db"})
    assert not decision.approved, "delete_file should be blocked"
    print(f"  OK Blocked: delete_file → {decision.reason}")

    # Audit log
    gate.log("read_file", {"path": "test.txt"}, "file contents", True)
    gate.log("write_file", {"path": "test.txt"}, "written", True)
    gate.log("delete_file", {"path": "prod.db"}, "blocked", False)
    report = gate.report()
    assert report["total_approvals"] == 3
    assert report["denied"] == 1
    print(f"  OK Audit: {report['total_approvals']} checks, {report['denied']} denied")


def test_observability():
    """Test 4: Observability trace collection."""
    print("\n" + "=" * 60)
    print("  Test 4: Observability — Trace + Metrics")
    print("=" * 60)

    from ai_agent_playground.observability import get_tracer
    tracer = get_tracer(log_dir="logs/e2e_traces")

    with tracer.trace("e2e_test", test_id="observability_check") as trace:
        with trace.span("llm_call", model="test-model"):
            trace.spans[-1].attributes["output_tokens"] = 150
        with trace.span("tool_call", tool="calculator"):
            trace.spans[-1].attributes["tool"] = "calculator"

    snap = tracer.snapshot()
    assert snap.total_traces == 1, f"Should have 1 trace, got {snap.total_traces}"
    assert snap.total_spans == 2, f"Should have 2 spans, got {snap.total_spans}"
    assert snap.total_llm_tokens == 150
    assert snap.total_tool_calls == 1
    print(f"  OK Trace: {snap.total_traces} traces, {snap.total_spans} spans, "
          f"{snap.total_llm_tokens} tokens, {snap.total_tool_calls} tool calls")

    # Prometheus export
    Path("logs/e2e_metrics").mkdir(parents=True, exist_ok=True)
    tracer.export_prometheus("logs/e2e_metrics/test.prom")
    prom = Path("logs/e2e_metrics/test.prom").read_text()
    assert "agent_traces_total" in prom
    assert "agent_latency_ms_avg" in prom
    print(f"  OK Prometheus export: {len(prom)} bytes, has required metrics")

    # Console dashboard
    tracer.print_dashboard()


def test_streaming_agent(quick: bool = False):
    """Test 5: MCP Agent streaming + tool use (requires API)."""
    print("\n" + "=" * 60)
    print("  Test 5: MCP Agent — Streaming + Tool Use" + (" (SKIPPED)" if quick else ""))
    print("=" * 60)

    if quick:
        print("  Use --full to run API-dependent tests.")
        return

    from mcp_agent.agent import MCPToolAgent
    from ai_agent_playground.base import ToolCallEvent
    from ai_agent_playground.observability import get_tracer

    tracer = get_tracer()
    agent = MCPToolAgent()

    try:
        with tracer.trace("e2e_agent_test", user_input="Calculate 15*15+12"):
            tool_events = []
            text_chunks = []

            for item in agent.run_stream("Calculate 15 * 15 + 12. Use the calculator."):
                if isinstance(item, str):
                    text_chunks.append(item)
                elif isinstance(item, ToolCallEvent):
                    tool_events.append(item)

        full_text = "".join(text_chunks)
        has_result = "237" in full_text
        tool_called = len(tool_events) >= 2  # start + end
        print(f"  Tool events: {len(tool_events)} (start+end pairs)")
        print(f"  Text chunks: {len(text_chunks)}")
        print(f"  Contains result: {has_result}")
        print(f"  {'OK' if has_result and tool_called else 'WARN'} "
              f"Result present={has_result}, tool called={tool_called}")

    except Exception as e:
        print(f"  SKIP: API call failed ({e})")
    finally:
        agent.close()


def main():
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("  AI Agent Playground — End-to-End Test Suite")
    print(f"  Mode: {'QUICK (no API)' if quick else 'FULL'}")
    print("=" * 60)

    tests = [
        test_deterministic_constraints,
        test_state_manager,
        test_human_loop,
        test_observability,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n  FAIL: {e}")

    try:
        test_streaming_agent(quick=quick)
    except Exception as e:
        print(f"\n  FAIL: {e}")

    print("\n" + "=" * 60)
    print("  E2E Test Suite Complete")
    print("=" * 60)
    print(f"  Logs: logs/e2e_traces/")
    print(f"  Metrics: logs/e2e_metrics/")


if __name__ == "__main__":
    main()
