"""三点式集成验证 — 测试三个方向的模块

方向一: orchestrator (任务拆解/拓扑排序/并行)
方向二: evaluator (自评估框架)
方向三: governance (审计/权限/熔断)
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestOrchestrator:
    """测试多 Agent 编排器逻辑层（使用 dict-based subtasks）"""

    def test_topological_sort_parallel(self):
        from agent.orchestrator import AgentOrchestrator

        tasks = [
            {"id": "1", "description": "设计架构", "role": "planner", "depends_on": []},
            {"id": "2", "description": "写前端", "role": "developer", "depends_on": ["1"]},
            {"id": "3", "description": "写后端", "role": "developer", "depends_on": ["1"]},
            {"id": "4", "description": "写测试", "role": "tester", "depends_on": ["2", "3"]},
        ]
        orch = AgentOrchestrator(None)
        batches = orch._topological_sort(tasks)

        assert len(batches) == 3
        assert [t["id"] for t in batches[0]] == ["1"]
        assert {t["id"] for t in batches[1]} == {"2", "3"}
        assert [t["id"] for t in batches[2]] == ["4"]

    def test_topological_sort_all_parallel(self):
        from agent.orchestrator import AgentOrchestrator

        tasks = [
            {"id": "a", "description": "A", "role": "researcher", "depends_on": []},
            {"id": "b", "description": "B", "role": "researcher", "depends_on": []},
            {"id": "c", "description": "C", "role": "researcher", "depends_on": []},
        ]
        orch = AgentOrchestrator(None)
        batches = orch._topological_sort(tasks)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_topological_sort_chain(self):
        from agent.orchestrator import AgentOrchestrator

        tasks = [
            {"id": "1", "description": "S1", "role": "planner", "depends_on": []},
            {"id": "2", "description": "S2", "role": "developer", "depends_on": ["1"]},
            {"id": "3", "description": "S3", "role": "tester", "depends_on": ["2"]},
        ]
        orch = AgentOrchestrator(None)
        batches = orch._topological_sort(tasks)
        assert len(batches) == 3

    def test_task_result_dataclass(self):
        from agent.crew_agent import TaskResult

        r = TaskResult(task_id="x1", agent_name="dev", content="OK", success=True)
        assert r.task_id == "x1"
        assert r.agent_name == "dev"
        assert r.success is True
        assert r.error is None


class TestGovernance:
    """测试治理三件套：审计 + 权限 + 熔断"""

    def test_permission_levels(self):
        from agent.governance import PermissionManager, PermissionLevel

        pm = PermissionManager()
        assert pm.get_level("read_file") == PermissionLevel.READONLY
        assert pm.get_level("write_file") == PermissionLevel.RESTRICTED
        assert pm.get_level("run_command") == PermissionLevel.EXECUTE
        assert pm.get_level("delete_file") == PermissionLevel.ADMIN
        assert not pm.requires_confirmation("read_file")
        assert pm.requires_confirmation("run_command")

    def test_permission_allow(self):
        from agent.governance import PermissionManager, PermissionLevel

        pm = PermissionManager()
        assert not pm.allow("read_file", PermissionLevel.EXECUTE)
        assert pm.allow("run_command", PermissionLevel.READONLY)

    def test_circuit_breaker_states(self):
        from agent.governance import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.before_call()

        cb.on_failure("e1")
        cb.on_failure("e2")
        assert cb.state == CircuitState.CLOSED
        cb.on_failure("e3")
        assert cb.state == CircuitState.OPEN
        assert not cb.before_call()

        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 1
        assert cb.before_call()

    def test_circuit_breaker_recovery(self):
        from agent.governance import CircuitBreaker, CircuitState as CS

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)
        cb.on_failure("e1")
        cb.on_failure("e2")
        assert cb.state == CS.OPEN

        time.sleep(0.15)
        cb.before_call()
        assert cb.state == CS.HALF_OPEN

        cb.on_success()
        assert cb.state == CS.CLOSED

    def test_audit_logger(self):
        from agent.governance import AuditLogger, AuditEntry
        import agent.governance as gv

        old_dir = gv.AUDIT_DIR
        with tempfile.TemporaryDirectory() as tmp:
            gv.AUDIT_DIR = Path(tmp)
            try:
                log = AuditLogger()
                log.log(AuditEntry(
                    operation="tool_call", tool="read_file",
                    args={"path": "/tmp/test.txt"},
                    result_summary="hello world", duration_ms=12.3,
                    success=True, trace_id="tr-001",
                ))
                log.log(AuditEntry(
                    operation="tool_call", tool="write_file",
                    args={"path": "/tmp/test.txt"},
                    result_summary="", duration_ms=5.0,
                    success=False, error="permission denied",
                    trace_id="tr-002",
                ))
                records = log.query(limit=10)
                assert len(records) == 2
                assert records[0]["success"] is True
                assert records[1]["success"] is False
                assert records[1]["error"] == "permission denied"

                stats = log.stats()
                assert stats["total"] == 2
                assert stats["success_rate"] == 0.5
            finally:
                gv.AUDIT_DIR = old_dir

    def test_governance_panel(self):
        from agent.governance import GovernancePanel
        import agent.governance as gv

        old_dir = gv.AUDIT_DIR
        with tempfile.TemporaryDirectory() as tmp:
            gv.AUDIT_DIR = Path(tmp)
            try:
                panel = GovernancePanel()

                def good_func():
                    return "result-ok"

                result = asyncio.run(
                    panel.wrap_tool_call("read_file", {"path": "test.txt"}, good_func, "tr-003")
                )
                assert result == "result-ok"

                def bad_func():
                    raise RuntimeError("boom")

                with pytest.raises(RuntimeError, match="boom"):
                    asyncio.run(
                        panel.wrap_tool_call("run_command", {"cmd": "rm -rf /"}, bad_func, "tr-004")
                    )

                records = panel.audit.query(limit=10)
                assert len(records) == 2
                assert records[0]["success"] is True
                assert records[1]["success"] is False

                report = panel.report()
                assert "安全治理报告" in report
            finally:
                gv.AUDIT_DIR = old_dir

    def test_permission_grant_revoke(self):
        from agent.governance import PermissionManager, PermissionLevel

        pm = PermissionManager()
        assert pm.get_level("write_file") == PermissionLevel.RESTRICTED
        pm.grant("write_file", PermissionLevel.ADMIN)
        assert pm.get_level("write_file") == PermissionLevel.ADMIN
        pm.revoke("write_file")
        assert pm.get_level("write_file") == PermissionLevel.RESTRICTED


class TestEvaluator:
    """测试 Agent 自评估器"""

    def test_eval_report_dataclass(self):
        from agent.specialized.evaluator import EvalReport, EvalDimension

        dims = {
            "task_effectiveness": EvalDimension(score=7.0, assessment="Good", suggestion="Add retry"),
        }
        report = EvalReport(
            overall=7.0, dimensions=dims, summary="Decent.",
            top_issue="No audit", top_strength="Fast",
        )
        assert report.overall == 7.0
        assert report.top_issue == "No audit"

    def test_json_parsing(self):
        from agent.specialized.evaluator import AgentEvaluator

        json_str = '{"overall":8,"dimensions":{},"summary":"x","top_issue":"","top_strength":""}'
        result = AgentEvaluator._parse_json(json_str)
        assert result["overall"] == 8


class TestIntegration:
    """三点联动：编排 → 治理包装 → 评估"""

    def test_governance_wraps_orchestration(self):
        from agent.governance import GovernancePanel
        from agent.orchestrator import AgentOrchestrator
        import agent.governance as gv

        old_dir = gv.AUDIT_DIR
        with tempfile.TemporaryDirectory() as tmp:
            gv.AUDIT_DIR = Path(tmp)
            try:
                panel = GovernancePanel()
                orch = AgentOrchestrator(None)

                tasks = [
                    {"id": "1", "description": "审查代码", "role": "reviewer", "depends_on": []},
                    {"id": "2", "description": "运行测试", "role": "tester", "depends_on": ["1"]},
                ]

                def execute_task():
                    return "task-done"

                for task in tasks:
                    result = asyncio.run(
                        panel.wrap_tool_call("read_file", {"task_id": task["id"]}, execute_task, "trace-xyz")
                    )
                    assert result == "task-done"

                records = panel.audit.query(limit=100)
                assert len(records) == 2

                batches = orch._topological_sort(tasks)
                assert len(batches) == 2
            finally:
                gv.AUDIT_DIR = old_dir

    def test_permission_report_includes_tools(self):
        from agent.governance import PermissionManager

        pm = PermissionManager()
        report = pm.report()
        assert "read_file" in report
        assert "run_command" in report


if __name__ == "__main__":
    import pytest as pt
    sys.exit(pt.main([__file__, "-v", "--tb=short"]))
