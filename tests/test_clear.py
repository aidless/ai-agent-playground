"""CLEAR 五维模块测试"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCostTracker:
    def test_estimate_cost(self):
        from agent.cost_tracker import CostTracker
        ct = CostTracker("deepseek-chat")
        cost = ct.estimate_cost(1000, 500)
        assert cost > 0
        assert cost < 0.01  # DeepSeek is cheap

    def test_budget_cap_allow(self):
        from agent.cost_tracker import BudgetCap
        cap = BudgetCap(5.0, 50.0)
        ok, reason = cap.check(2.0, 20.0)
        assert ok
        assert reason == "ok"

    def test_budget_cap_deny_daily(self):
        from agent.cost_tracker import BudgetCap
        cap = BudgetCap(5.0, 50.0)
        ok, reason = cap.check(6.0, 20.0)
        assert not ok
        assert "budget" in reason.lower() or "已超" in reason

    def test_summary(self):
        from agent.cost_tracker import CostTracker
        ct = CostTracker("deepseek-chat")
        s = ct.summary()
        assert "model" in s
        assert "today_cost_usd" in s
        assert "budget_daily_limit" in s


class TestReliability:
    def test_record_and_metrics(self):
        import agent.reliability as rmod
        old_dir = rmod.TRACKING_DIR
        with tempfile.TemporaryDirectory() as tmp:
            rmod.TRACKING_DIR = Path(tmp)
            try:
                rt = rmod.ReliabilityTracker()
                for i in range(7):
                    rt.record("code-review", f"task-{i}", success=(i < 5), latency_ms=100 + i * 10, tool_calls=2)

                metrics = rt.get_metrics()
                assert metrics["samples"] == 7
                assert metrics["success_rate"] == pytest.approx(5 / 7, abs=0.05)
                assert metrics["failures"] == 2
                assert metrics["consecutive_fails"] == 2
            finally:
                rmod.TRACKING_DIR = old_dir

    def test_consecutive_fail_stable(self):
        import agent.reliability as rmod
        old_dir = rmod.TRACKING_DIR
        with tempfile.TemporaryDirectory() as tmp:
            rmod.TRACKING_DIR = Path(tmp)
            try:
                rt = rmod.ReliabilityTracker()
                for i in range(5):
                    rt.record("test", f"t-{i}", success=True, latency_ms=50)

                metrics = rt.get_metrics()
                assert metrics["consecutive_fails"] == 0
                assert metrics["stability"] == "stable"
            finally:
                rmod.TRACKING_DIR = old_dir

    def test_degradation_detection(self):
        import agent.reliability as rmod
        old_dir = rmod.TRACKING_DIR
        with tempfile.TemporaryDirectory() as tmp:
            rmod.TRACKING_DIR = Path(tmp)
            try:
                rt = rmod.ReliabilityTracker()
                for i in range(4):
                    rt.record("deg-test", f"d-{i}", success=True, latency_ms=100)
                for i in range(4, 8):
                    rt.record("deg-test", f"d-{i}", success=False, latency_ms=100)

                alert = rt.detect_degradation()
                assert alert is not None
                assert "100%" in alert or "degrad" in alert.lower()
            finally:
                rmod.TRACKING_DIR = old_dir


class TestCLEARPanel:
    def test_report(self):
        from agent.cost_tracker import CostTracker
        from agent.governance import GovernancePanel
        from agent.reliability import ReliabilityTracker
        from observability.clear_metrics import CLEARPanel

        ct = CostTracker("deepseek-chat")
        gp = GovernancePanel()
        rt = ReliabilityTracker()

        for i in range(3):
            rt.record("test", f"t-{i}", success=True, latency_ms=200)

        panel = CLEARPanel(ct, gp, rt)
        report = panel.report()
        assert "COST" in report
        assert "LATENCY" in report
        assert "EFFICACY" in report
        assert "ASSURANCE" in report
        assert "RELIABILITY" in report
        assert "CLEAR Score" in report

    def test_to_json(self):
        from agent.cost_tracker import CostTracker
        from agent.governance import GovernancePanel
        from agent.reliability import ReliabilityTracker
        from observability.clear_metrics import CLEARPanel

        panel = CLEARPanel(CostTracker(), GovernancePanel(), ReliabilityTracker())
        data = panel.to_json()
        assert set(data.keys()) == {"timestamp", "cost", "latency", "efficacy", "assurance", "reliability"}

    def test_clear_score(self):
        from observability.clear_metrics import CLEARPanel
        score = CLEARPanel._score(
            {"success_rate": 0.9, "consistency_score": 0.85, "avg_latency_ms": 300},
            {"total": 80, "success_rate": 0.95},
            {"monthly_cost_usd": 20, "budget_monthly_limit": 50},
        )
        assert 5 <= score <= 10


if __name__ == "__main__":
    import pytest as pt
    sys.exit(pt.main([__file__, "-v", "--tb=short"]))
