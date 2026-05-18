"""CLEAR 五维指标面板

    成本 (Cost)     — cost_tracker 每日/月费用
    延迟 (Latency)  — monitoring 响应时间分布
    效能 (Efficacy) — reliability 任务成功率+恢复力
    保障 (Assurance)— governance 审计+权限+熔断
    可靠性(Reliability)— 一致性+稳定性+退化检测

用法:
    from observability.clear_metrics import CLEARPanel
    panel = CLEARPanel()
    print(panel.report())
"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class CLEARPanel:
    """五维指标统一面板"""

    def __init__(self, cost_tracker=None, governance=None, reliability=None):
        from agent.cost_tracker import CostTracker
        from agent.governance import GovernancePanel
        from agent.reliability import ReliabilityTracker

        self.cost = cost_tracker or CostTracker()
        self.governance = governance or GovernancePanel()
        self.reliability = reliability or ReliabilityTracker()
        self._start_time = datetime.now()

    def report(self) -> str:
        """生成 CLEAR 五维报告"""
        cost_summary = self.cost.summary()
        audit_stats = self.governance.audit.stats()
        breaker_statuses = {
            k: v.status()
            for k, v in self.governance.breakers.items()
        }
        reliability_metrics = self.reliability.get_metrics()
        degradation = self.reliability.detect_degradation()

        lines = [
            "=" * 50,
            "CLEAR Metrics Report",
            "=" * 50,
            f"Uptime: {(datetime.now() - self._start_time).total_seconds() / 3600:.1f}h",
            "",
            # ── Cost ──
            "--- COST ---",
            f"  Today: ${cost_summary.get('today_cost_usd', 0):.4f} ({cost_summary.get('today_requests', 0)} reqs, {cost_summary.get('today_tokens', 0)} tokens)",
            f"  Monthly: ${cost_summary.get('monthly_cost_usd', 0):.4f} (limit: ${cost_summary.get('budget_monthly_limit', 50):.0f})",
            f"  Budget: daily=${cost_summary.get('budget_daily_limit', 5):.0f}, monthly=${cost_summary.get('budget_monthly_limit', 50):.0f}",
            "",
            # ── Latency ──
            "--- LATENCY ---",
            f"  Avg: {reliability_metrics.get('avg_latency_ms', 0):.0f}ms (from {reliability_metrics.get('samples', 0)} samples)",
            f"  SLO: < 2000ms (P95)",
            "",
            # ── Efficacy ──
            "--- EFFICACY ---",
            f"  Success Rate: {reliability_metrics.get('success_rate', 1.0):.1%} ({reliability_metrics.get('samples', 0)} tasks)",
            f"  Trend: {reliability_metrics.get('trend', 'n/a')}",
            f"  Consecutive Fails: {reliability_metrics.get('consecutive_fails', 0)}",
            f"  Consistency: {reliability_metrics.get('consistency_score', 0):.2f}",
            "",
            # ── Assurance ──
            "--- ASSURANCE ---",
            f"  Audit Records: {audit_stats.get('total', 0)} today",
            f"  Success Rate: {audit_stats.get('success_rate', 1.0):.1%}",
            f"  Breakers: {len(breaker_statuses)} active",
        ]

        for tool, status in breaker_statuses.items():
            lines.append(f"    {tool}: {status['state']} (fails={status['failures']})")

        lines.extend([
            f"  Permission Levels: 4 (readonly/restricted/execute/admin)",
            "",
            # ── Reliability ──
            "--- RELIABILITY ---",
            f"  Stability: {reliability_metrics.get('stability', 'unknown')}",
            f"  Degradation: {'NONE' if not degradation else '⚠️  ' + degradation}",
            f"  Samples: {reliability_metrics.get('samples', 0)}",
            "",
            # ── Overall ──
            "--- OVERALL ---",
            f"  CLEAR Score: {self._score(reliability_metrics, audit_stats, cost_summary)}/10",
            "=" * 50,
        ])

        return "\n".join(lines)

    def to_json(self) -> dict:
        cost = self.cost.summary()
        audit = self.governance.audit.stats()
        rel = self.reliability.get_metrics()

        return {
            "timestamp": datetime.now().isoformat(),
            "cost": {
                "today_usd": cost.get("today_cost_usd", 0),
                "monthly_usd": cost.get("monthly_cost_usd", 0),
                "requests_today": cost.get("today_requests", 0),
            },
            "latency": {
                "avg_ms": rel.get("avg_latency_ms", 0),
            },
            "efficacy": {
                "success_rate": rel.get("success_rate", 1.0),
                "trend": rel.get("trend", "n/a"),
                "consecutive_fails": rel.get("consecutive_fails", 0),
            },
            "assurance": {
                "audit_records": audit.get("total", 0),
                "audit_success_rate": audit.get("success_rate", 1.0),
                "permission_levels": 4,
            },
            "reliability": {
                "stability": rel.get("stability", "unknown"),
                "consistency": rel.get("consistency_score", 0),
                "degradation": self.reliability.detect_degradation(),
            },
        }

    @staticmethod
    def _score(rel: dict, audit: dict, cost: dict) -> float:
        """综合 CLEAR 评分（简单加权）"""
        scores = []
        # Cost: 预算利用率 50%-80% 为最佳
        monthly_used = cost.get("monthly_cost_usd", 0)
        monthly_limit = cost.get("budget_monthly_limit", 50)
        cost_ratio = monthly_used / monthly_limit if monthly_limit > 0 else 0
        if cost_ratio < 0.3:
            scores.append(10)  # 低成本
        elif cost_ratio < 0.7:
            scores.append(8)
        elif cost_ratio < 1.0:
            scores.append(6)
        else:
            scores.append(3)  # 超预算

        # Latency
        avg_ms = rel.get("avg_latency_ms", 0)
        if avg_ms < 500:
            scores.append(10)
        elif avg_ms < 2000:
            scores.append(7)
        else:
            scores.append(4)

        # Efficacy
        scores.append(round(rel.get("success_rate", 0.5) * 10))

        # Assurance
        audit_total = audit.get("total", 0)
        if audit_total > 50:
            scores.append(9)
        elif audit_total > 10:
            scores.append(7)
        else:
            scores.append(5)

        # Reliability
        scores.append(round(rel.get("consistency_score", 0.5) * 10))

        return round(sum(scores) / len(scores), 1)
